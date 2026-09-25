"""Streaming benchmark of four model settings on twenty scored scenarios.

Run: python scripts/compare_luna_streaming.py
Preview without API calls: python scripts/compare_luna_streaming.py --dry-run

The script makes 80 requests by default (20 cases x 4 configurations).
Each case has a reproducible, roughly 15k-token user prompt. Results and full
answers are written to benchmark_results/ for inspection.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from luna_cases import KINDS as BASIC_KINDS, make_case as make_basic_case
from luna_advanced_cases import KINDS as ADVANCED_KINDS, make_case as make_advanced_case


CONFIGS = {
    "4o-mini-temp0": ("gpt-4o-mini", None, 0),
    "5.6-luna-none": ("gpt-5.6-luna", "none", None),
    "6-luna-low": ("gpt-6-luna", "low", None),
    "6-luna-medium": ("gpt-6-luna", "medium", None),
}
ROOT = Path(__file__).resolve().parent.parent


def api_key_from_env_file(path: Path) -> str | None:
    """Read only OPENAI_API_KEY from a simple .env file, without extra packages."""
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        key, separator, value = line.strip().partition("=")
        if not separator or key.removeprefix("export ").strip() != "OPENAI_API_KEY":
            continue
        value = value.strip()
        if value.startswith(("'", '"')) and value.endswith(value[0]):
            value = value[1:-1]
        else:
            value = value.split(" #", 1)[0].strip()
        return value or None
    return None


def grade(answer: str, expected: dict[str, str] | list[str]) -> dict:
    try:
        parsed = json.loads(answer.strip())
    except json.JSONDecodeError:
        return {"correct": 0, "total": 50 if isinstance(expected, list) else len(expected),
                "json_valid": False}
    if isinstance(expected, list):
        valid = isinstance(parsed, list) and all(isinstance(x, str) for x in parsed)
        selected = set(parsed) if valid else set()
        universe = {f"MCQ-{i:03d}" for i in range(1, 51)}
        exact_format = valid and len(parsed) == len(selected) and selected <= universe
        details = {qid: (qid in selected) == (qid in expected) for qid in sorted(universe)}
        return {"correct": sum(details.values()) if exact_format else 0,
                "total": 50, "details": details, "json_valid": valid,
                "format_valid": exact_format, "true_ids_returned": sorted(selected)}
    if not isinstance(parsed, dict):
        return {"correct": 0, "total": len(expected), "json_valid": False,
                "details": {k: False for k in expected}}
    details = {k: isinstance(parsed, dict) and parsed.get(k) == v for k, v in expected.items()}
    return {"correct": sum(details.values()), "total": len(expected), "details": details, "json_valid": True}


def run_one(client, config: str, prompt: str, max_output_tokens: int, show_stream: bool = False) -> dict:
    model, effort, temperature = CONFIGS[config]
    start = time.perf_counter()
    first_event = first_text = last_text = None
    chunks: list[str] = []
    deltas: list[dict] = []
    response = None
    request = {
        "model": model,
        "input": prompt,
        "max_output_tokens": max_output_tokens,
        "stream": True,
        "store": False,
    }
    if temperature is not None:
        request["temperature"] = temperature
    else:
        request["reasoning"] = {"effort": effort}
    stream = client.responses.create(**request)
    for event in stream:
        now = time.perf_counter()
        if first_event is None:
            first_event = now
        if event.type == "response.output_text.delta" and event.delta:
            if first_text is None:
                first_text = now
            last_text = now
            chunks.append(event.delta)
            deltas.append({"at_s": round(now-start, 4), "text": event.delta})
            if show_stream:
                print(event.delta, end="", flush=True)
        elif event.type == "response.completed":
            response = event.response
        elif event.type in ("response.failed", "response.incomplete"):
            response = event.response
    end = time.perf_counter()
    if show_stream:
        print()
    usage = getattr(response, "usage", None)
    input_tokens = getattr(usage, "input_tokens", None)
    output_tokens = getattr(usage, "output_tokens", None)
    output_details = getattr(usage, "output_tokens_details", None)
    reasoning_tokens = getattr(output_details, "reasoning_tokens", None)
    if (temperature is not None or effort == "none") and reasoning_tokens is None:
        reasoning_tokens = 0
    visible_tokens = (
        output_tokens - reasoning_tokens
        if output_tokens is not None and reasoning_tokens is not None
        else None
    )
    visible_seconds = end - first_text if first_text is not None else None
    active_stream_seconds = last_text - first_text if first_text is not None and last_text is not None else None
    input_details = getattr(usage, "input_tokens_details", None)
    return {
        "config": config,
        "model": model,
        "effort": effort,
        "temperature": temperature,
        "status": getattr(response, "status", "unknown"),
        "response_id": getattr(response, "id", None),
        "seconds_total": round(end - start, 3),
        "seconds_first_event": round(first_event - start, 3) if first_event else None,
        "seconds_first_text": round(first_text - start, 3) if first_text else None,
        "seconds_last_text": round(last_text - start, 3) if last_text else None,
        "seconds_first_to_last_delta": round(active_stream_seconds, 3) if active_stream_seconds is not None else None,
        "seconds_visible_generation": round(visible_seconds, 3) if visible_seconds else None,
        "input_tokens": input_tokens,
        "cached_input_tokens": getattr(input_details, "cached_tokens", None),
        "output_tokens": output_tokens,
        "total_tokens": getattr(usage, "total_tokens", None),
        "reasoning_tokens": reasoning_tokens,
        "visible_tokens_estimate": visible_tokens,
        "stream_delta_count": len(deltas),
        "stream_character_count": sum(len(part) for part in chunks),
        "stream_deltas": deltas,
        "visible_tokens_per_second": round(visible_tokens / visible_seconds, 2)
        if visible_tokens is not None and visible_seconds and visible_seconds > 0 else None,
        "output_tokens_per_second_end_to_end": round(output_tokens / (end - start), 2)
        if output_tokens is not None else None,
        "answer": "".join(chunks),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Generate prompts only; no API calls")
    parser.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=CONFIGS)
    parser.add_argument("--repeats", type=int, default=1, help="Repeat each pair to assess timing variance")
    parser.add_argument("--max-output-tokens", type=int, default=10000)
    parser.add_argument("--show-stream", action="store_true", help="Print response text chunks as they arrive")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "benchmark_results")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")

    cases = {kind: make_basic_case(kind) for kind in BASIC_KINDS}
    cases.update({kind: make_advanced_case(kind) for kind in ADVANCED_KINDS})
    for kind, (prompt, _) in cases.items():
        print(f"{kind}: {len(prompt.split()):,} words, {len(prompt):,} characters")
    print(f"Planned requests: {len(cases) * len(args.configs) * args.repeats}")
    if args.dry_run:
        return
    api_key = os.getenv("OPENAI_API_KEY") or api_key_from_env_file(ROOT / ".env")
    if not api_key:
        parser.error("OPENAI_API_KEY is missing from the process environment and project .env")
    try:
        from openai import OpenAI
    except ImportError as exc:
        parser.error(f"Cannot import openai: {exc}. Install or repair the environment first.")

    client = OpenAI(api_key=api_key, timeout=180.0, max_retries=0)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    results = []
    jobs = [(kind, config, repeat) for kind in cases
            for repeat in range(args.repeats)
            for config in args.configs]
    # Rotate request order across cases to reduce systematic time drift.
    jobs = sorted(jobs, key=lambda job: (list(cases).index(job[0]), job[2],
                                        (args.configs.index(job[1]) + list(cases).index(job[0])) % len(args.configs)))
    for index, (kind, config, repeat) in enumerate(jobs, 1):
        prompt, expected = cases[kind]
        print(f"[{index}/{len(jobs)}] {kind} / {config} / repeat {repeat + 1}", flush=True)
        try:
            result = run_one(client, config, prompt, args.max_output_tokens, args.show_stream)
            result["quality"] = grade(result["answer"], expected)
        except Exception as exc:
            result = {"config": config, "model": CONFIGS[config][0],
                      "effort": CONFIGS[config][1], "temperature": CONFIGS[config][2],
                      "error": f"{type(exc).__name__}: {exc}"}
        result.update({"case": kind, "repeat": repeat + 1, "expected": expected})
        results.append(result)
        print("  " + json.dumps({k: result.get(k) for k in
              ("status", "seconds_first_event", "seconds_first_text",
               "seconds_last_text", "seconds_total",
               "input_tokens", "cached_input_tokens", "output_tokens", "total_tokens",
               "reasoning_tokens", "visible_tokens_estimate",
               "stream_delta_count", "visible_tokens_per_second", "quality", "error")}))
        (args.output_dir / f"four_config_benchmark_{stamp}.json").write_text(
            json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    print("\nSUMMARY (successful responses only)")
    for config in args.configs:
        group = [r for r in results if r["config"] == config and r.get("quality")]
        if not group:
            continue
        def median(key):
            values = [r[key] for r in group if r.get(key) is not None]
            return round(statistics.median(values), 2) if values else None
        score = sum(r["quality"]["correct"] for r in group)
        total = sum(r["quality"]["total"] for r in group)
        sum_tokens = lambda key: sum(r.get(key) or 0 for r in group)
        print(f"{config:17} quality {score}/{total}; "
              f"median TTFT {median('seconds_first_text')}s; "
              f"median total {median('seconds_total')}s; "
              f"median visible {median('visible_tokens_per_second')} tok/s; "
              f"tokens in/cached/out/reasoning/visible/total "
              f"{sum_tokens('input_tokens')}/{sum_tokens('cached_input_tokens')}/"
              f"{sum_tokens('output_tokens')}/{sum_tokens('reasoning_tokens')}/"
              f"{sum_tokens('visible_tokens_estimate')}/{sum_tokens('total_tokens')}")
    print(f"Full results: {args.output_dir / f'four_config_benchmark_{stamp}.json'}")


if __name__ == "__main__":
    main()
