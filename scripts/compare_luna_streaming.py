"""Streaming GPT-4o mini (temperature 0) vs GPT-5.6 Luna (reasoning none).

Run: python scripts/compare_luna_streaming.py
Preview without API calls: python scripts/compare_luna_streaming.py --dry-run

The script makes 20 requests by default (10 cases x 2 models).
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

from luna_cases import KINDS, make_case


MODELS = ("gpt-5.6-luna", "gpt-4o-mini")
EFFORT = "none"
ROOT = Path(__file__).resolve().parent.parent


def grade(answer: str, expected: dict[str, str]) -> dict:
    try:
        parsed = json.loads(answer.strip())
    except json.JSONDecodeError:
        return {"correct": 0, "total": len(expected), "details": {k: False for k in expected}, "json_valid": False}
    details = {k: isinstance(parsed, dict) and parsed.get(k) == v for k, v in expected.items()}
    return {"correct": sum(details.values()), "total": len(expected), "details": details, "json_valid": True}


def run_one(client, model: str, effort: str, prompt: str, max_output_tokens: int) -> dict:
    start = time.perf_counter()
    first_event = first_text = None
    chunks: list[str] = []
    response = None
    request = {
        "model": model,
        "input": prompt,
        "max_output_tokens": max_output_tokens,
        "stream": True,
        "store": False,
    }
    if model == "gpt-4o-mini":
        request["temperature"] = 0
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
            chunks.append(event.delta)
        elif event.type == "response.completed":
            response = event.response
        elif event.type in ("response.failed", "response.incomplete"):
            response = event.response
    end = time.perf_counter()
    usage = getattr(response, "usage", None)
    input_tokens = getattr(usage, "input_tokens", None)
    output_tokens = getattr(usage, "output_tokens", None)
    output_details = getattr(usage, "output_tokens_details", None)
    reasoning_tokens = getattr(output_details, "reasoning_tokens", None)
    if model == "gpt-4o-mini" and reasoning_tokens is None:
        reasoning_tokens = 0
    visible_tokens = (
        output_tokens - reasoning_tokens
        if output_tokens is not None and reasoning_tokens is not None
        else None
    )
    visible_seconds = end - first_text if first_text is not None else None
    return {
        "model": model,
        "effort": effort if model != "gpt-4o-mini" else None,
        "temperature": 0 if model == "gpt-4o-mini" else None,
        "status": getattr(response, "status", "unknown"),
        "response_id": getattr(response, "id", None),
        "seconds_total": round(end - start, 3),
        "seconds_first_event": round(first_event - start, 3) if first_event else None,
        "seconds_first_text": round(first_text - start, 3) if first_text else None,
        "seconds_visible_generation": round(visible_seconds, 3) if visible_seconds else None,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "reasoning_tokens": reasoning_tokens,
        "visible_tokens_estimate": visible_tokens,
        "visible_tokens_per_second": round(visible_tokens / visible_seconds, 2)
        if visible_tokens is not None and visible_seconds and visible_seconds > 0 else None,
        "output_tokens_per_second_end_to_end": round(output_tokens / (end - start), 2)
        if output_tokens is not None else None,
        "answer": "".join(chunks),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Generate prompts only; no API calls")
    parser.add_argument("--models", nargs="+", default=list(MODELS))
    parser.add_argument("--repeats", type=int, default=1, help="Repeat each pair to assess timing variance")
    parser.add_argument("--max-output-tokens", type=int, default=1500)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "benchmark_results")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")

    cases = {kind: make_case(kind) for kind in KINDS}
    for kind, (prompt, _) in cases.items():
        print(f"{kind}: {len(prompt.split()):,} words, {len(prompt):,} characters")
    if args.dry_run:
        return
    if not os.getenv("OPENAI_API_KEY"):
        try:
            from dotenv import load_dotenv
            load_dotenv(ROOT / ".env")
        except ImportError:
            pass
    if not os.getenv("OPENAI_API_KEY"):
        parser.error("OPENAI_API_KEY is missing from the process environment")
    try:
        from openai import OpenAI
    except ImportError as exc:
        parser.error(f"Cannot import openai: {exc}. Install or repair the environment first.")

    client = OpenAI(timeout=180.0, max_retries=0)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    results = []
    jobs = [(kind, EFFORT, model, repeat) for kind in cases
            for repeat in range(args.repeats)
            for model in args.models]
    # Alternate models within each case/repeat to reduce drift bias.
    for index, (kind, effort, model, repeat) in enumerate(jobs, 1):
        prompt, expected = cases[kind]
        setting = "temperature 0" if model == "gpt-4o-mini" else f"reasoning {effort}"
        print(f"[{index}/{len(jobs)}] {kind} / {model} / {setting} / repeat {repeat + 1}", flush=True)
        try:
            result = run_one(client, model, effort, prompt, args.max_output_tokens)
            result["quality"] = grade(result["answer"], expected)
        except Exception as exc:
            result = {"model": model, "effort": effort if model != "gpt-4o-mini" else None,
                      "temperature": 0 if model == "gpt-4o-mini" else None,
                      "error": f"{type(exc).__name__}: {exc}"}
        result.update({"case": kind, "repeat": repeat + 1, "expected": expected})
        results.append(result)
        print("  " + json.dumps({k: result.get(k) for k in
              ("seconds_first_text", "seconds_total", "visible_tokens_per_second", "quality", "error")}))
        (args.output_dir / f"luna_vs_4o_mini_{stamp}.json").write_text(
            json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    print("\nSUMMARY (successful responses only)")
    for model in args.models:
        group = [r for r in results if r["model"] == model and r.get("quality")]
        if not group:
            continue
        def median(key):
            values = [r[key] for r in group if r.get(key) is not None]
            return round(statistics.median(values), 2) if values else None
        score = sum(r["quality"]["correct"] for r in group)
        total = sum(r["quality"]["total"] for r in group)
        setting = "temp=0" if model == "gpt-4o-mini" else "effort=none"
        print(f"{model:16} {setting:11} quality {score}/{total}; "
              f"median TTFT {median('seconds_first_text')}s; "
              f"median total {median('seconds_total')}s; "
              f"visible {median('visible_tokens_per_second')} tok/s; "
              f"end-to-end {median('output_tokens_per_second_end_to_end')} tok/s")
    print(f"Full results: {args.output_dir / f'luna_vs_4o_mini_{stamp}.json'}")


if __name__ == "__main__":
    main()
