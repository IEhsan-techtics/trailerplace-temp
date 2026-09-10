"""LangSmith tracing wiring (M9 §2).

Traces are tagged with the session id and carry the turn's intent + category as
metadata, so a run in the `TrailerPlace` project can be found from a support
ticket ("what did session X do?") without reading logs.

We call the OpenAI SDK directly rather than through LangChain, so nothing is
traced implicitly: `@traceable_or_passthrough` on the client's `structured()` is
what produces the Analyze/Respond spans, and `trace_turn()` groups them into one
run.

Every entry point is gated on `tracing_enabled()` **at call time**, not on the
LANGSMITH_TRACING env var the SDK reads. That distinction matters: a developer's
`.env` routinely has tracing on, and without the runtime gate the pytest suite
would ship real traces to LangSmith just by exercising a decorated function.

Settings are read through `config.settings` (not a `from ... import settings`
binding) so a test can swap the frozen Settings object wholesale.
"""
from __future__ import annotations

import functools
import logging
import os
from contextlib import contextmanager
from typing import Any, Callable, Iterator

from src import config

logger = logging.getLogger(__name__)

_configured = False


def tracing_enabled() -> bool:
    settings = config.settings
    return bool(settings.langsmith_tracing and settings.langsmith_api_key)


def configure_langsmith() -> bool:
    """Map our LANGSMITH_* settings onto the env vars the SDK reads. Idempotent.

    Returns whether tracing ended up enabled, so startup can log it.
    """
    global _configured
    if _configured:
        return tracing_enabled()
    _configured = True

    settings = config.settings
    if not tracing_enabled():
        # Explicitly off: a stale LANGSMITH_TRACING=true in the ambient environment
        # must not make the SDK try to ship traces without a key.
        os.environ["LANGSMITH_TRACING"] = "false"
        logger.info("LangSmith tracing disabled")
        return False

    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
    if settings.langsmith_endpoint:
        os.environ["LANGSMITH_ENDPOINT"] = settings.langsmith_endpoint
    if settings.langsmith_project:
        os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
    logger.info("LangSmith tracing enabled (project=%s)", settings.langsmith_project or "default")
    return True


@contextmanager
def trace_turn(session_id: str, **metadata: Any) -> Iterator[None]:
    """Tag every span produced inside this block with the session id."""
    if not tracing_enabled():
        yield
        return
    try:
        from langsmith.run_helpers import tracing_context
    except ImportError:  # pragma: no cover - langsmith is pinned in requirements
        yield
        return

    meta = {"session_id": session_id, **{k: v for k, v in metadata.items() if v is not None}}
    with tracing_context(tags=[f"session:{session_id}"], metadata=meta):
        yield


def add_turn_metadata(**metadata: Any) -> None:
    """Attach intent/category to the in-flight run once Analyze has produced them.

    Called mid-turn, after `trace_turn` opened the run — the values do not exist
    yet at open time. Silently no-ops outside a run.
    """
    if not tracing_enabled():
        return
    try:
        from langsmith.run_helpers import get_current_run_tree
    except ImportError:  # pragma: no cover
        return
    run = get_current_run_tree()
    if run is None:
        return
    clean = {key: value for key, value in metadata.items() if value is not None}
    if clean:
        run.extra.setdefault("metadata", {}).update(clean)


def traceable_or_passthrough(name: str, run_type: str = "chain") -> Callable:
    """Trace the wrapped callable when tracing is on; call it untouched when off.

    Applied at import time, long before settings are known, so the decision has to
    happen per call. The traced variant is built once and cached.

    `run_type="llm"` makes LangSmith treat the span as a model call, which is what
    lets it show a Tokens/Cost row once `report_llm_usage()` attaches the counts.
    """

    def decorator(func: Callable) -> Callable:
        traced: list[Callable] = []

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not tracing_enabled():
                return func(*args, **kwargs)
            if not traced:
                try:
                    from langsmith import traceable

                    traced.append(traceable(name=name, run_type=run_type)(func))
                except ImportError:  # pragma: no cover
                    traced.append(func)
            return traced[0](*args, **kwargs)

        return wrapper

    return decorator


def report_llm_usage(model: str, prompt_tokens: int, completion_tokens: int) -> None:
    """Attach token usage to the in-flight LLM run so LangSmith shows Tokens/Cost.

    We call `beta.chat.completions.parse` directly, and `wrap_openai` does not
    instrument that method — so without this the traced span is a bare run with no
    usage, and LangSmith leaves both columns blank. `usage_metadata` supplies the
    counts; `ls_model_name` is the key LangSmith uses to price them from its model
    table (unknown models still show tokens, just no cost). No-ops off, or outside
    a run, or on an older langsmith without `RunTree.set`.
    """
    if not tracing_enabled():
        return
    try:
        from langsmith.run_helpers import get_current_run_tree
    except ImportError:  # pragma: no cover
        return
    run = get_current_run_tree()
    if run is None or not hasattr(run, "set"):
        return
    prompt = int(prompt_tokens or 0)
    completion = int(completion_tokens or 0)
    run.set(
        usage_metadata={
            "input_tokens": prompt,
            "output_tokens": completion,
            "total_tokens": prompt + completion,
        },
        metadata={"ls_model_name": model, "ls_provider": "openai"},
    )


__all__ = [
    "add_turn_metadata",
    "configure_langsmith",
    "report_llm_usage",
    "trace_turn",
    "traceable_or_passthrough",
    "tracing_enabled",
]
