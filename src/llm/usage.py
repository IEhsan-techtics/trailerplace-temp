"""Per-turn LLM call/token accounting (M9 §3).

The cost audit asserts 2 LLM calls per normal turn (Analyze + Respond), with one
additional tagged completion(s) only when batched semantic feature reranking runs. Search
search turns add one embedding; inventory-lookup turns add no extra calls. To
assert that from local logs we have to count the calls where they happen.

A turn is a contextvar scope: `usage_scope()` installs a fresh `TurnUsage`, the
OpenAI client records each chat completion into it, and the embedding helper
records each embedding. Nothing outside the scope records anything, so library
code stays safe to call from tests and scripts.

contextvars (not a plain global) because /chat turns run on a ThreadPoolExecutor:
each graph run needs its own counters even when several sessions overlap.
"""
from __future__ import annotations

import contextvars
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Iterator


@dataclass
class TurnUsage:
    """Counters for one /chat turn."""

    chat_completions: int = 0
    feature_reranks: int = 0
    embeddings: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    models: list[str] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def as_dict(self) -> dict:
        data = asdict(self)
        data["total_tokens"] = self.total_tokens
        return data


_current: contextvars.ContextVar[TurnUsage | None] = contextvars.ContextVar("trailerplace_turn_usage", default=None)


@contextmanager
def usage_scope() -> Iterator[TurnUsage]:
    """Install a fresh TurnUsage for the duration of one turn."""
    usage = TurnUsage()
    token = _current.set(usage)
    try:
        yield usage
    finally:
        _current.reset(token)


def current_usage() -> TurnUsage | None:
    """The active turn's counters, or None when called outside a scope."""
    return _current.get()


def record_completion(
    model: str,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    *,
    purpose: str | None = None,
) -> None:
    usage = _current.get()
    if usage is None:
        return
    usage.chat_completions += 1
    if purpose == "feature_rerank":
        usage.feature_reranks += 1
    usage.prompt_tokens += int(prompt_tokens or 0)
    usage.completion_tokens += int(completion_tokens or 0)
    if model and model not in usage.models:
        usage.models.append(model)


def record_embedding(model: str, prompt_tokens: int = 0) -> None:
    usage = _current.get()
    if usage is None:
        return
    usage.embeddings += 1
    usage.prompt_tokens += int(prompt_tokens or 0)
    if model and model not in usage.models:
        usage.models.append(model)
