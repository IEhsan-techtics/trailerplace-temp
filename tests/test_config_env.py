"""Settings read out of the environment.

Written after a live outage: CHAT_REASONING_EFFORT was set in the Azure portal to "\tnone"
- a tab that came along when the value was pasted out of a table. Every model call returned
400, every turn fell back to the degraded empty output, and the chatbot answered a whole
day's customers without reading a word they said.
"""
from __future__ import annotations

import pytest

from src.config import Settings


@pytest.fixture
def env(monkeypatch):
    """A clean environment, so a developer's own .env cannot change the answer."""
    import os

    for name in list(os.environ):
        if name.isupper():
            monkeypatch.delenv(name, raising=False)
    # from_env calls load_dotenv, which would put the real file back.
    monkeypatch.setattr("src.config.load_dotenv", lambda *a, **k: None)
    return monkeypatch


@pytest.mark.parametrize("raw", ["\tnone", " none", "none ", "\nnone\n", "  none  "])
def test_whitespace_around_a_value_is_ignored(env, raw):
    """The API rejects '\tnone' and accepts 'none'. Nothing else about the deployment
    differs, so this one character is the difference between working and not."""
    env.setenv("CHAT_REASONING_EFFORT", raw)
    assert Settings.from_env().chat_reasoning_effort == "none"


def test_a_padded_number_still_parses(env):
    env.setenv("SEARCH_MAX_RECOMMENDATIONS", " 7 ")
    assert Settings.from_env().search_max_recommendations == 7


def test_a_padded_flag_still_parses(env):
    env.setenv("FEATURE_LLM_RERANK_ENABLED", " 0 ")
    assert Settings.from_env().feature_llm_rerank_enabled is False


def test_an_unset_variable_still_takes_the_default(env):
    """The helper has to keep os.getenv's contract: unset is not the same as empty."""
    env.delenv("CHAT_MODEL", raising=False)
    assert Settings.from_env().chat_model == "gpt-5.6-luna"


def test_a_variable_set_to_only_whitespace_reads_as_unset(env):
    """Better than the alternative: a key of "   " used to be truthy, so the app started
    and failed on the first customer instead of on the health check."""
    env.setenv("OPENAI_API_KEY", "   ")
    assert Settings.from_env().openai_api_key == ""
