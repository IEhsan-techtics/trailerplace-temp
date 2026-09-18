from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


def _bool(value: str | None, default: bool = False) -> bool:
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _int(value: str | None, default: int) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


def _float(value: str | None, default: float = 0.0) -> float:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class Settings:
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    analyze_model: str = "gpt-5-mini"
    analyze_reasoning_effort: str = "minimal"
    # The ONE conversational call per turn (see src/graph/nodes/analyze.py). Separate from
    # analyze_model, which belongs to the older two-call stack this repo was extracted from.
    chat_model: str = "gpt-5.6-luna"
    # "none", not "minimal": gpt-5.6-luna rejects minimal with a 400, and its lowest
    # setting is none. Checked against the live API.
    chat_reasoning_effort: str = "none"
    search_max_recommendations: int = 5
    feature_llm_rerank_enabled: bool = True
    feature_rerank_model: str = "gpt-5-nano-2025-08-07"
    # gpt-5-nano is the reverse: it rejects "none", and "minimal" is its lowest.
    feature_rerank_reasoning_effort: str = "minimal"
    feature_rerank_timeout_seconds: float = 60.0
    feature_rerank_batch_size: int = 20
    feature_rerank_max_parallel_batches: int = 4
    feature_rerank_weight: float = 0.85
    feature_fit_weight: float = 0.15
    rerank_enabled: bool = True
    rerank_warn_ratio: float = 1.35
    rerank_extreme_ratio: float = 1.9
    rerank_length_weight: float = 8.0
    rerank_missing_dim_penalty: float = 0.35
    rerank_verbose_logs: bool = False
    make_rerank_verbose_logs: bool = False
    show_only_llm_mentioned_cards: bool = True
    trailerplace_website: str = ""
    host: str = ""
    pguser: str = ""
    password: str = ""
    database: str = ""
    port: int = 0
    trailerplace_persist_chats: bool = True
    email_backend: str = "smtp"
    tenant_id: str = ""
    client_id: str = ""
    client_secret: str = ""
    sender_email: str = ""
    recipient_email: str = ""
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    smtp_from: str = ""
    email_to: str = ""
    chatbot_api_port: int = 8000
    langsmith_tracing: bool = False
    langsmith_endpoint: str = ""
    langsmith_api_key: str = ""
    langsmith_project: str = ""
    debug_state_endpoint: bool = False
    db_auto_create: bool = False
    inventory_lookup_limit: int = 5
    # Must stay below app.py's 180 s client timeout on POST /chat.
    chat_timeout_seconds: float = 150.0
    chat_max_message_chars: int = 4000
    # Streaming (POST /chat/stream). The turn itself is unchanged - the reply is validated
    # and repaired in full before a single word leaves - so these only pace the delivery of
    # the finished text: how big a step the typing takes, how long between steps, and how
    # long the UI holds between one message bubble and the next.
    chat_stream_enabled: bool = True
    chat_stream_words_per_delta: int = 3
    chat_stream_delta_seconds: float = 0.035
    chat_stream_chunk_pause_seconds: float = 0.45
    # Append-only JSONL of per-turn records; scripts/cost_report.py reads it (M9).
    turn_log_path: str = ""
    # Question rules (src/rules). How often a process checks the DB for a newly activated
    # version - an admin save takes effect in the process that made it at once, and in
    # every other process within this many seconds.
    rules_refresh_seconds: float = 30.0
    # Shared secret for the /admin/rules API. Empty means the API is not mounted at all.
    admin_api_token: str = ""

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        return cls(
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            analyze_model=os.getenv("ANALYZE_MODEL", "gpt-5-mini"),
            analyze_reasoning_effort=os.getenv("ANALYZE_REASONING_EFFORT", "minimal"),
            chat_model=os.getenv("CHAT_MODEL", "gpt-5.6-luna"),
            chat_reasoning_effort=os.getenv("CHAT_REASONING_EFFORT", "none"),
            search_max_recommendations=_int(os.getenv("SEARCH_MAX_RECOMMENDATIONS"), 5),
            feature_llm_rerank_enabled=_bool(os.getenv("FEATURE_LLM_RERANK_ENABLED"), True),
            feature_rerank_model=os.getenv("FEATURE_RERANK_MODEL", "gpt-5-nano-2025-08-07"),
            feature_rerank_reasoning_effort=os.getenv("FEATURE_RERANK_REASONING_EFFORT", "minimal"),
            feature_rerank_timeout_seconds=_float(os.getenv("FEATURE_RERANK_TIMEOUT_SECONDS"), 60.0),
            feature_rerank_batch_size=_int(os.getenv("FEATURE_RERANK_BATCH_SIZE"), 20),
            feature_rerank_max_parallel_batches=_int(os.getenv("FEATURE_RERANK_MAX_PARALLEL_BATCHES"), 4),
            feature_rerank_weight=_float(os.getenv("FEATURE_RERANK_WEIGHT"), 0.85),
            feature_fit_weight=_float(os.getenv("FEATURE_FIT_WEIGHT"), 0.15),
            rerank_enabled=_bool(os.getenv("RERANK_ENABLED"), True),
            rerank_warn_ratio=_float(os.getenv("RERANK_WARN_RATIO"), 1.35),
            rerank_extreme_ratio=_float(os.getenv("RERANK_EXTREME_RATIO"), 1.9),
            rerank_length_weight=_float(os.getenv("RERANK_LENGTH_WEIGHT"), 8.0),
            rerank_missing_dim_penalty=_float(os.getenv("RERANK_MISSING_DIM_PENALTY"), 0.35),
            rerank_verbose_logs=_bool(os.getenv("RERANK_VERBOSE_LOGS")),
            make_rerank_verbose_logs=_bool(os.getenv("MAKE_RERANK_VERBOSE_LOGS")),
            show_only_llm_mentioned_cards=_bool(os.getenv("SHOW_ONLY_LLM_MENTIONED_CARDS"), True),
            trailerplace_website=os.getenv("TRAILERPLACE_WEBSITE", ""),
            host=os.getenv("HOST", ""),
            pguser=os.getenv("PGUSER", ""),
            password=os.getenv("PASSWORD", ""),
            database=os.getenv("DATABASE", ""),
            port=_int(os.getenv("PORT"), 0),
            trailerplace_persist_chats=_bool(os.getenv("TRAILERPLACE_PERSIST_CHATS"), True),
            email_backend=os.getenv("EMAIL_BACKEND", "smtp"),
            tenant_id=os.getenv("TENANT_ID", ""),
            client_id=os.getenv("CLIENT_ID", ""),
            client_secret=os.getenv("CLIENT_SECRET", ""),
            sender_email=os.getenv("SENDER_EMAIL", ""),
            recipient_email=os.getenv("RECIPIENT_EMAIL", ""),
            smtp_host=os.getenv("SMTP_HOST", ""),
            smtp_port=_int(os.getenv("SMTP_PORT"), 587),
            smtp_user=os.getenv("SMTP_USER", ""),
            smtp_password=os.getenv("SMTP_PASSWORD", ""),
            smtp_from=os.getenv("SMTP_FROM", ""),
            email_to=os.getenv("EMAIL_TO", ""),
            chatbot_api_port=_int(os.getenv("CHATBOT_API_PORT"), 8000),
            langsmith_tracing=_bool(os.getenv("LANGSMITH_TRACING")),
            langsmith_endpoint=os.getenv("LANGSMITH_ENDPOINT", ""),
            langsmith_api_key=os.getenv("LANGSMITH_API_KEY", ""),
            langsmith_project=os.getenv("LANGSMITH_PROJECT", ""),
            debug_state_endpoint=_bool(os.getenv("DEBUG_STATE_ENDPOINT")),
            db_auto_create=_bool(os.getenv("DB_AUTO_CREATE")),
            inventory_lookup_limit=_int(os.getenv("INVENTORY_LOOKUP_LIMIT"), 5),
            chat_timeout_seconds=_float(os.getenv("CHAT_TIMEOUT_SECONDS"), 150.0),
            chat_max_message_chars=_int(os.getenv("CHAT_MAX_MESSAGE_CHARS"), 4000),
            chat_stream_enabled=_bool(os.getenv("CHAT_STREAM_ENABLED"), True),
            chat_stream_words_per_delta=_int(os.getenv("CHAT_STREAM_WORDS_PER_DELTA"), 3),
            chat_stream_delta_seconds=_float(os.getenv("CHAT_STREAM_DELTA_SECONDS"), 0.035),
            chat_stream_chunk_pause_seconds=_float(os.getenv("CHAT_STREAM_CHUNK_PAUSE_SECONDS"), 0.45),
            turn_log_path=os.getenv("TURN_LOG_PATH", ""),
            rules_refresh_seconds=_float(os.getenv("RULES_REFRESH_SECONDS"), 30.0),
            admin_api_token=os.getenv("ADMIN_API_TOKEN", ""),
        )


settings = Settings.from_env()
