from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


def _env(name: str, default: str | None = None) -> str | None:
    """One setting, with the whitespace taken off. Same contract as ``os.getenv``.

    A value pasted into the Azure portal out of a table or a spreadsheet brings the cell
    separator with it. Live, CHAT_REASONING_EFFORT was set to "	none", every model call
    came back 400 ("Invalid value: '	none'"), and the chatbot spent a day answering with
    the degraded fallback - reading nothing the customer said and asking the same opening
    question over and over. Nothing in the logs pointed at a config typo until the API
    quoted the value back with the tab in it.

    No setting here has ever wanted leading or trailing whitespace, so this is only ever
    the difference between a deployment working and a deployment failing in a way that
    takes a day to see.
    """
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip()


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
    chat_model: str = "gpt-6-luna"
    # "low": a little reasoning buys a reply that follows the state block. gpt-6-luna takes
    # low (checked against the live API); gpt-5.6-luna's lowest was "none", not "minimal".
    chat_reasoning_effort: str = "low"
    # The model writes the whole reply and names the question it asked; Python checks it and
    # falls back to assembling the reply itself (src/graph/nodes/reply_check.py). Off, the
    # reply is assembled from the model's pieces exactly as before.
    llm_writes_reply: bool = False
    # The 5-minute rule (src/idle_timer.py): minutes of silence, with a question of ours
    # waiting, before that category's trailers are shown anyway. 0 switches it off.
    idle_results_minutes: float = 5.0
    search_max_recommendations: int = 5
    feature_llm_rerank_enabled: bool = True
    feature_rerank_model: str = "gpt-5-nano-2025-08-07"
    # gpt-5-nano is the reverse: it rejects "none", and "minimal" is its lowest.
    feature_rerank_reasoning_effort: str = "medium"
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
    # Where the Streamlit frontend is served, so a team email can link straight to the
    # conversation it is about (app.py restores a session from ?chat_session=<id>). Unset
    # means no link: a wrong one is worse than none.
    chat_ui_url: str = ""
    # Commit a finished turn off the reply path (src/turn_saver.py). About 2.4 s of every
    # turn against Azure Postgres from outside the region. Set to 0 to go back to committing
    # before the customer is answered.
    background_turn_save: bool = True
    # The connection pool. The server allows 50, of which 15 are reserved (10 superuser, 5
    # ordinary), so 35 are actually available - and they are shared with the other agents on
    # this database, the control panel and anyone with a SQL client open. SQLAlchemy's
    # defaults are pool_size=5 + max_overflow=10, so three replicas would ask for 45 of the
    # 35 and get connection refusals. Five per replica is ample: a turn holds one connection
    # briefly, twice, and the two background workers hold one each.
    db_pool_size: int = 3
    db_max_overflow: int = 2
    # Waiting beats failing - the turn budget is 150 s, so a burst queues rather than errors.
    db_pool_timeout: float = 30.0
    # Azure closes idle connections; recycling means we retire them first rather than
    # discovering it on checkout.
    db_pool_recycle_seconds: int = 1800
    # Without this a TCP connect to an unreachable server hangs for the OS default - which
    # is how a local server here sat wedged for minutes instead of failing fast.
    db_connect_timeout: int = 10
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
    # ---- Facebook Messenger (src/api/messenger.py) ----
    # Both webhook routes 404 unless this is on AND the app secret and page token are set.
    # A page token with no app secret would serve a bot anyone could speak through, so the
    # secret is part of "configured", not an optional extra.
    messenger_enabled: bool = False
    # Echoed back during Meta's one-time subscription handshake.
    messenger_verify_token: str = ""
    # Signs every webhook body; we verify X-Hub-Signature-256 against it.
    messenger_app_secret: str = ""
    messenger_page_access_token: str = ""
    messenger_graph_api_version: str = "v21.0"
    messenger_send_timeout_seconds: float = 15.0
    # How many message ids to remember in-process, purely to reject a retry that reaches
    # this same instance. The durable check is the unique key on chatbot_inbound_messages.
    messenger_seen_mid_cache_size: int = 2048
    # Whether a trailer goes to Messenger as a generic-template card (photo, title, price,
    # View Trailer button) or as plain bubbles. Off, the trailer still reaches them with a
    # tappable link - see src/domain/cards.py.
    messenger_listing_cards: bool = True
    # How long to hold between one bubble and the next. Longer than the web UI's pause: a
    # phone notification per bubble arriving all at once reads as a machine, not a person.
    messenger_chunk_pause_seconds: float = 0.8
    # Messenger drops a typing indicator after about 20 s, and a turn takes longer than
    # that, so it is re-sent on this interval. 0 turns the refresh off.
    messenger_typing_refresh_seconds: float = 10.0
    # Send the "let me check what we have" line the moment the search starts, rather than
    # leaving the customer with nothing until the trailers land.
    messenger_send_search_status: bool = True

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        return cls(
            openai_api_key=_env("OPENAI_API_KEY", ""),
            openai_model=_env("OPENAI_MODEL", "gpt-4o-mini"),
            analyze_model=_env("ANALYZE_MODEL", "gpt-5-mini"),
            analyze_reasoning_effort=_env("ANALYZE_REASONING_EFFORT", "minimal"),
            chat_model=_env("CHAT_MODEL", "gpt-6-luna"),
            chat_reasoning_effort=_env("CHAT_REASONING_EFFORT", "low"),
            llm_writes_reply=_bool(_env("LLM_WRITES_REPLY")),
            idle_results_minutes=_float(_env("IDLE_RESULTS_MINUTES"), 5.0),
            search_max_recommendations=_int(_env("SEARCH_MAX_RECOMMENDATIONS"), 5),
            feature_llm_rerank_enabled=_bool(_env("FEATURE_LLM_RERANK_ENABLED"), True),
            feature_rerank_model=_env("FEATURE_RERANK_MODEL", "gpt-5-nano-2025-08-07"),
            feature_rerank_reasoning_effort=_env("FEATURE_RERANK_REASONING_EFFORT", "medium"),
            feature_rerank_timeout_seconds=_float(_env("FEATURE_RERANK_TIMEOUT_SECONDS"), 60.0),
            feature_rerank_batch_size=_int(_env("FEATURE_RERANK_BATCH_SIZE"), 20),
            feature_rerank_max_parallel_batches=_int(_env("FEATURE_RERANK_MAX_PARALLEL_BATCHES"), 4),
            feature_rerank_weight=_float(_env("FEATURE_RERANK_WEIGHT"), 0.85),
            feature_fit_weight=_float(_env("FEATURE_FIT_WEIGHT"), 0.15),
            rerank_enabled=_bool(_env("RERANK_ENABLED"), True),
            rerank_warn_ratio=_float(_env("RERANK_WARN_RATIO"), 1.35),
            rerank_extreme_ratio=_float(_env("RERANK_EXTREME_RATIO"), 1.9),
            rerank_length_weight=_float(_env("RERANK_LENGTH_WEIGHT"), 8.0),
            rerank_missing_dim_penalty=_float(_env("RERANK_MISSING_DIM_PENALTY"), 0.35),
            rerank_verbose_logs=_bool(_env("RERANK_VERBOSE_LOGS")),
            make_rerank_verbose_logs=_bool(_env("MAKE_RERANK_VERBOSE_LOGS")),
            show_only_llm_mentioned_cards=_bool(_env("SHOW_ONLY_LLM_MENTIONED_CARDS"), True),
            trailerplace_website=_env("TRAILERPLACE_WEBSITE", ""),
            host=_env("HOST", ""),
            pguser=_env("PGUSER", ""),
            password=_env("PASSWORD", ""),
            database=_env("DATABASE", ""),
            port=_int(_env("PORT"), 0),
            trailerplace_persist_chats=_bool(_env("TRAILERPLACE_PERSIST_CHATS"), True),
            email_backend=_env("EMAIL_BACKEND", "smtp"),
            tenant_id=_env("TENANT_ID", ""),
            client_id=_env("CLIENT_ID", ""),
            client_secret=_env("CLIENT_SECRET", ""),
            sender_email=_env("SENDER_EMAIL", ""),
            recipient_email=_env("RECIPIENT_EMAIL", ""),
            smtp_host=_env("SMTP_HOST", ""),
            smtp_port=_int(_env("SMTP_PORT"), 587),
            smtp_user=_env("SMTP_USER", ""),
            smtp_password=_env("SMTP_PASSWORD", ""),
            smtp_from=_env("SMTP_FROM", ""),
            email_to=_env("EMAIL_TO", ""),
            chat_ui_url=_env("CHAT_UI_URL", ""),
            background_turn_save=_bool(_env("BACKGROUND_TURN_SAVE"), True),
            db_pool_size=_int(_env("DB_POOL_SIZE"), 3),
            db_max_overflow=_int(_env("DB_MAX_OVERFLOW"), 2),
            db_pool_timeout=_float(_env("DB_POOL_TIMEOUT"), 30.0),
            db_pool_recycle_seconds=_int(_env("DB_POOL_RECYCLE_SECONDS"), 1800),
            db_connect_timeout=_int(_env("DB_CONNECT_TIMEOUT"), 10),
            chatbot_api_port=_int(_env("CHATBOT_API_PORT"), 8000),
            langsmith_tracing=_bool(_env("LANGSMITH_TRACING")),
            langsmith_endpoint=_env("LANGSMITH_ENDPOINT", ""),
            langsmith_api_key=_env("LANGSMITH_API_KEY", ""),
            langsmith_project=_env("LANGSMITH_PROJECT", ""),
            debug_state_endpoint=_bool(_env("DEBUG_STATE_ENDPOINT")),
            db_auto_create=_bool(_env("DB_AUTO_CREATE")),
            inventory_lookup_limit=_int(_env("INVENTORY_LOOKUP_LIMIT"), 5),
            chat_timeout_seconds=_float(_env("CHAT_TIMEOUT_SECONDS"), 150.0),
            chat_max_message_chars=_int(_env("CHAT_MAX_MESSAGE_CHARS"), 4000),
            chat_stream_enabled=_bool(_env("CHAT_STREAM_ENABLED"), True),
            chat_stream_words_per_delta=_int(_env("CHAT_STREAM_WORDS_PER_DELTA"), 3),
            chat_stream_delta_seconds=_float(_env("CHAT_STREAM_DELTA_SECONDS"), 0.035),
            chat_stream_chunk_pause_seconds=_float(_env("CHAT_STREAM_CHUNK_PAUSE_SECONDS"), 0.45),
            turn_log_path=_env("TURN_LOG_PATH", ""),
            rules_refresh_seconds=_float(_env("RULES_REFRESH_SECONDS"), 30.0),
            admin_api_token=_env("ADMIN_API_TOKEN", ""),
            messenger_enabled=_bool(_env("MESSENGER_ENABLED")),
            messenger_verify_token=_env("MESSENGER_VERIFY_TOKEN", ""),
            messenger_app_secret=_env("MESSENGER_APP_SECRET", ""),
            messenger_page_access_token=_env("MESSENGER_PAGE_ACCESS_TOKEN", ""),
            messenger_graph_api_version=_env("MESSENGER_GRAPH_API_VERSION", "v21.0"),
            messenger_send_timeout_seconds=_float(_env("MESSENGER_SEND_TIMEOUT_SECONDS"), 15.0),
            messenger_seen_mid_cache_size=_int(_env("MESSENGER_SEEN_MID_CACHE_SIZE"), 2048),
            messenger_listing_cards=_bool(_env("MESSENGER_LISTING_CARDS"), True),
            messenger_chunk_pause_seconds=_float(_env("MESSENGER_CHUNK_PAUSE_SECONDS"), 0.8),
            messenger_typing_refresh_seconds=_float(_env("MESSENGER_TYPING_REFRESH_SECONDS"), 10.0),
            messenger_send_search_status=_bool(_env("MESSENGER_SEND_SEARCH_STATUS"), True),
        )


settings = Settings.from_env()
