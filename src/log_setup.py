from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone


class JsonFormatter(logging.Formatter):
    """Structured JSON log lines (M9 §4).

    A record carrying a `turn` payload (see src/turn_log.py) is emitted with those
    fields inlined, so an ops query like `.intent == "faq"` works on the main log
    stream, not just the dedicated TURN_LOG_PATH file.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        turn = getattr(record, "turn", None)
        if isinstance(turn, dict):
            payload.update(turn)
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class DailyFileHandler(logging.Handler):
    """Writes to `<directory>/<YYYY-MM-DD>.log`, switching files when the date changes.

    A stdlib `TimedRotatingFileHandler` rotates by renaming the *previous* day's
    file with a date suffix and keeps appending to one fixed current-file name — it
    never gives the live file itself a date-stamped name. This does, directly: the
    file an operator is tailing right now is always `<today>.log`, no suffix
    lookup required. The date check runs per-record rather than on a timer, so a
    process idle across midnight still rolls to the new file on its next log line.
    """

    def __init__(self, directory: str, level: int = logging.NOTSET) -> None:
        super().__init__(level)
        self._directory = directory
        self._current_date: str | None = None
        self._inner: logging.FileHandler | None = None

    def _today(self) -> str:
        return datetime.now().strftime("%Y-%m-%d")

    def _ensure_inner(self) -> logging.FileHandler:
        today = self._today()
        if self._inner is None or today != self._current_date:
            if self._inner is not None:
                self._inner.close()
            os.makedirs(self._directory, exist_ok=True)
            self._inner = logging.FileHandler(os.path.join(self._directory, f"{today}.log"), encoding="utf-8")
            if self.formatter is not None:
                self._inner.setFormatter(self.formatter)
            self._current_date = today
        return self._inner

    def setFormatter(self, fmt: logging.Formatter | None) -> None:
        super().setFormatter(fmt)
        if self._inner is not None:
            self._inner.setFormatter(fmt)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._ensure_inner().emit(record)
        except Exception:  # noqa: BLE001 - logging must never crash the caller
            self.handleError(record)

    def close(self) -> None:
        if self._inner is not None:
            self._inner.close()
        super().close()


def configure_trailerplace_logging() -> None:
    """Console + a same-day log file, both structured, both always on.

    Every backend run gets `<LOG_DIR>/<YYYY-MM-DD>.log` (default `logs/`) alongside
    the terminal, so the conversation-reasoning blocks (`src/conversation_log.py`)
    and the per-turn JSON records are never terminal-only — a session from
    yesterday is still on disk today.
    """
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    root = logging.getLogger()
    if not root.handlers:
        # LOG_FORMAT=json for machine-readable ops logs; plain text stays the default
        # so a developer running the backend locally still gets readable output.
        if os.getenv("LOG_FORMAT", "text").lower() == "json":
            formatter: logging.Formatter = JsonFormatter()
        else:
            formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")

        console = logging.StreamHandler()
        console.setFormatter(formatter)
        root.addHandler(console)

        file_handler = DailyFileHandler(os.getenv("LOG_DIR", "logs"))
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
    root.setLevel(level)
