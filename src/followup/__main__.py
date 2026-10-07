"""``python -m src.followup``: one sweep, then exit. The Azure job's command."""
from __future__ import annotations

import sys

from src.log_setup import configure_trailerplace_logging


def main() -> int:
    configure_trailerplace_logging()
    from src.followup import sweep

    counts = sweep.run()
    return 1 if counts.get("error") else 0


if __name__ == "__main__":
    sys.exit(main())
