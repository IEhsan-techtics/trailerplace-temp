"""POST /internal/idle-sweep - the 5-minute rule's minute tick (src/idle_sweep.py).

Called by an Azure Container Apps job on a one-minute schedule. Mounted only when
IDLE_SWEEP_TOKEN is set, and the caller must present it: the sweep sends messages to
customers, so nothing outside our own schedule may trigger it.
"""
from __future__ import annotations

import hmac

from fastapi import APIRouter, Header, HTTPException

from src.config import settings

router = APIRouter()


@router.post("/internal/idle-sweep")
def idle_sweep(x_idle_sweep_token: str | None = Header(default=None)) -> dict:
    # Compared in constant time; the token is the only thing between the internet and a
    # route that messages customers.
    if not x_idle_sweep_token or not hmac.compare_digest(x_idle_sweep_token, settings.idle_sweep_token):
        raise HTTPException(status_code=401, detail="unauthorised")
    from src import idle_sweep as sweeper

    results = sweeper.sweep()
    return {"swept": len(results), "results": results}
