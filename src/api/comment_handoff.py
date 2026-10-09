"""POST /internal/comment-handoff - a conversation that starts with our DM, not theirs.

The Omni Channel Agent answers public Facebook comments. When one shows buying interest or
is a complaint it also sends a private reply - one Messenger DM to the commenter - and calls
this with the PSID Meta returned. The customer's answer then arrives on our own Messenger
webhook, and src/conversation_store.seed_from_comment is what makes it a continuation:
their comment and our DM are already in the transcript, they are not greeted as a new
contact, and they are asked for their details only if we do not have them.

Mounted only when COMMENT_HANDOFF_TOKEN is set, and the caller must present it: the route
writes into a customer's conversation.
"""
from __future__ import annotations

import hmac

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field

from src import conversation_store
from src.config import settings

router = APIRouter()


class CommentHandoff(BaseModel):
    psid: str = Field(min_length=1, max_length=255)
    platform: str = Field(default="facebook", max_length=32)
    intent: str = Field(min_length=1, max_length=64)
    comment_id: str = Field(min_length=1, max_length=255)
    comment_text: str = Field(default="", max_length=4000)
    commenter_name: str | None = Field(default=None, max_length=255)
    post_id: str | None = Field(default=None, max_length=255)
    post_summary: str = Field(default="", max_length=1000)
    # The one trailer the post is about ("Aluma 6310H-TG"), when it is about one. Interest in
    # it is logged and the team gets a Listing Interest ticket.
    post_trailer: str | None = Field(default=None, max_length=200)
    dm_text: str = Field(min_length=1, max_length=2000)


@router.post("/internal/comment-handoff")
def comment_handoff(
    body: CommentHandoff,
    x_comment_handoff_token: str | None = Header(default=None),
) -> dict:
    if not x_comment_handoff_token or not hmac.compare_digest(
        x_comment_handoff_token, settings.comment_handoff_token
    ):
        raise HTTPException(status_code=401, detail="unauthorised")
    return conversation_store.seed_from_comment(
        body.psid,
        platform=body.platform,
        intent=body.intent,
        comment_text=body.comment_text,
        dm_text=body.dm_text,
        comment_id=body.comment_id,
        post_summary=body.post_summary,
        commenter_name=body.commenter_name,
        post_trailer=body.post_trailer,
        post_id=body.post_id,
    )
