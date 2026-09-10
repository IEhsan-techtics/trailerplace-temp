from __future__ import annotations

import logging
import smtplib
from email.mime.text import MIMEText

import requests

from src.config import settings

logger = logging.getLogger(__name__)

# Spec §Tools Email body format (prompt_structured.md lines 56-63):
#   Full Name: ...
#   Email: ...
#   Phone Number: ...
#
#   [Reason] one-line description
# Reason vocabulary is fixed by milestone.md M7 step 2.
_NOT_PROVIDED = "Not provided"


def render_email_body(
    *,
    name: str | None,
    email: str | None,
    phone: str | None,
    reason: str,
    description: str,
) -> str:
    """Exact email body per spec §Tools / milestone.md M7 step 2 — do not paraphrase."""
    return (
        f"Full Name: {name or _NOT_PROVIDED}\n"
        f"Email: {email or _NOT_PROVIDED}\n"
        f"Phone Number: {phone or _NOT_PROVIDED}\n"
        f"\n"
        f"[{reason}] {description}"
    )


def render_subject(*, reason: str, name: str | None, session_id: str) -> str:
    """Subject: `TrailerPlace Lead — {Reason} — {name or session id}` (M7 step 2)."""
    return f"TrailerPlace Lead — {reason} — {name or session_id}"


def send_email(subject: str, body: str) -> bool:
    """Send one internal notification email. Never raises — returns False on failure.

    Backend selected by EMAIL_BACKEND (default `smtp`; `.env` currently sets `graph`).
    No msal dependency: the Graph path talks to the token + sendMail endpoints with
    plain `requests` (milestone.md M7 step 1 / Locked Decisions).
    """
    backend = settings.email_backend.strip().lower()
    try:
        ok = _send_via_graph(subject, body) if backend == "graph" else _send_via_smtp(subject, body)
        if ok:
            recipient = settings.recipient_email if backend == "graph" else settings.email_to
            logger.info("email_sent | backend=%s subject=%r to=%s", backend, subject, recipient)
        return ok
    except Exception:  # pragma: no cover - defensive; the bot must never crash on email
        logger.exception("email_send_failed | backend=%s subject=%r", backend, subject)
        return False


def _send_via_graph(subject: str, body: str) -> bool:
    token_resp = requests.post(
        f"https://login.microsoftonline.com/{settings.tenant_id}/oauth2/v2.0/token",
        data={
            "grant_type": "client_credentials",
            "client_id": settings.client_id,
            "client_secret": settings.client_secret,
            "scope": "https://graph.microsoft.com/.default",
        },
        timeout=30,
    )
    token_resp.raise_for_status()
    access_token = token_resp.json().get("access_token")
    if not access_token:
        logger.error("graph_token_missing | subject=%s", subject)
        return False

    send_resp = requests.post(
        f"https://graph.microsoft.com/v1.0/users/{settings.sender_email}/sendMail",
        headers={
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        },
        json={
            "message": {
                "subject": subject,
                "body": {"contentType": "Text", "content": body},
                "toRecipients": [{"emailAddress": {"address": settings.recipient_email}}],
            }
        },
        timeout=30,
    )
    if send_resp.status_code == 202:
        return True
    logger.error("graph_sendmail_failed | status=%s | body=%s", send_resp.status_code, send_resp.text[:500])
    return False


def _send_via_smtp(subject: str, body: str) -> bool:
    message = MIMEText(body, "plain", "utf-8")
    message["Subject"] = subject
    message["From"] = settings.smtp_from
    # SMTP recipient var is EMAIL_TO; RECIPIENT_EMAIL is Graph's (milestone.md M7 step 1).
    message["To"] = settings.email_to

    with smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=30) as server:
        server.starttls()
        if settings.smtp_user:
            server.login(settings.smtp_user, settings.smtp_password)
        server.sendmail(settings.smtp_from, [settings.email_to], message.as_string())
    return True
