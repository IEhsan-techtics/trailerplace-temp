"""
TrailerPlace AI Assistant — Streamlit frontend.

Run from this folder (where pyproject.toml lives), using your venv:
    .\\.venv\\Scripts\\streamlit.exe run app.py

Requires the FastAPI backend (`python main.py` with the same venv) unless CHATBOT_API_URL points elsewhere.

If you see imports using another project's .venv, deactivate it first
(PowerShell: Remove-Item Env:\\VIRTUAL_ENV) or use run_streamlit.ps1.
"""
import html
import json
import logging
import os
import secrets
import time
import uuid
from datetime import datetime, timezone
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import requests
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

load_dotenv()
from src.log_setup import configure_trailerplace_logging

configure_trailerplace_logging()

from src.conversation_store import (
    enqueue_save_user_feedback,
    persistence_enabled,
)
from src.shown_listings_store import (
    accumulate_shown_urls_from_chat_messages,
    add_shown_keys_and_urls,
    add_shown_urls,
)
from src.domain.reply_chunks import split_reply_into_chunks, urls_in_chunk
from src.models import TrailerListing
from src.thinking_agent import (
    generate_thinking_flow,
    log_thinking_flow,
    thinking_agent_background,
    thinking_agent_enabled,
)

_AUTH_USER = (os.getenv("TRAILERPLACE_APP_USERNAME") or "").strip()
_AUTH_PASS = (os.getenv("TRAILERPLACE_APP_PASSWORD") or "").strip()
_AUTH_CONFIGURED = bool(_AUTH_USER and _AUTH_PASS)
_THINKING_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="tp_thinking")
_BACKEND_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="tp_backend_ready")
# Its own pool: the /chat call must never queue behind a thinking job, or the status polling
# below would sit idle while the turn waits its turn for a worker.
_CHAT_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="tp_chat")
_THINKING_POLL_MS = int((os.getenv("THINKING_AGENT_POLL_MS") or "700").strip())
_BACKEND_READY_TIMEOUT = float((os.getenv("BACKEND_READY_TIMEOUT_SECONDS") or "180").strip())
# Poll granularity is dead time added on top of the backend's real boot: at 2 s the sign-in
# screen could sit there for a further two seconds after /health had already gone green. The
# request is a single readiness-flag read, so asking more often costs nothing.
_BACKEND_READY_POLL = float((os.getenv("BACKEND_READY_POLL_SECONDS") or "0.4").strip())
CHATBOT_API_URL = (os.getenv("CHATBOT_API_URL") or "http://127.0.0.1:8000").strip().rstrip("/")
_RULES_DOC_PATH = Path(__file__).with_name("langgraph_rules_vs_excel.md")
# Keep listing data in the chat/API flow for the assistant and session tracking,
# but do not render the listing-card UI in Streamlit unless explicitly enabled.
SHOW_LISTING_CARDS = (os.getenv("SHOW_LISTING_CARDS") or "0").strip().lower() in {
    "1", "true", "yes", "on"
}

# Streaming: the reply is typed out over POST /chat/stream and arrives as several messages -
# the intro, one per trailer, then the closing question - instead of one wall of text. Off
# falls back to the blocking POST /chat and a single bubble.
CHAT_STREAMING = (os.getenv("CHAT_STREAM_ENABLED") or "1").strip().lower() in {
    "1", "true", "yes", "on"
}
# The block cursor shown at the end of the text still being typed.
_TYPING_CARET = "▌"

# The one waiting indicator, shown from the moment the turn starts until the reply lands.
TYPING_DOTS_HTML = '<div class="tp-typing"><span></span><span></span><span></span></div>'

# How often the UI asks the backend what it is doing, while /chat is still running.
# No artificial floor on top: the line cannot appear until the graph has run analyze and
# routed into the search node, which is several seconds of real work on its own.
_TURN_STATUS_POLL_SECONDS = 0.35


def _fetch_turn_status(session_id: str) -> str | None:
    """What the backend is doing right now, or None. Never raises: this is decoration."""
    try:
        response = requests.get(
            f"{CHATBOT_API_URL}/session/{session_id}/turn-status", timeout=3
        )
        response.raise_for_status()
        return (response.json().get("search_status_message") or "").strip() or None
    except (requests.RequestException, ValueError):
        return None


def _search_note_html(text: str) -> str:
    return f'<p class="tp-search-note">{html.escape(str(text).strip())}</p>'


def _waiting_bubble_html(note: str | None) -> str:
    """What sits in the assistant bubble while the turn runs.

    Dots on their own until the search fires; once it does, the line it published sits above
    them, because the dots still have work left to describe (the reply is not written yet).
    """
    if not note:
        return TYPING_DOTS_HTML
    return f"{_search_note_html(note)}{TYPING_DOTS_HTML}"


def _render_search_note(text: str | None) -> None:
    """The "let me go and look" line, above the results it belongs to.

    The backend sets it inside search_node, so it appears if and only if the inventory search
    actually ran - it can never tell the customer we went to check stock on a turn that never
    searched.
    """
    if not str(text or "").strip():
        return
    st.markdown(_search_note_html(text), unsafe_allow_html=True)


def _iter_sse_events(response: requests.Response):
    """Yield (event_name, payload_dict) for each frame of a text/event-stream response.

    Small on purpose: the backend only ever sends single-line JSON `data:` fields, so there is
    no multi-line data to reassemble. A frame we cannot parse is skipped rather than raised -
    a decoration event must never take the reply down with it.
    """
    event = "message"
    for raw in response.iter_lines(decode_unicode=True):
        if raw is None:
            continue
        line = raw.strip()
        if not line:
            event = "message"
            continue
        if line.startswith(":"):
            continue
        if line.startswith("event:"):
            event = line[len("event:") :].strip()
            continue
        if line.startswith("data:"):
            try:
                yield event, json.loads(line[len("data:") :].strip())
            except ValueError:
                continue


def _listings_for_chunk(chunk: str, listings: list, claimed: set[int]) -> list[tuple[int, object]]:
    """The listings this chunk links to, so a trailer's card sits under its own message.

    `claimed` is carried across the chunks of one reply: a listing belongs to the first chunk
    that links it, and whatever no chunk links falls through to the caller's leftovers.
    """
    linked = {str(url or "").rstrip("/").lower() for url in urls_in_chunk(chunk)}
    if not linked:
        return []
    matched = []
    for index, listing in enumerate(listings):
        if index in claimed:
            continue
        url = str(getattr(listing, "url", "") or "").rstrip("/").lower()
        if url and url in linked:
            claimed.add(index)
            matched.append((index + 1, listing))
    return matched


def _render_message_bubbles(msg: dict):
    """Render one stored turn as the bubbles it was sent in; return the LAST bubble.

    An assistant reply is re-split here rather than stored pre-split, so a session reloaded
    from the API (which returns only the text) shows the same bubbles as the live turn did.
    The caller keeps writing into the returned container - the feedback control and the
    insights note belong under the last bubble, not repeated under every one.
    """
    role = msg.get("role", "assistant")
    content = str(msg.get("content") or "")
    listings = list(msg.get("listings") or [])
    chunks = split_reply_into_chunks(content) if role == "assistant" else []
    if not chunks:
        chunks = [content]
    claimed: set[int] = set()
    bubble = None
    for index, chunk in enumerate(chunks):
        bubble = st.chat_message(role)
        with bubble:
            if index == 0:
                _render_search_note(msg.get("search_note"))
            st.markdown(chunk)
            if SHOW_LISTING_CARDS:
                for rank, listing in _listings_for_chunk(chunk, listings, claimed):
                    render_card(listing, rank)
    if SHOW_LISTING_CARDS and bubble is not None:
        # Anything the reply did not link (or every card, when the text has no cards at all).
        with bubble:
            for rank, listing in enumerate(listings, 1):
                if rank - 1 not in claimed:
                    render_card(listing, rank)
    return bubble


def _reset_api_session(session_id: str) -> None:
    try:
        requests.post(
            f"{CHATBOT_API_URL}/session/reset",
            json={"session_id": session_id},
            timeout=8,
        )
    except requests.RequestException:
        pass


def _restore_api_session(session_id: str) -> dict | None:
    try:
        response = requests.get(f"{CHATBOT_API_URL}/session/{session_id}", timeout=10)
        response.raise_for_status()
        return response.json()
    except (requests.RequestException, ValueError):
        return None


def _password_matches(got: str, expected: str) -> bool:
    ga, ea = got.encode("utf-8"), expected.encode("utf-8")
    if len(ga) != len(ea):
        return False
    return secrets.compare_digest(ga, ea)


def _run_thinking_job(session_id: str, payload: dict) -> dict:
    result = generate_thinking_flow(payload)
    log_thinking_flow(session_id, payload, result)
    return result


def _wait_for_backend_ready() -> dict[str, str]:
    deadline = time.monotonic() + max(5.0, _BACKEND_READY_TIMEOUT)
    last_error = "Backend did not become ready in time."
    while time.monotonic() < deadline:
        try:
            response = requests.get(f"{CHATBOT_API_URL}/health", timeout=5)
            if response.ok and (response.json().get("status") == "ok"):
                return {"status": "ready", "error": ""}
            last_error = f"Health check returned HTTP {response.status_code}."
        except (requests.RequestException, ValueError) as exc:
            last_error = str(exc)
        time.sleep(max(0.5, _BACKEND_READY_POLL))
    return {"status": "error", "error": last_error}


def _load_rules_markdown() -> str:
    try:
        return _RULES_DOC_PATH.read_text(encoding="utf-8")
    except FileNotFoundError:
        return "The rules document could not be found."
    except OSError as exc:
        return f"Unable to load the rules document: {exc!s}"


def _render_rules_page() -> None:
    _render_header("Rules", "Reference view for the current chatbot rules and behavior.")
    st.markdown(_load_rules_markdown())


st.set_page_config(
    page_title="TrailerPlace · Assistant",
    page_icon="🚛",
    layout="wide",
    initial_sidebar_state="expanded",
)

_requested_theme = st.query_params.get("theme")
if "ui_dark_mode" not in st.session_state:
    st.session_state.ui_dark_mode = _requested_theme == "dark"

# Streamlit session state is reset when the server/app restarts. Restore the
# last browser preference through localStorage, with the URL as the bridge
# that Python can read before rendering the theme.
components.html(
    """
<script>
(function() {
  const parentWindow = window.parent;
  const storageKey = "trailerplace-ui-theme";
  const url = new URL(parentWindow.location.href);
  const urlTheme = url.searchParams.get("theme");
  const savedTheme = parentWindow.localStorage.getItem(storageKey);
  const chatKey = "trailerplace-chat-session";
  const urlSession = url.searchParams.get("chat_session");
  // localStorage, not sessionStorage: the session id must survive the tab closing and the
  // hosted app going to sleep, or every fresh visit silently starts a brand-new conversation.
  // (sessionStorage is still read once so sessions saved by older builds keep working.)
  const savedSession =
    parentWindow.localStorage.getItem(chatKey) ||
    parentWindow.sessionStorage.getItem(chatKey);

  if (!urlSession && savedSession) {
    url.searchParams.set("chat_session", savedSession);
    parentWindow.location.replace(url.toString());
    return;
  } else if (urlSession) {
    parentWindow.localStorage.setItem(chatKey, urlSession);
    parentWindow.sessionStorage.setItem(chatKey, urlSession);
  }

  if (!urlTheme && (savedTheme === "light" || savedTheme === "dark")) {
    url.searchParams.set("theme", savedTheme);
    parentWindow.location.replace(url.toString());
    return;
  }
  if (!urlTheme) {
    parentWindow.localStorage.setItem(storageKey, "light");
    url.searchParams.set("theme", "light");
    parentWindow.history.replaceState({}, "", url.toString());
  } else if (urlTheme === "light" || urlTheme === "dark") {
    parentWindow.localStorage.setItem(storageKey, urlTheme);
  }
})();
</script>
""",
    height=0,
)

# ─────────────────────────────────────────────────────────────
# CSS INJECTION — via iframe JS so Streamlit sanitizer is bypassed
#
# Design direction: a dealer's dispatch desk, not a generic AI chat.
# Dark asphalt-and-steel surface, a "spec sheet" card for listings,
# Clean system typography for a familiar chat interface, with a
# hazard-stripe rule (the
# reflective tape on a trailer's rear doors) as the one repeating
# signature motif instead of a generic divider.
# ─────────────────────────────────────────────────────────────
components.html("""
<script>
(function() {
  var css = `
    :root {
      --tp-font: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      --tp-bg: #000000;
      --tp-secondary-bg: #0D0D0D;
      --tp-card-bg: #171717;
      --tp-text: #ECECEC;
      --tp-text-muted: #AFAFAF;
      --tp-border: rgba(255,255,255,0.12);
      --tp-accent: #F97316;
      --tp-accent-dim: rgba(249,115,22,0.38);
      --tp-amber: #FBBF24;
      --tp-green: #22C55E;
      --tp-shadow: rgba(0,0,0,0.45);
    }

    html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
      font-family: var(--tp-font) !important;
    }
    [data-testid="stAppViewContainer"], [data-testid="stMain"], [data-testid="stApp"] {
      background-color: var(--tp-bg) !important;
      background-image: none !important;
      color: var(--tp-text) !important;
    }
    [data-testid="stHeader"] { background: transparent !important; }

    /* Hide chrome — keep sidebar collapse control, it lives outside the toolbar */
    #MainMenu, footer, [data-testid="stDecoration"] { display: none !important; }

    .block-container {
      padding: 1.25rem 1.5rem 1rem 1.5rem !important;
      max-width: 760px !important;
    }

    /* ── Signature element: hazard-stripe rule, like reflective trailer tape ── */
    .tp-hazard {
      height: 4px;
      border-radius: 3px;
      margin: 10px 0 22px 0;
      background: repeating-linear-gradient(
        135deg,
        var(--tp-accent) 0px, var(--tp-accent) 10px,
        #14161A 10px, #14161A 20px
      );
      opacity: 0.92;
    }

    /* ── Header: dispatch-ticket style ── */
    .tp-header-eyebrow {
      font-family: var(--tp-font);
      font-weight: 600;
      font-size: 11px;
      letter-spacing: 1.1px;
      color: var(--tp-amber);
      text-transform: uppercase;
      display: flex; align-items: center; gap: 7px;
      margin-bottom: 4px;
    }
    .tp-header-eyebrow .dot {
      width: 7px; height: 7px; border-radius: 50%;
      background: var(--tp-green);
      box-shadow: 0 0 6px var(--tp-green);
      flex-shrink: 0;
    }
    .tp-header-title {
      font-family: var(--tp-font);
      font-weight: 600;
      font-size: 27px;
      letter-spacing: -0.55px;
      color: var(--tp-text);
      line-height: 1.2;
    }
    .tp-header-sub {
      font-size: 13px;
      color: var(--tp-text-muted);
      margin-top: 3px;
    }

    /* ── Sidebar: vehicle-plate styling ── */
    [data-testid="stSidebar"] {
      background: var(--tp-secondary-bg) !important;
      border-right: 1px solid var(--tp-border) !important;
      color: var(--tp-text) !important;
      font-family: var(--tp-font) !important;
    }
    [data-testid="stSidebar"] * {
      color: var(--tp-text) !important;
    }
    [data-testid="stSidebar"] hr { border-color: var(--tp-border) !important; }
    [data-testid="stSidebar"] button {
      background: var(--tp-bg) !important;
      border: 1px solid var(--tp-border) !important;
      color: var(--tp-text) !important;
      border-radius: 9px !important;
      font-family: var(--tp-font) !important;
      font-weight: 500 !important;
      transition: border-color .15s, background .15s;
    }
    [data-testid="stSidebar"] button:hover {
      border-color: var(--tp-accent-dim) !important;
      background: #20242C !important;
    }
    .tp-plate {
      font-family: var(--tp-font);
      font-weight: 600;
      font-size: 20px;
      letter-spacing: -.3px;
      color: var(--tp-text);
      line-height: 1.15;
    }
    .tp-plate-sub {
      font-family: var(--tp-font);
      font-weight: 500;
      font-size: 10.5px;
      color: var(--tp-amber);
      letter-spacing: 1.2px;
      text-transform: uppercase;
      margin-top: 2px;
    }
    .tp-info-row {
      font-family: var(--tp-font) !important;
      font-size: 12.5px;
      color: var(--tp-text-muted) !important;
      display: flex; gap: 9px; align-items: center;
      padding: 5px 0;
    }
    .tp-info-row b { color: var(--tp-text) !important; font-weight: 500; }

    /* ── Generic form controls (login + feedback) ── */
    [data-testid="stTextInput"] input, [data-testid="stTextArea"] textarea {
      background: var(--tp-secondary-bg) !important;
      border: 1px solid var(--tp-border) !important;
      color: var(--tp-text) !important;
      border-radius: 8px !important;
    }
    [data-testid="stTextInput"] input:focus, [data-testid="stTextArea"] textarea:focus {
      border-color: var(--tp-accent-dim) !important;
      box-shadow: 0 0 0 1px var(--tp-accent-dim) !important;
    }
    div[data-testid="stButton"] button, div[data-testid="stFormSubmitButton"] button {
      border-radius: 9px !important;
      font-family: var(--tp-font) !important;
      font-weight: 500 !important;
      background: var(--tp-secondary-bg);
      border: 1px solid var(--tp-border);
      color: var(--tp-text);
    }
    div[data-testid="stFormSubmitButton"] button[kind="primary"] {
      background: var(--tp-accent) !important;
      border: none !important;
      color: #fff !important;
    }
    div[data-testid="stFormSubmitButton"] button[kind="primary"]:hover { background: #EA6A0A !important; }
    div[data-testid="stButton"] button[kind="primary"] {
      background: var(--tp-accent) !important;
      border: none !important;
      color: #fff !important;
    }
    div[data-testid="stButton"] button[kind="primary"]:hover { background: #EA6A0A !important; }
    [data-testid="stSidebar"] button[kind="primary"] {
      background: var(--tp-accent) !important;
      border: none !important;
      color: #fff !important;
    }
    [data-testid="stSidebar"] button[kind="primary"]:hover { background: #EA6A0A !important; }
    /* Tab-style nav buttons (Chatbot / Rules) */
    .tp-nav-row div[data-testid="stButton"] button {
      font-weight: 600 !important;
      letter-spacing: .2px;
    }
    /* Keep Streamlit's native animated sidebar, but give both toggle states
       the same compact, ChatGPT-like treatment. */
    [data-testid="stSidebar"] {
      transition: transform 240ms cubic-bezier(.22, 1, .36, 1),
                  margin-left 240ms cubic-bezier(.22, 1, .36, 1) !important;
    }
    [data-testid="stSidebarCollapseButton"] button,
    [data-testid="stSidebarCollapsedControl"] button {
      width: 36px !important;
      min-width: 36px !important;
      height: 36px !important;
      padding: 0 !important;
      font-size: 17px !important;
      line-height: 1 !important;
      border-color: transparent !important;
      background: transparent !important;
    }
    [data-testid="stSidebarCollapseButton"] button:hover,
    [data-testid="stSidebarCollapsedControl"] button:hover {
      background: rgba(255,255,255,0.07) !important;
      border-color: var(--tp-border) !important;
    }
    [data-testid="stSidebarCollapseButton"] {
      top: 0.65rem !important;
      right: 0.75rem !important;
    }
    [data-testid="stSidebarCollapsedControl"] {
      display: block !important;
      position: fixed !important;
      top: 0.8rem !important;
      left: 0.8rem !important;
      z-index: 1000000 !important;
    }
    [data-testid="stVerticalBlockBorderWrapper"] {
      border-color: var(--tp-border) !important;
      background: var(--tp-secondary-bg) !important;
      border-radius: 10px !important;
    }
    [data-testid="stCaptionContainer"] { color: var(--tp-text-muted) !important; }

    /* ── Chat input ── */
    [data-testid="stBottom"],
    [data-testid="stBottom"] > div,
    [data-testid="stBottomBlockContainer"],
    [data-testid="stChatInputContainer"] {
      background: #000000 !important;
      background-color: #000000 !important;
      background-image: none !important;
    }
    [data-testid="stBottom"]::before,
    [data-testid="stBottom"]::after,
    [data-testid="stBottomBlockContainer"]::before,
    [data-testid="stBottomBlockContainer"]::after {
      background: #000000 !important;
      background-image: none !important;
    }
    [data-testid="stChatInput"] > div {
      background: var(--tp-secondary-bg) !important;
      border: 1.5px solid var(--tp-border) !important;
      border-radius: 14px !important;
      box-shadow: 0 4px 18px var(--tp-shadow) !important;
    }
    [data-testid="stChatInput"] textarea {
      font-family: var(--tp-font) !important;
      font-size: 15px !important;
      color: var(--tp-text) !important;
      caret-color: var(--tp-accent) !important;
      background: transparent !important;
    }
    [data-testid="stChatInput"] textarea::placeholder {
      color: rgba(156, 163, 175, 0.85) !important;
    }
    [data-testid="stChatInput"] button {
      background: var(--tp-accent) !important;
      border-radius: 10px !important;
      border: none !important;
    }
    [data-testid="stChatInput"] button:hover { background: #EA6A0A !important; }
    [data-testid="stChatInput"] button svg { stroke: #fff !important; fill: #fff !important; }
    .tp-init-status {
      display: flex; align-items: center; justify-content: center; gap: 10px;
      color: var(--tp-text-muted); font-size: 14px; padding: 10px 0 4px;
    }
    .tp-init-spinner {
      width: 16px; height: 16px; border-radius: 50%;
      border: 2px solid rgba(255,255,255,.18); border-top-color: var(--tp-accent);
      animation: tp-spin .8s linear infinite;
    }
    @keyframes tp-spin { to { transform: rotate(360deg); } }

    /* "Assistant is typing" — three dots that rise in sequence. This is the ONLY waiting
       indicator: st.spinner's circle was removed so the two do not stack. */
    .tp-typing { display: flex; align-items: center; gap: 5px; padding: 6px 2px; }
    .tp-typing span {
      width: 7px; height: 7px; border-radius: 50%;
      background: var(--tp-text-muted);
      animation: tp-bounce 1.2s infinite ease-in-out;
    }
    .tp-typing span:nth-child(2) { animation-delay: .18s; }
    .tp-typing span:nth-child(3) { animation-delay: .36s; }
    @keyframes tp-bounce {
      0%, 60%, 100% { transform: translateY(0);     opacity: .35; }
      30%           { transform: translateY(-5px);  opacity: 1;   }
    }

    /* The one-line "let me go and look" note shown above a set of results. */
    /* Reads as an ordinary line of the assistant's reply: same colour, size, weight and
       style as the text under it. Only the spacing below is ours. */
    .tp-search-note {
      color: var(--tp-text);
      font-size: inherit; font-style: normal; font-weight: inherit; font-family: inherit;
      line-height: inherit;
      margin: 0 0 12px; padding: 0;
    }

    /* Remove chat message default background box */
    [data-testid="stChatMessage"] {
      background: transparent !important;
      box-shadow: none !important;
      border: none !important;
      padding: 3px 0 !important;
      gap: 8px !important;
    }
    [data-testid="stChatMessageContent"] { background: transparent !important; }
    [data-testid="stChatMessageContent"] p { color: var(--tp-text) !important; }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: #3A3F49; border-radius: 10px; }
  `;
  var styleEl = window.parent.document.createElement('style');
  styleEl.textContent = css;
  window.parent.document.head.appendChild(styleEl);
})();
</script>
""", height=0)

# Theme-aware polish layer. It follows the base skin so every Streamlit
# surface uses one semantic palette in both modes.
_dark = st.session_state.ui_dark_mode
_theme = {
    "bg": "#0B1018" if _dark else "#F5F7FA",
    "sidebar": "#101722" if _dark else "#FFFFFF",
    "surface": "#151E2B" if _dark else "#FFFFFF",
    "surface_2": "#1B2636" if _dark else "#F8FAFC",
    "text": "#F1F5F9" if _dark else "#172033",
    "muted": "#94A3B8" if _dark else "#64748B",
    "border": "rgba(148,163,184,.20)" if _dark else "#E2E8F0",
    "accent_soft": "rgba(249,115,22,.14)" if _dark else "#FFF1E8",
    "shadow": "rgba(0,0,0,.28)" if _dark else "rgba(15,23,42,.09)",
}
components.html(
    f"""
<script>
(function() {{
  const doc = window.parent.document;
  const activeTheme = "{'dark' if _dark else 'light'}";
  const parentWindow = window.parent;
  const themeUrl = new URL(parentWindow.location.href);
  const serverHadTheme = {str(_requested_theme in ('light', 'dark')).lower()};
  doc.documentElement.dataset.tpTheme = activeTheme;
  if (serverHadTheme) {{
    parentWindow.localStorage.setItem("trailerplace-ui-theme", activeTheme);
  }}
  if (serverHadTheme && themeUrl.searchParams.get("theme") !== activeTheme) {{
    themeUrl.searchParams.set("theme", activeTheme);
    parentWindow.history.replaceState({{}}, "", themeUrl.toString());
  }}
  const old = doc.getElementById('tp-theme-polish');
  if (old) old.remove();
  const style = doc.createElement('style');
  style.id = 'tp-theme-polish';
  style.textContent = `
    :root {{
      --tp-font: Inter, Aptos, "Segoe UI Variable", "Segoe UI", ui-sans-serif, system-ui, sans-serif;
      --tp-bg: {_theme['bg']}; --tp-secondary-bg: {_theme['surface']};
      --tp-card-bg: {_theme['surface']}; --tp-text: {_theme['text']};
      --tp-text-muted: {_theme['muted']}; --tp-border: {_theme['border']};
      --tp-accent: #F97316; --tp-accent-dim: {_theme['accent_soft']};
      --tp-shadow: {_theme['shadow']};
    }}
    html, body, [class*="css"] {{ font-family: var(--tp-font) !important; }}
    body, [data-testid="stAppViewContainer"], [data-testid="stMain"],
    [data-testid="stBottom"], [data-testid="stBottom"] > div,
    [data-testid="stBottomBlockContainer"], [data-testid="stChatInputContainer"] {{
      background: var(--tp-bg) !important; color: var(--tp-text) !important;
    }}
    [data-testid="stBottom"]::before, [data-testid="stBottom"]::after,
    [data-testid="stBottomBlockContainer"]::before, [data-testid="stBottomBlockContainer"]::after {{
      background: var(--tp-bg) !important;
    }}
    [data-testid="stAppViewContainer"] {{
      background-image: none !important;
    }}
    [data-testid="stSidebar"] {{
      background: {_theme['sidebar']} !important; border-right: 1px solid var(--tp-border) !important;
      box-shadow: 8px 0 28px rgba(15,23,42,.04) !important;
    }}
    [data-testid="stSidebar"] button {{ background: transparent !important; }}
    [data-testid="stSidebar"] button:hover {{ background: {_theme['surface_2']} !important; border-color: var(--tp-border) !important; }}
    [data-testid="stSidebar"] button[kind="primary"] {{
      background: var(--tp-accent) !important; color: white !important;
      box-shadow: 0 4px 14px rgba(249,115,22,.22) !important;
    }}
    [data-testid="stSidebarCollapseButton"] button,
    [data-testid="stSidebarCollapsedControl"] button {{
      color: {_theme['text']} !important;
      background: transparent !important;
      border-color: transparent !important;
      outline: none !important;
      box-shadow: none !important;
    }}
    [data-testid="stSidebarCollapseButton"] button span,
    [data-testid="stSidebarCollapsedControl"] button span {{
      color: {_theme['text']} !important;
      -webkit-text-fill-color: {_theme['text']} !important;
    }}
    [data-testid="stSidebarCollapseButton"] button svg,
    [data-testid="stSidebarCollapsedControl"] button svg {{
      color: {_theme['text']} !important;
      fill: currentColor !important;
      stroke: currentColor !important;
    }}
    [data-testid="stSidebarCollapsedControl"] button,
    [data-testid="stSidebarCollapsedControl"] button * {{
      color: {'#111827' if not _dark else '#F8FAFC'} !important;
      -webkit-text-fill-color: {'#111827' if not _dark else '#F8FAFC'} !important;
      opacity: 1 !important;
    }}
    [data-testid="stSidebarCollapsedControl"] button svg,
    [data-testid="stSidebarCollapsedControl"] button svg * {{
      color: {'#111827' if not _dark else '#F8FAFC'} !important;
      fill: {'#111827' if not _dark else '#F8FAFC'} !important;
      stroke: {'#111827' if not _dark else '#F8FAFC'} !important;
      opacity: 1 !important;
      filter: {'brightness(0)' if not _dark else 'brightness(0) invert(1)'} !important;
    }}
    [data-testid="stExpandSidebarButton"],
    [data-testid="stExpandSidebarButton"] * {{
      color: {'#111827' if not _dark else '#F8FAFC'} !important;
      -webkit-text-fill-color: {'#111827' if not _dark else '#F8FAFC'} !important;
      opacity: 1 !important;
    }}
    [data-testid="stExpandSidebarButton"] svg,
    [data-testid="stExpandSidebarButton"] svg * {{
      color: {'#111827' if not _dark else '#F8FAFC'} !important;
      fill: {'#111827' if not _dark else '#F8FAFC'} !important;
      stroke: {'#111827' if not _dark else '#F8FAFC'} !important;
      opacity: 1 !important;
      filter: {'brightness(0)' if not _dark else 'brightness(0) invert(1)'} !important;
    }}
    [data-testid="stSidebarCollapseButton"] button:hover,
    [data-testid="stSidebarCollapsedControl"] button:hover {{
      color: {_theme['text']} !important;
      background: {_theme['surface_2']} !important;
      border-color: var(--tp-border) !important;
    }}
    [data-testid="stSidebar"] [data-testid="stCheckbox"] {{ padding: .25rem .1rem .7rem; }}
    [data-testid="stSidebar"] [data-testid="stCheckbox"] label p {{
      color: var(--tp-text-muted) !important; font-size: .82rem !important; font-weight: 600 !important;
    }}
    /* Streamlit renders st.toggle through its checkbox primitive. Give the
       switch explicit geometry and contrast so light mode never washes out. */
    [data-testid="stCheckbox"] label[data-baseweb="checkbox"] > div:first-of-type {{
      width: 48px !important; min-width: 48px !important;
      height: 28px !important; min-height: 28px !important;
      margin: 0 !important; padding: 3px !important;
      box-sizing: border-box !important;
      border: 1px solid {'#475569' if _dark else '#B8BEC8'} !important;
      border-radius: 999px !important;
      background: {'#334155' if _dark else '#D8DAE0'} !important;
      box-shadow: inset 0 1px 2px rgba(15,23,42,.13) !important;
    }}
    [data-testid="stCheckbox"] label[data-baseweb="checkbox"] > div:first-of-type > div {{
      width: 20px !important; height: 20px !important;
      border: 1px solid rgba(15,23,42,.12) !important;
      border-radius: 50% !important; background: #FFFFFF !important;
      box-shadow: 0 1px 3px rgba(15,23,42,.28) !important;
      transform: translateX(0) !important;
    }}
    [data-testid="stCheckbox"] label[data-baseweb="checkbox"]:has(input:checked) > div:first-of-type {{
      border-color: #EA580C !important; background: #F97316 !important;
    }}
    [data-testid="stCheckbox"] label[data-baseweb="checkbox"]:has(input:checked) > div:first-of-type > div {{
      transform: translateX(20px) !important;
    }}
    [data-testid="stTooltipIcon"] button {{
      width: 22px !important; min-width: 22px !important; height: 22px !important;
      padding: 0 !important; border: 0 !important; border-radius: 50% !important;
      background: transparent !important; box-shadow: none !important;
    }}
    [data-testid="stTooltipIcon"] svg {{
      width: 16px !important; height: 16px !important;
      color: {_theme['muted']} !important; stroke: {_theme['muted']} !important;
      opacity: 1 !important;
    }}
    [data-testid="stSpinner"], [data-testid="stSpinner"] > div {{
      color: {_theme['text']} !important;
      opacity: 1 !important;
    }}
    [data-testid="stSpinner"] svg {{
      color: {_theme['text']} !important;
      fill: {_theme['text']} !important;
      opacity: 1 !important;
    }}
    [data-testid="stSpinner"] svg path {{
      fill: currentColor !important;
    }}
    [data-testid="stSpinner"] i {{
      border-color: {'rgba(241,245,249,.22)' if _dark else 'rgba(23,32,51,.18)'} !important;
      border-top-color: {_theme['text']} !important;
      opacity: 1 !important;
    }}
    .block-container {{ max-width: 960px !important; padding: 2rem 2.25rem 7.5rem !important; }}
    .tp-header-eyebrow {{ color: var(--tp-accent) !important; letter-spacing: 1.35px !important; }}
    .tp-header-title {{ font-size: clamp(1.7rem, 3vw, 2.25rem) !important; font-weight: 720 !important; }}
    .tp-header-sub {{ font-size: .92rem !important; line-height: 1.6 !important; }}
    .tp-hazard {{ height: 2px !important; margin-top: 16px !important; background: linear-gradient(90deg, var(--tp-accent), rgba(249,115,22,.12), transparent) !important; }}
    .tp-plate {{ font-size: 1.28rem !important; letter-spacing: -.02em !important; }}
    .tp-plate-sub {{ color: var(--tp-text-muted) !important; letter-spacing: .11em !important; }}
    [data-testid="stChatMessage"] {{ padding: .65rem .75rem !important; margin: .4rem 0 !important; border-radius: 16px !important; }}
    [data-testid="stChatMessage"][aria-label="user"],
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {{
      background: var(--tp-accent-dim) !important; margin-left: clamp(1rem, 12vw, 7rem) !important;
    }}
    [data-testid="stChatMessage"][aria-label="assistant"],
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {{
      background: {_theme['surface']} !important; border: 1px solid var(--tp-border) !important;
      box-shadow: 0 5px 18px var(--tp-shadow) !important; margin-right: clamp(0rem, 5vw, 3rem) !important;
    }}
    [data-testid="stChatMessageContent"] p {{ line-height: 1.65 !important; }}
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"],
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] li,
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] strong,
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] b,
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] em,
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] blockquote,
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] h1,
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] h2,
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] h3,
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] h4 {{
      color: {_theme['text']} !important;
    }}
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] a {{
      color: {'#60A5FA' if _dark else '#2563EB'} !important;
      text-decoration-color: currentColor !important;
    }}
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] hr {{
      height: 0 !important;
      margin: 1.75rem 0 !important;
      border: 0 !important;
      border-top: 1px solid {_theme['border']} !important;
      background: transparent !important;
      opacity: 1 !important;
    }}
    [data-testid="stMain"] [data-testid="stMarkdownContainer"] h1,
    [data-testid="stMain"] [data-testid="stMarkdownContainer"] h2,
    [data-testid="stMain"] [data-testid="stMarkdownContainer"] h3,
    [data-testid="stMain"] [data-testid="stMarkdownContainer"] h4,
    [data-testid="stMain"] [data-testid="stMarkdownContainer"] h5,
    [data-testid="stMain"] [data-testid="stMarkdownContainer"] h6 {{
      color: {_theme['text']} !important;
    }}
    [data-testid="stMain"] [data-testid="stMarkdownContainer"] code {{
      padding: .12rem .38rem !important;
      color: {'#FDBA74' if _dark else '#C2410C'} !important;
      background: {_theme['surface_2']} !important;
      border: 1px solid {_theme['border']} !important;
      border-radius: 5px !important;
      font-family: "Cascadia Code", "SFMono-Regular", Consolas, monospace !important;
      font-size: .88em !important;
    }}
    [data-testid="stMain"] [data-testid="stMarkdownContainer"] pre {{
      color: {_theme['text']} !important;
      background: {_theme['surface_2']} !important;
      border: 1px solid {_theme['border']} !important;
      border-radius: 10px !important;
    }}
    [data-testid="stMain"] [data-testid="stMarkdownContainer"] pre code {{
      padding: 0 !important;
      color: {_theme['text']} !important;
      background: transparent !important;
      border: 0 !important;
    }}
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] code {{
      padding: 0 !important;
      margin: 0 !important;
      color: inherit !important;
      background: transparent !important;
      border: 0 !important;
      border-radius: 0 !important;
      font-family: inherit !important;
      font-size: inherit !important;
      font-weight: inherit !important;
      letter-spacing: inherit !important;
    }}
    [data-testid="stChatInput"] > div {{
      background: {_theme['surface']} !important; border: 1px solid var(--tp-border) !important;
      border-radius: 18px !important; box-shadow: 0 12px 36px var(--tp-shadow) !important;
    }}
    [data-testid="stChatInputTextArea"],
    [data-testid="stChatInputTextArea"] textarea,
    [data-testid="stChatInput"] [data-baseweb="textarea"] {{
      background: {_theme['surface']} !important;
      background-color: {_theme['surface']} !important;
      color: {_theme['text']} !important;
    }}
    [data-testid="stChatInput"] textarea[data-testid="stChatInputTextArea"],
    [data-testid="stBottom"] [data-testid="stChatInput"] textarea {{
      background: {_theme['surface']} !important;
      background-color: {_theme['surface']} !important;
      color: {_theme['text']} !important;
      caret-color: {_theme['text']} !important;
      -webkit-text-fill-color: {_theme['text']} !important;
      opacity: 1 !important;
    }}
    [data-testid="stChatInputTextArea"]::placeholder,
    [data-testid="stChatInputTextArea"] textarea::placeholder {{
      color: {_theme['muted']} !important;
      opacity: 1 !important;
    }}
    [data-testid="stChatInput"] > div:focus-within {{
      border-color: rgba(249,115,22,.65) !important;
      box-shadow: 0 0 0 3px {_theme['accent_soft']}, 0 12px 36px var(--tp-shadow) !important;
    }}
    [data-testid="stChatInput"] [data-testid="stChatInputSubmitButton"],
    [data-testid="stChatInput"] button[data-testid="stChatInputSubmitButton"] {{
      width: 82px !important; min-width: 82px !important;
      height: 40px !important; min-height: 40px !important;
      padding: 0 14px !important;
      display: inline-flex !important; align-items: center !important; justify-content: center !important;
      background: {_theme['surface_2']} !important;
      color: {_theme['text']} !important;
      border: 0 !important; outline: 0 !important;
      border-radius: 8px !important; box-shadow: none !important;
      font-size: 0 !important;
    }}
    [data-testid="stChatInput"] [data-testid="stChatInputSubmitButton"] svg {{
      display: none !important;
    }}
    [data-testid="stChatInput"] [data-testid="stChatInputSubmitButton"]::before {{
      content: "→  Ask";
      display: block;
      color: {_theme['text']};
      font-family: var(--tp-font);
      font-size: 14px;
      font-weight: 650;
      line-height: 1;
      white-space: pre;
    }}
    [data-testid="stChatInput"] [data-testid="stChatInputSubmitButton"]:hover {{
      background: {'#263447' if _dark else '#EEF2F6'} !important;
      border: 0 !important; outline: 0 !important; box-shadow: none !important;
    }}
    [data-testid="stChatInput"] [data-testid="stChatInputSubmitButton"]:focus,
    [data-testid="stChatInput"] [data-testid="stChatInputSubmitButton"]:focus-visible,
    [data-testid="stChatInput"] [data-testid="stChatInputSubmitButton"]:active {{
      border: 0 !important; outline: 0 !important; box-shadow: none !important;
    }}
    [data-testid="stChatInput"] textarea:disabled,
    [data-testid="stChatInputTextArea"]:disabled,
    [data-testid="stChatInputTextArea"] textarea:disabled,
    [data-testid="stChatInput"]:has(textarea:disabled) textarea,
    [data-testid="stChatInput"]:has(textarea:disabled) [data-testid="stChatInputTextArea"],
    [data-testid="stChatInput"]:has(textarea:disabled) [data-baseweb="textarea"] {{
      cursor: not-allowed !important;
      opacity: 1 !important;
      background: {_theme['surface']} !important;
      background-color: {_theme['surface']} !important;
      -webkit-text-fill-color: {_theme['muted']} !important;
      color: {_theme['muted']} !important;
    }}
    [data-testid="stChatInput"]:has(textarea:disabled) > div {{
      opacity: 1 !important;
      background: {_theme['surface']} !important;
      background-color: {_theme['surface']} !important;
      cursor: not-allowed !important;
      box-shadow: 0 12px 36px var(--tp-shadow) !important;
    }}
    [data-testid="stChatInput"]:has(textarea:disabled) [data-testid="stChatInputSubmitButton"] {{
      opacity: 0.45 !important;
      pointer-events: none !important;
      cursor: not-allowed !important;
    }}
    [data-testid="stTextInput"] input, [data-testid="stTextArea"] textarea {{
      background: {_theme['surface_2']} !important;
      color: {_theme['text']} !important;
      caret-color: {_theme['text']} !important;
      -webkit-text-fill-color: {_theme['text']} !important;
    }}
    /* Browser saved-info / autofill paints its own colors unless overridden. */
    [data-testid="stTextInput"] input:-webkit-autofill,
    [data-testid="stTextInput"] input:-webkit-autofill:hover,
    [data-testid="stTextInput"] input:-webkit-autofill:focus,
    [data-testid="stTextInput"] input:-webkit-autofill:active,
    [data-testid="stTextArea"] textarea:-webkit-autofill,
    [data-testid="stTextArea"] textarea:-webkit-autofill:hover,
    [data-testid="stTextArea"] textarea:-webkit-autofill:focus,
    [data-testid="stTextArea"] textarea:-webkit-autofill:active {{
      -webkit-box-shadow: 0 0 0 1000px {_theme['surface_2']} inset !important;
      box-shadow: 0 0 0 1000px {_theme['surface_2']} inset !important;
      -webkit-text-fill-color: {_theme['text']} !important;
      caret-color: {_theme['text']} !important;
      border-color: {_theme['border']} !important;
      transition: background-color 99999s ease-out 0s;
    }}
    [data-testid="stTextInputRootElement"]:has(input:-webkit-autofill),
    [data-testid="stTextInputRootElement"]:has(input:-webkit-autofill:focus) {{
      background: {_theme['surface_2']} !important;
      background-color: {_theme['surface_2']} !important;
    }}
    [data-testid="stTextInput"] input:autofill,
    [data-testid="stTextArea"] textarea:autofill {{
      box-shadow: 0 0 0 1000px {_theme['surface_2']} inset !important;
      -webkit-text-fill-color: {_theme['text']} !important;
      caret-color: {_theme['text']} !important;
    }}
    [data-testid="stTextInput"] [data-testid="stWidgetLabel"] p {{
      color: {_theme['text']} !important;
    }}
    [data-testid="stTextInputRootElement"],
    [data-testid="stTextInputRootElement"][data-baseweb="input"] {{
      background: {_theme['surface_2']} !important;
      background-color: {_theme['surface_2']} !important;
      overflow: hidden !important;
    }}
    [data-testid="stTextInputRootElement"]:focus-within {{
      border-color: #F97316 !important;
      box-shadow: 0 0 0 3px {_theme['accent_soft']} !important;
    }}
    [data-testid="stTextInputRootElement"]:has(input[type="password"]) > div:last-child {{
      background: {_theme['surface_2']} !important;
      background-color: {_theme['surface_2']} !important;
      color: {_theme['text']} !important;
      border-left: 1px solid {_theme['border']} !important;
    }}
    [data-testid="stTextInputRootElement"]:has(input[type="password"]) > div:last-child > * {{
      background: transparent !important;
      background-color: transparent !important;
    }}
    [data-testid="stTextInputRootElement"]:has(input[type="password"]) > div:last-child button {{
      background: transparent !important; border: 0 !important;
      color: {_theme['text']} !important; box-shadow: none !important;
    }}
    [data-testid="stTextInputRootElement"]:has(input[type="password"]) > div:last-child svg,
    [data-testid="stTextInputRootElement"]:has(input[type="password"]) > div:last-child svg path {{
      color: {_theme['text']} !important; fill: currentColor !important;
    }}
    /* Keep the login form visually quiet: Streamlit adds this focus-only
       keyboard hint, while Edge/WebKit may add a second password reveal UI. */
    [data-testid="InputInstructions"] {{ display: none !important; }}
    [data-testid="stTextInput"] input[type="password"]::-ms-reveal,
    [data-testid="stTextInput"] input[type="password"]::-ms-clear {{
      display: none !important; width: 0 !important; height: 0 !important;
    }}
    [data-testid="stTextInput"] input[type="password"]::-webkit-credentials-auto-fill-button,
    [data-testid="stTextInput"] input[type="password"]::-webkit-contacts-auto-fill-button {{
      visibility: hidden !important; display: none !important;
      pointer-events: none !important; position: absolute !important; right: 0 !important;
    }}
    [data-testid="stVerticalBlockBorderWrapper"] {{ background: {_theme['surface']} !important; border-radius: 14px !important; }}
    [data-testid="stChatMessage"] [data-testid="stVerticalBlockBorderWrapper"],
    [data-testid="stChatMessage"] [data-testid="stForm"] {{
      background: {_theme['surface']} !important;
      border: 1px solid {'#334155' if _dark else '#CBD5E1'} !important;
      border-radius: 12px !important;
    }}
    [data-testid="stChatMessage"] [data-testid="stTextArea"] textarea {{
      border-color: {'#475569' if _dark else '#94A3B8'} !important;
    }}
    [data-testid="stFormSubmitButton"] button {{
      background: #F97316 !important; color: #FFFFFF !important;
      border: 1px solid #F97316 !important;
      box-shadow: 0 5px 14px rgba(249,115,22,.20) !important;
    }}
    [data-testid="stFormSubmitButton"] button:hover {{
      background: #EA580C !important; border-color: #EA580C !important;
    }}
    [data-testid="stFormSubmitButton"] button:focus,
    [data-testid="stFormSubmitButton"] button:focus-visible {{
      outline: none !important;
      box-shadow: 0 0 0 3px {_theme['accent_soft']}, 0 5px 14px rgba(249,115,22,.20) !important;
    }}
    [data-testid="stChatMessage"] [data-testid="stFormSubmitButton"] button {{
      background: {_theme['surface_2']} !important;
      color: {_theme['text']} !important;
      border: 1px solid {'#475569' if _dark else '#CBD5E1'} !important;
      box-shadow: none !important;
    }}
    [data-testid="stChatMessage"] [data-testid="stFormSubmitButton"] button:hover {{
      background: {'#263447' if _dark else '#EEF2F6'} !important;
      color: {_theme['text']} !important;
      border-color: {'#475569' if _dark else '#CBD5E1'} !important;
    }}
    [data-testid="stChatMessage"] [data-testid="stFormSubmitButton"] button:focus,
    [data-testid="stChatMessage"] [data-testid="stFormSubmitButton"] button:focus-visible {{
      outline: none !important;
      box-shadow: 0 0 0 3px {'rgba(148,163,184,.16)' if _dark else 'rgba(100,116,139,.12)'} !important;
    }}
    .trailerplace-insights {{ color: var(--tp-text-muted) !important; background: {_theme['surface_2']} !important; border-color: var(--tp-border) !important; }}
    [data-testid="stAlert"] {{ border-radius: 12px !important; }}
    @media (max-width: 700px) {{
      .block-container {{ padding: 1.25rem 1rem 7rem !important; }}
      [data-testid="stChatMessage"] {{ margin-left: 0 !important; margin-right: 0 !important; }}
    }}
  `;
  doc.head.appendChild(style);
}})();
</script>
""",
    height=0,
)


# ─────────────────────────────────────────────────────────────
# TRAILER CARD — inline styles only, no class dependencies
# Styled as a work-order / spec sheet: bright paper against the
# dark dispatch-desk background, hazard-stripe across the top edge.
# ─────────────────────────────────────────────────────────────
def _format_type_for_card(category_subcategory: str) -> str:
    """Storage may be 'Category > Subcategory'; the card shows only the main category."""
    s = (category_subcategory or "").strip()
    if not s:
        return s
    return s.split(" > ")[0].strip()


def render_card(listing: TrailerListing, rank: int):
    dark = st.session_state.get("ui_dark_mode", True)
    card_bg = "#151E2B" if dark else "#FFFFFF"
    card_text = "#F1F5F9" if dark else "#172033"
    card_muted = "#94A3B8" if dark else "#64748B"
    card_border = "rgba(148,163,184,.20)" if dark else "#E2E8F0"
    card_rule = "#263447" if dark else "#EEF2F6"
    price_str = (
        listing.price_display
        or (f"${listing.price:,.0f}" if listing.price is not None else "Call for price")
    )
    badge_bg  = "#DCFCE7" if listing.condition == "New" else "#FEF9C3"
    badge_fg  = "#15803D" if listing.condition == "New" else "#A16207"

    specs = [
        ("Make",     listing.make),
        ("Year",     listing.year),
        ("Type",     _format_type_for_card(listing.category_subcategory)),
        ("Hitch",    listing.hitch_type),
        ("Color",    listing.color),
        ("Length",   listing.length),
        ("Width",    listing.width),
        ("GVWR",     listing.gvwr),
        ("Axles",    listing.axles),
        ("Material", listing.trailer_material),
        ("Floor",    listing.floor),
        ("Payload",  listing.payload_capacity),
    ]
    spec_cells = "".join(
        f'<div style="min-width:88px;">'
        f'<div style="font-size:10px;text-transform:uppercase;letter-spacing:.6px;color:{card_muted};font-weight:600;font-family:var(--tp-font);">{lbl}</div>'
        f'<div style="font-size:13px;font-weight:600;color:{card_text};font-family:var(--tp-font);">{val}</div>'
        f'</div>'
        for lbl, val in specs if val
    )
    pay_html = (
        f'<div style="font-size:12px;color:{card_muted};margin-top:3px;font-family:var(--tp-font);">'
        f'Payments from <b style="color:{card_text};">{listing.payments_from}</b></div>'
    ) if listing.payments_from else ""

    # No line may start with 4+ spaces — Streamlit Markdown treats that as a code block
    # and would render literal tags like </div> in a monospace box.
    st.markdown(
        f'<div style="background:{card_bg};border:1px solid {card_border};border-radius:14px;'
        f'overflow:hidden;margin:12px 0 6px 0;box-shadow:0 8px 24px var(--tp-shadow);">'
        f'<div style="height:3px;background:linear-gradient(90deg,#F97316,#FB923C,transparent);"></div>'
        f'<div style="padding:14px 18px 16px 18px;">'
        f'<div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:8px;">'
        f'<div>'
        f'<div style="font-family:var(--tp-font);font-size:10.5px;letter-spacing:.8px;'
        f'color:{card_muted};text-transform:uppercase;margin-bottom:4px;">Listing #{rank:02d}</div>'
        f'<div style="font-size:15px;font-weight:700;color:{card_text};margin-bottom:5px;font-family:var(--tp-font);">'
        f"{listing.title}</div>"
        f'<span style="display:inline-block;padding:2px 10px;border-radius:20px;'
        f"background:{badge_bg};color:{badge_fg};font-size:11px;font-weight:600;"
        f'font-family:var(--tp-font);">{listing.condition}</span></div>'
        f'<div style="text-align:right;">'
        f'<div style="font-size:21px;font-weight:700;color:#F97316;font-family:var(--tp-font);">{price_str}</div>'
        f"{pay_html}</div></div>"
        f'<div style="border-top:1px solid {card_rule};margin:12px 0;"></div>'
        f'<div style="display:flex;flex-wrap:wrap;gap:14px 20px;">{spec_cells}</div>'
        f'<a href="{listing.url}" target="_blank" '
        f'style="display:inline-block;margin-top:14px;background:#F97316;color:#FFFFFF;'
        f"padding:8px 18px;border-radius:8px;font-size:13px;font-weight:600;letter-spacing:.2px;"
        f'text-decoration:none;font-family:var(--tp-font);">'
        f"View Full Listing &rarr;</a></div></div>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────
if "auth_ok" not in st.session_state:
    st.session_state.auth_ok = False
if "messages" not in st.session_state:
    st.session_state.messages = []
if "sales_phase" not in st.session_state:
    st.session_state.sales_phase = "main"
if "onboarding_api_messages" not in st.session_state:
    st.session_state.onboarding_api_messages = []
if "app_page" not in st.session_state:
    st.session_state.app_page = "Chatbot"
def _render_header(title: str, subtitle: str) -> None:
    st.markdown(
        '<div class="tp-header-eyebrow"><span class="dot"></span>TRAILERPLACE · LIVE CHAT</div>'
        f'<div class="tp-header-title">{title}</div>'
        f'<div class="tp-header-sub">{subtitle}</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="tp-hazard"></div>', unsafe_allow_html=True)


def _start_backend_initialization() -> None:
    st.session_state.backend_ready_status = "initializing"
    st.session_state.backend_ready_error = ""
    st.session_state.backend_ready_future = _BACKEND_EXECUTOR.submit(
        _wait_for_backend_ready
    )


# ─────────────────────────────────────────────────────────────
# LOGIN (env: TRAILERPLACE_APP_USERNAME + TRAILERPLACE_APP_PASSWORD)
# ─────────────────────────────────────────────────────────────
if not _AUTH_CONFIGURED:
    st.error(
        "App login is not configured. Add **TRAILERPLACE_APP_USERNAME** and "
        "**TRAILERPLACE_APP_PASSWORD** to your `.env` file (both non-empty), then restart."
    )
    st.stop()

if not st.session_state.auth_ok:
    with st.sidebar:
        st.markdown(
            '<div class="tp-plate">🚛 TrailerPlace</div>'
            '<div class="tp-plate-sub">Sign in to continue</div>',
            unsafe_allow_html=True,
        )
        st.toggle(
            "Dark mode",
            key="ui_dark_mode",
            help="Switch between the light and dark workspace themes.",
        )
    _render_header("Sales Chat", "Sign in to use the assistant")
    with st.form("app_login"):
        u = st.text_input("Username", autocomplete="username")
        p = st.text_input("Password", type="password", autocomplete="current-password")
        submitted = st.form_submit_button("Sign in", use_container_width=True, type="primary")
        if submitted:
            if u.strip() == _AUTH_USER and _password_matches(p, _AUTH_PASS):
                st.session_state.auth_ok = True
                _start_backend_initialization()
                st.rerun()
            else:
                st.error("Incorrect username or password.")
    st.stop()


requested_chat_session = st.query_params.get("chat_session")
if (
    "chat_session_id" not in st.session_state
    or (requested_chat_session and requested_chat_session != st.session_state.chat_session_id)
):
    st.session_state.chat_session_id = requested_chat_session or str(uuid.uuid4())
    st.query_params["chat_session"] = st.session_state.chat_session_id
    st.session_state.pop("durable_session_restored", None)
if "backend_ready_status" not in st.session_state:
    _start_backend_initialization()
backend_ready_future = st.session_state.get("backend_ready_future")
if (
    st.session_state.get("backend_ready_status") == "initializing"
    and isinstance(backend_ready_future, Future)
    and backend_ready_future.done()
):
    backend_ready_result = backend_ready_future.result()
    st.session_state.backend_ready_status = backend_ready_result.get("status", "error")
    st.session_state.backend_ready_error = backend_ready_result.get("error", "")
    st.session_state.backend_ready_future = None

# A sleeping server must be awake before durable state is requested. Do not
# render an empty/stale chatroom while initialization or restoration is still
# in progress.
if st.session_state.get("backend_ready_status") == "initializing":
    st.markdown(
        '<div class="tp-init-status"><span class="tp-init-spinner"></span>'
        '<span>Loading conversation…</span></div>',
        unsafe_allow_html=True,
    )
    # Second source of dead time after sign-in: the readiness thread can finish moments after
    # this rerun starts, and nothing notices until the next one. Kept short so the wait tracks
    # the backend's actual boot rather than this loop's cadence.
    time.sleep(0.4)
    st.rerun()

if (
    st.session_state.get("backend_ready_status") == "ready"
    and not st.session_state.get("durable_session_restored")
):
    restored = _restore_api_session(st.session_state.chat_session_id)
    if restored is None:
        st.session_state.backend_ready_status = "error"
        st.session_state.backend_ready_error = "Conversation could not be restored."
        st.rerun()
    # A closed conversation still restores: "New Conversation" and "Log out" always move the
    # URL to a fresh session id, so a URL pointing at a closed session is the user coming back
    # to that conversation on purpose — show them their chat, don't blank the screen.
    if restored.get("exists"):
        st.session_state.messages = restored.get("messages") or []
        st.session_state.sales_phase = restored.get("sales_phase") or "main"
        for key in ("customer_full_name", "customer_email", "customer_phone"):
            if restored.get(key) is not None:
                st.session_state[key] = restored[key]
    logging.getLogger("trailerplace.app").info(
        "session restore: sid=%s exists=%s closed=%s messages=%d",
        st.session_state.chat_session_id,
        bool(restored.get("exists")),
        bool(restored.get("closed")),
        len(restored.get("messages") or []),
    )
    st.session_state.durable_session_restored = True
if "last_thinking_result" not in st.session_state:
    st.session_state.last_thinking_result = None
if "thinking_status" not in st.session_state:
    st.session_state.thinking_status = "idle"
if "thinking_future" not in st.session_state:
    st.session_state.thinking_future = None
if "last_thinking_payload" not in st.session_state:
    st.session_state.last_thinking_payload = None
if "chat_awaiting_response" not in st.session_state:
    st.session_state.chat_awaiting_response = False
if "pending_turn_id" not in st.session_state:
    st.session_state.pending_turn_id = None

thinking_future = st.session_state.get("thinking_future")
if isinstance(thinking_future, Future) and thinking_future.done():
    try:
        _thinking_result = thinking_future.result()
        st.session_state.last_thinking_result = _thinking_result
        st.session_state.thinking_status = "done"
    except Exception as exc:
        _thinking_result = {
            "status": "error",
            "error": str(exc),
            "thinking_markdown": "",
        }
        st.session_state.last_thinking_result = _thinking_result
        st.session_state.thinking_status = "error"
    # Attach result to the last assistant message so it renders inline.
    for _j in range(len(st.session_state.messages) - 1, -1, -1):
        if st.session_state.messages[_j]["role"] == "assistant":
            st.session_state.messages[_j]["thinking_result"] = _thinking_result
            break
    st.session_state.thinking_future = None
elif (
    isinstance(thinking_future, Future)
    and st.session_state.get("thinking_status") == "pending"
):
    # Auto-refresh while background thinking generation is running so the
    # result appears without requiring the user to send another message.
    time.sleep(max(0.2, _THINKING_POLL_MS / 1000.0))
    st.rerun()


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<div class="tp-plate">🚛 TrailerPlace</div>'
        '<div class="tp-plate-sub">Sales Desk · Wharton, TX</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="tp-hazard" style="margin:12px 0 16px 0;"></div>', unsafe_allow_html=True)
    st.toggle(
        "Dark mode",
        key="ui_dark_mode",
        help="Switch between the light and dark workspace themes.",
    )
    st.markdown('<div class="tp-nav-row">', unsafe_allow_html=True)
    nav_cols = st.columns(2)
    with nav_cols[0]:
        if st.button(
            "💬 Chatbot",
            use_container_width=True,
            type=("primary" if st.session_state.app_page == "Chatbot" else "secondary"),
        ):
            st.session_state.app_page = "Chatbot"
            st.rerun()
    with nav_cols[1]:
        if st.button(
            "📋 Rules",
            use_container_width=True,
            type=("primary" if st.session_state.app_page == "Rules" else "secondary"),
        ):
            st.session_state.app_page = "Rules"
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)
    st.divider()
    st.markdown(
        '<div class="tp-info-row">📍 <b>Wharton, TX</b></div>'
        '<div class="tp-info-row">📞 <b>(979) 532-1486</b></div>'
        '<div class="tp-info-row">💳 <b>Financing available</b></div>'
        '<div class="tp-info-row">🚚 <b>Delivery available</b></div>',
        unsafe_allow_html=True,
    )
    st.divider()
    if st.button("↺  New Conversation", use_container_width=True):
        old_sid = st.session_state.get("chat_session_id")
        if old_sid:
            _reset_api_session(old_sid)
        st.session_state.sales_phase = "main"
        st.session_state.onboarding_api_messages = []
        st.session_state.messages = []
        for k in ("main_prior_messages", "customer_full_name", "customer_email", "customer_phone"):
            if k in st.session_state:
                del st.session_state[k]
        st.session_state.chat_session_id = str(uuid.uuid4())
        st.query_params["chat_session"] = st.session_state.chat_session_id
        st.session_state.pop("durable_session_restored", None)
        st.session_state.last_thinking_result = None
        st.session_state.thinking_status = "idle"
        st.session_state.thinking_future = None
        st.session_state.last_thinking_payload = None
        st.session_state.chat_awaiting_response = False
        st.session_state.pending_turn_id = None
        _start_backend_initialization()
        st.rerun()
    if st.button("Log out", use_container_width=True):
        old_sid = st.session_state.get("chat_session_id")
        if old_sid:
            _reset_api_session(old_sid)
        st.session_state.auth_ok = False
        st.query_params["chat_session"] = str(uuid.uuid4())
        st.session_state.messages = []
        for k in (
            "chat_session_id",
            "last_thinking_result",
            "thinking_status",
            "thinking_future",
            "last_thinking_payload",
            "sales_phase",
            "onboarding_api_messages",
            "main_prior_messages",
            "customer_full_name",
            "customer_email",
            "customer_phone",
            "backend_ready_status",
            "backend_ready_error",
            "backend_ready_future",
            "chat_awaiting_response",
            "pending_turn_id",
            "durable_session_restored",
        ):
            if k in st.session_state:
                del st.session_state[k]
        st.rerun()

if st.session_state.get("app_page") == "Rules":
    _render_rules_page()
    st.stop()


# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────
_render_header(
    "Sales Chat",
    "Ask about any trailer in our inventory — we'll pull real listings as we go.",
)


# ─────────────────────────────────────────────────────────────
# RENDER HISTORY
# ─────────────────────────────────────────────────────────────
if not st.session_state.messages:
    _welcome_hint = (
        "Tell us what you're looking for — trailers, towing needs, or budget — and we'll help you find a match. "
        "You can share contact details if you'd like, but it isn't required to start."
    )
    st.markdown(
        f"""
<div style="text-align:center;padding:56px 20px 36px 20px;">
  <div style="font-size:34px;margin-bottom:10px;">🚛</div>
  <div class="tp-header-title" style="font-size:20px;margin-bottom:8px;">Welcome to TrailerPlace</div>
  <div style="font-size:14px;color:var(--tp-text-muted);max-width:380px;margin:0 auto;line-height:1.65;font-family:var(--tp-font);">
    {_welcome_hint}
  </div>
</div>
""",
        unsafe_allow_html=True,
    )
else:
    for i, msg in enumerate(st.session_state.messages):
        # The search note, the text (as its per-trailer bubbles) and the cards are all drawn
        # by this call; what follows hangs off the LAST bubble it made.
        with _render_message_bubbles(msg):
            if msg.get("role") == "assistant":
                prev_fb = msg.get("user_feedback")
                if isinstance(prev_fb, dict):
                    default_txt = (prev_fb.get("text") or "").strip()
                else:
                    default_txt = (prev_fb or "") if isinstance(prev_fb, str) else ""
                # Avoid st.expander / st.popover: they use Material icon fonts; when the font
                # fails, names like "expand_more" render as text on top of the label.
                open_key = f"feedback_open_{i}"
                if not st.session_state.get(open_key, False):
                    if st.button(
                        "Optional feedback",
                        key=f"fb_open_{i}",
                        use_container_width=True,
                    ):
                        st.session_state[open_key] = True
                        st.rerun()
                else:
                    with st.container(border=True):
                        st.caption("How was this response? (optional — helps us improve.)")
                        with st.form(f"user_feedback_{i}"):
                            fb = st.text_area(
                                "Your notes",
                                value=default_txt,
                                height=88,
                                placeholder="Something off, or what we should do next time…",
                            )
                            c1, c2 = st.columns(2)
                            with c1:
                                do_save = st.form_submit_button(
                                    "Save feedback",
                                    use_container_width=True,
                                    type="primary",
                                )
                            with c2:
                                do_close = st.form_submit_button(
                                    "Close", use_container_width=True
                                )
                            if do_close:
                                st.session_state[open_key] = False
                                st.rerun()
                            if do_save:
                                txt = (fb or "").strip()
                                st.session_state[open_key] = False
                                st.session_state.messages[i] = {
                                    **msg,
                                    "user_feedback": txt or None,
                                }
                                if st.session_state.get("chat_session_id") and persistence_enabled():
                                    t_iso = (
                                        datetime.now(timezone.utc)
                                        .replace(microsecond=0)
                                        .isoformat()
                                    )
                                    turn_idx = i // 2
                                    enqueue_save_user_feedback(
                                        st.session_state.chat_session_id,
                                        turn_idx,
                                        txt,
                                        t_iso,
                                    )
                                st.rerun()

                # ── Inline thinking note (every turn) ─────────────────────
                thinking_result = msg.get("thinking_result")
                is_last_msg = (i == len(st.session_state.messages) - 1)

                if thinking_agent_enabled():
                    if is_last_msg and st.session_state.get("thinking_status") == "pending":
                        st.markdown(
                            '<div style="color:#9CA3AF;font-size:12px;margin-top:4px;font-style:italic;">Insights…</div>',
                            unsafe_allow_html=True,
                        )
                    elif thinking_result:
                        th_text = (thinking_result.get("thinking_markdown") or "").strip()
                        if th_text:
                            safe = html.escape(th_text)
                            st.markdown(
                                """
<div class="trailerplace-insights" style="max-height: min(50vh, 22rem); overflow-y: auto;
  -webkit-overflow-scrolling: touch; color: #9CA3AF; font-size: 12px; margin-top: 6px; line-height: 1.55;
  padding: 10px 12px; font-style: italic; background: rgba(0,0,0,0.18); border-radius: 8px;
  border: 1px solid #374151; box-sizing: border-box; margin-bottom: 8px;">
  <div style="white-space: pre-wrap; word-wrap: break-word; overflow-wrap: anywhere;">Insights: """
                                + safe
                                + """</div>
</div>
""",
                                unsafe_allow_html=True,
                            )


# ─────────────────────────────────────────────────────────────
# CHAT INPUT
# ─────────────────────────────────────────────────────────────
def _show_waiting_bubble(slot, note: str | None) -> None:
    """Draw (or redraw) the whole waiting bubble inside `slot`.

    The bubble is redrawn rather than patched in place because `slot.empty()` has to be able
    to take the CHAT BUBBLE away too, not just its contents - once the reply starts arriving
    in bubbles of its own, an empty avatar left hanging above them is a visible artefact.
    """
    with slot.container():
        with st.chat_message("assistant"):
            st.markdown(_waiting_bubble_html(note), unsafe_allow_html=True)


def _run_blocking_turn(payload: dict, slot) -> dict:
    """POST /chat and wait. The fallback when streaming is off or unavailable.

    The turn runs on a worker thread so this one stays free to poll the backend and update
    the bubble; a plain blocking post() cannot show anything until the whole turn is done,
    which is the entire point of the status line.
    """
    turn_future = _CHAT_EXECUTOR.submit(
        requests.post, f"{CHATBOT_API_URL}/chat", json=payload, timeout=180
    )
    live_note: str | None = None
    while not turn_future.done():
        time.sleep(_TURN_STATUS_POLL_SECONDS)
        if live_note:
            continue
        live_note = _fetch_turn_status(payload["session_id"])
        if live_note:
            _show_waiting_bubble(slot, live_note)
    response = turn_future.result()
    response.raise_for_status()
    return response.json()


class _StreamingUnavailable(Exception):
    """The backend does not serve /chat/stream. Run the turn the blocking way instead."""


def _run_streaming_turn(payload: dict, slot) -> dict:
    """POST /chat/stream and type the reply out as it arrives, one bubble per chunk.

    The rendering happens HERE rather than after the call, which is the whole point: the
    customer watches the intro land, then each trailer, then the closing question, instead of
    staring at dots and then a wall of text. Every bubble drawn here is thrown away by the
    st.rerun() that follows the turn and redrawn from history by _render_message_bubbles -
    same splitter, same bubbles - so none of it needs storing.

    Raises the same exceptions the blocking path does, so the caller handles failure once.
    """
    note: str | None = None
    typing = None
    text = ""
    data: dict | None = None
    started = False
    with requests.post(
        f"{CHATBOT_API_URL}/chat/stream",
        json=payload,
        stream=True,
        timeout=180,
        headers={"Accept": "text/event-stream"},
    ) as response:
        if response.status_code in (404, 405, 501):
            # Streaming turned off on the backend, or an older API behind this UI. Nothing
            # has been rendered yet, so the caller can still run the turn the blocking way.
            raise _StreamingUnavailable(f"/chat/stream returned {response.status_code}")
        response.raise_for_status()
        for event, body in _iter_sse_events(response):
            if event == "status":
                note = (body.get("search_status_message") or "").strip() or note
                if note and not started:
                    _show_waiting_bubble(slot, note)
            elif event == "chunk_start":
                if not started:
                    # The dots have said all they can; the note moves into the reply itself.
                    slot.empty()
                    started = True
                text = ""
                with st.chat_message("assistant"):
                    if not body.get("index"):
                        _render_search_note(note)
                    typing = st.empty()
            elif event == "delta":
                if typing is None:
                    continue
                text += str(body.get("text") or "")
                typing.markdown(text + _TYPING_CARET)
            elif event == "chunk_end":
                if typing is None:
                    continue
                # The caret goes, and the chunk is re-rendered whole so any markdown that was
                # still half-typed (a listing hyperlink, mid-URL) resolves.
                typing.markdown(str(body.get("text") or text))
                typing = None
            elif event == "error":
                raise requests.RequestException(str(body.get("detail") or "the streamed turn failed"))
            elif event == "done":
                data = body
    if data is None:
        raise ValueError("the reply stream ended without a result")
    return data


def _process_assistant_reply(prompt: str) -> bool:
    """Call the chat API, append the assistant turn, and run optional thinking."""
    listings = []
    thinking_context = None
    search_note = None
    payload = {
        "session_id": st.session_state.chat_session_id,
        "turn_id": st.session_state.pending_turn_id,
        "sales_phase": st.session_state.sales_phase,
        "message": prompt,
        "onboarding_api_messages": st.session_state.onboarding_api_messages,
        "customer_full_name": st.session_state.get("customer_full_name"),
        "customer_email": st.session_state.get("customer_email"),
        "customer_phone": st.session_state.get("customer_phone"),
        "already_shown_listing_urls": (
            accumulate_shown_urls_from_chat_messages(st.session_state.messages)
            if st.session_state.sales_phase == "main"
            else []
        ),
    }
    # The whole waiting state lives in this slot: dots immediately, then the search line above
    # them once the backend publishes it, then gone the moment the reply starts arriving.
    waiting_slot = st.empty()
    _show_waiting_bubble(waiting_slot, None)
    streamed = False
    try:
        try:
            if not CHAT_STREAMING:
                raise _StreamingUnavailable("streaming disabled in this UI")
            data = _run_streaming_turn(payload, waiting_slot)
            streamed = True
        except _StreamingUnavailable:
            data = _run_blocking_turn(payload, waiting_slot)
    except (requests.RequestException, ValueError) as exc:
        # Stop the dots, or they keep bouncing under the error message.
        waiting_slot.empty()
        st.error(f"The assistant service is unavailable ({exc!s}). Retry when it is ready.")
        return False

    response_text = (data.get("assistant_text") or "").strip() or " "
    # Set by search_node only when the inventory search actually fired.
    search_note = (data.get("search_status_message") or "").strip() or None
    st.session_state.onboarding_api_messages = data.get(
        "onboarding_api_messages"
    ) or st.session_state.onboarding_api_messages
    sp = data.get("sales_phase")
    if sp in ("onboarding", "main"):
        st.session_state.sales_phase = sp
    if data.get("customer_full_name"):
        st.session_state.customer_full_name = data["customer_full_name"]
    if "customer_email" in data:
        st.session_state.customer_email = data.get("customer_email") or ""
    if data.get("customer_phone"):
        st.session_state.customer_phone = data["customer_phone"]
    if data.get("main_prior_messages") is not None:
        st.session_state.main_prior_messages = data["main_prior_messages"]
    thinking_context = data.get("thinking_context")
    st.session_state.pending_turn_id = None

    listings = []
    for d in (data.get("listings") or []):
        if not isinstance(d, dict):
            continue
        try:
            listings.append(
                TrailerListing(
                    listing_id=str(d.get("url") or d.get("title") or ""),
                    title=str(d.get("title") or ""),
                    condition=str(d.get("condition") or "New"),
                    price=float(
                        d["price"].replace("$", "").replace(",", "")
                    )
                    if isinstance(d.get("price"), str)
                    and d.get("price")
                    not in ("Call for price", None, "")
                    else d.get("price"),
                    price_display=str(d.get("price") or "") or None,
                    payments_from=None,
                    category_subcategory=str(d.get("category") or ""),
                    make=str(d.get("make") or ""),
                    color=str(d.get("color") or ""),
                    hitch_type=d.get("hitch_type"),
                    year=d.get("year"),
                    length=d.get("length"),
                    width=d.get("width"),
                    axles=d.get("axles"),
                    gvwr=d.get("gvwr"),
                    payload_capacity=d.get("payload_capacity"),
                    trailer_material=d.get("material"),
                    floor=d.get("floor"),
                    url=str(d.get("url") or ""),
                    score=d.get("relevance_score"),
                )
            )
        except Exception:
            pass

    message = {
        "role": "assistant",
        "content": response_text,
        "search_note": search_note,
        "listings": listings or None,
        "user_feedback": None,
        "thinking_payload": (
            thinking_context
            if thinking_agent_enabled() and thinking_context is not None
            else None
        ),
        "thinking_result": None,
    }
    st.session_state.messages.append(message)
    assistant_message_index = len(st.session_state.messages) - 1
    # Dots (and the live copy of the note) go; the note is re-rendered below as part of the
    # finished message so it stays in the transcript instead of vanishing with them.
    waiting_slot.empty()
    if not streamed:
        # Streaming already drew the reply, bubble by bubble, as it came in.
        _render_message_bubbles(message)

    if thinking_agent_enabled() and thinking_context is not None:
        st.session_state.last_thinking_payload = thinking_context
        if thinking_agent_background():
            st.session_state.thinking_status = "pending"
            st.session_state.thinking_future = _THINKING_EXECUTOR.submit(
                _run_thinking_job,
                st.session_state.chat_session_id,
                thinking_context,
            )
        else:
            _sync_thinking_result = _run_thinking_job(
                st.session_state.chat_session_id, thinking_context
            )
            st.session_state.last_thinking_result = _sync_thinking_result
            st.session_state.thinking_status = (
                "done" if _sync_thinking_result.get("status") == "ok" else "error"
            )
            st.session_state.thinking_future = None
            st.session_state.messages[assistant_message_index][
                "thinking_result"
            ] = _sync_thinking_result

    sid = st.session_state.get("chat_session_id")
    if sid and listings:
        add_shown_urls(
            sid,
            [str(x.url or "") for x in listings if getattr(x, "url", None)],
        )
    return st.session_state.pending_turn_id is None


placeholder = (
    "Type a message…" if st.session_state.messages else "What kind of trailer are you looking for?"
)

backend_status = st.session_state.get("backend_ready_status", "initializing")
backend_ready = backend_status == "ready"
awaiting_response = st.session_state.get("chat_awaiting_response", False)
chat_input_disabled = not backend_ready or awaiting_response

if backend_status == "initializing":
    st.markdown(
        '<div class="tp-init-status"><span class="tp-init-spinner"></span>'
        '<span>Chatbot initializing…</span></div>',
        unsafe_allow_html=True,
    )
    placeholder = "Chatbot initializing…"
elif backend_status == "error":
    st.error("The chatbot is taking longer than expected to initialize.")
    if st.button("Retry initialization", type="primary"):
        _start_backend_initialization()
        st.rerun()
    placeholder = "Chatbot unavailable"
elif awaiting_response:
    placeholder = "Waiting for a response…"

_pending_user_turn = (
    awaiting_response
    and st.session_state.messages
    and st.session_state.messages[-1]["role"] == "user"
)

if _pending_user_turn:
    st.chat_input(placeholder, disabled=True)
    if _process_assistant_reply(st.session_state.messages[-1]["content"]):
        st.session_state.chat_awaiting_response = False
        st.rerun()
elif prompt := st.chat_input(placeholder, disabled=chat_input_disabled):
    st.session_state.messages.append({"role": "user", "content": prompt, "listings": None})
    st.session_state.pending_turn_id = str(uuid.uuid4())
    st.session_state.chat_awaiting_response = True
    st.rerun()

if backend_status == "initializing":
    time.sleep(1)
    st.rerun()
