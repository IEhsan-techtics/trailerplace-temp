# TrailerPlace — company info & email scenarios

Reference for the two things the assistant states as fact about the business, and every
case where it notifies the team by email.

---

## Company info

| | |
|---|---|
| **Name** | TrailerPlace |
| **Location** | Wharton, TX |
| **Phone** | 979-532-1486 |
| **Website** | https://trailerplace.com (`TRAILERPLACE_WEBSITE` is set to `https://www.trailerplace.com`) |
| **Services** | Trailer sales, financing, delivery, trade-in appraisals, service and parts |

Financing, trade-ins, service and parts are all handled by **people, not the bot** — every
one of those questions is answered with the phone number and a notification to the team.

### Inventory scope

14 canonical categories: Aluminum, Car Hauler, Equipment, Enclosed, Utility, Fiber, Race
Trailer, Roll Off, Diesel Tank, Flatbed, Dump, Tilt, Livestock, Concession.

What is actually *advertised* is never this list — it is whatever `trailer_listings`
currently holds, computed by `brands.stocked_categories()`. At the last check that was 13
of the 14 (Diesel Tank unstocked) across 20 makes. A category sells out and it drops off
the list on its own; nothing needs editing.

### The rule about stating facts

The bot only ever claims inventory the search can actually return. Brands and categories
are read from the same table the search reads, specifically so the assistant cannot
promise a make the index has never seen.

---

## Email scenarios

Every email is an **internal notification to the dealership** — none of it goes to the
customer. The customer gets a canned line in the reply telling them their request was
passed on.

Two families: **customer-triggered** (someone asked for something) and **system**
(something happened worth logging). Only customer-triggered ones ever hold up the
conversation to ask for contact details; system alerts wait silently.

### Customer-triggered

| Scenario | Fires when | Reason line | Outbox event | What the customer is told |
|---|---|---|---|---|
| **FAQ** | They ask something non-sales that we answer from a fixed script | `FAQ – {faq_key}` | `non_sales_faq` | The matching canned answer below |
| **Escalation** | They report a problem, or ask something we cannot answer (restock dates, when new stock lands) | `Escalation` | `escalation_alert` | "I'm sorry to hear that. I've noted it and passed it to our team — they'll reach out to you, and you can also reach them directly on 979-532-1486." |
| **Team request** | They ask for a human action — a call back, a quote, a meeting | `Team Request` | `team_request` | "Thanks, I shared that request with the team so they can help you with it. If you'd rather not wait, our sales team is on 979-532-1486." |
| **Listing interest** | They express interest in a trailer that was shown | `Listing Interest` | `interested_listing` | One of the three variants below |

#### The five FAQ answers

| `faq_key` | Canned response |
|---|---|
| `contact_human` | "You can reach our team at 979-532-1486. Happy to keep helping with your trailer search too!" |
| `financing` | "We offer financing. Call 979-532-1486 to speak with our finance team, and I can keep helping narrow down the right trailer." |
| `trade_in` | "Our sales team handles trade-in appraisals. Call 979-532-1486." |
| `service_parts` | "Our service and parts team can help. Reach them at 979-532-1486." |
| `store_info` | "We're located in Wharton, TX. Call 979-532-1486 or visit https://trailerplace.com. We also offer financing and delivery." |

Every one of these names the phone number, deliberately. They all fire on a turn we could
not settle ourselves, so the number is the one thing the customer must not be left without.

#### The three listing-interest variants

Which one is used depends on whether we can pin down *which* trailer they meant:

| Variant | When | Response |
|---|---|---|
| `listing_interest_selected` | A specific shown trailer is identified — by position, stock number, or an unambiguous make | "Your interest in the selected trailer has been logged. Our team can follow up. In the meantime, feel free to visit https://trailerplace.com or call 979-532-1486." |
| `listing_interest_unselected` | Listings are on screen but we cannot tell which one | "Your interest has been logged. Our team can follow up. In the meantime, feel free to visit https://trailerplace.com or call 979-532-1486." |
| `listing_interest_fallback` | No listings on screen at all | "Great, I shared your interest in that trailer with the team. They can follow up with you, or you can reach them on 979-532-1486." |

Identification is resolved against the batch **currently on screen**, not the cumulative
history — otherwise "the 5th one" after a "show me more" points at a trailer they already
scrolled past. A make match only counts when it is unambiguous: with two Iron Bulls
showing, "the Iron Bull one" identifies nothing, and logging no trailer beats logging the
wrong one.

### System-triggered (silent — no canned text, never asks for contact)

| Scenario | Fires when | Reason line | Outbox event |
|---|---|---|---|
| **Results shown** | A search or lookup actually returned trailers | `Results Shown to User` | `results_shown` |
| **Unanswered question** | A qualification question was skipped without an answer | `Unanswered Question` | `unanswered_question` |

---

## The contact gate

**Minimum to send: name + (email OR phone).**

- Complete → the whole batch goes out, stashed items included.
- Incomplete → the request is **stashed, not dropped**, and the reply asks for every
  missing piece at once rather than one question per piece.
- Declined → the batch is dropped silently and the conversation carries on normally. We
  do not keep asking.

One exception worth knowing: a customer who later *asks us to act* — a listing interest, a
FAQ, a callback — has reopened the question of how we reach them, so we ask once more even
if they brushed off the opening invite. Their original decline still permanently closes
the *opening* invite.

The customer always gets their canned answer immediately, gate or not. Only the email
waits.

### Deduplication

A trigger stashed while waiting for contact details, plus the extractor re-emitting that
same trigger on the turn the details arrive, is **one** request. Collapsed on
`(kind, canned_key, item_of_interest)` — deliberately not on the description, because the
extractor re-words the request each time and keying on wording sent the team the same
Listing Interest email twice.

---

## Email format

Subject:

    TrailerPlace Lead — {Reason} — {customer name or session id}

Body:

    Full Name: {name or "Not provided"}
    Email: {email or "Not provided"}
    Phone Number: {phone or "Not provided"}

    [{Reason}] {one-line description}

Reason vocabulary is fixed: `Escalation`, `Team Request`, `FAQ – {faq_key}`,
`Listing Interest`, `Results Shown to User`, `Unanswered Question`.

Delivery is Microsoft Graph (`EMAIL_BACKEND=graph`) with SMTP as the alternative; the
recipient comes from `RECIPIENT_EMAIL` / `EMAIL_TO` in `.env`. With persistence on, mail
is queued as a `chatbot_outbox` row inside the turn's transaction and drained separately;
with it off, it sends inline. Sending never raises — a failed email cannot take the bot
down.

---

## Note on this file

The company facts and canned text above are transcribed from the **New Prompt** codebase
(`src/domain/canned_responses.py`, `src/graph/nodes/email_actions.py`,
`src/tools/email_sender.py`). None of that code is in this repo — 5.6 Luna carries the
domain model and retrieval path only. This is reference material, not a description of
anything running here.
