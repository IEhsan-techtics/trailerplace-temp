"""Deterministic, automatically graded long-context reasoning cases."""

from __future__ import annotations

import random


KINDS = (
    "revisions", "ledger", "routing", "inventory", "scheduling",
    "policy", "dependencies", "quorum", "pricing", "timeline",
)
TARGET_TOKENS = 15_000


def build_puzzle(kind: str, rng: random.Random) -> tuple[str, list[str], dict[str, str]]:
    rows: list[str] = []
    expected: dict[str, str] = {}
    ids = [f"CASE-{i}" for i in range(1, 7)]

    if kind == "revisions":
        rule = ("For each case, among APPROVED revisions effective on or before day 30, "
                "choose the highest revision number. Ignore drafts and withdrawn revisions. "
                "Return its ZONE code.")
        for i, cid in enumerate(ids, 1):
            old, new = f"ZONE-{10+i}", f"ZONE-{40+i}"
            rows += [f"{cid} | revision 1 | approved | day 10 | {old}",
                     f"{cid} | revision 2 | approved | day {20+i} | {new}",
                     f"{cid} | revision 3 | withdrawn | day 25 | ZONE-99",
                     f"{cid} | revision 4 | approved | day 35 | ZONE-98"]
            expected[cid] = new
    elif kind == "ledger":
        rule = ("For each case, sum the signed USD amounts of POSTED entries only. "
                "A VOID entry contributes zero. Return the net as an integer string, "
                "including a minus sign when negative.")
        for i, cid in enumerate(ids, 1):
            amounts = [120+i*7, -(65+i*9), 31+i*3, -(110+i*4)]
            statuses = ["POSTED", "POSTED", "VOID", "POSTED"]
            for j, (amount, status) in enumerate(zip(amounts, statuses), 1):
                rows.append(f"{cid} | transaction {j} | {status} | USD {amount:+d}")
            expected[cid] = str(sum(a for a, s in zip(amounts, statuses) if s == "POSTED"))
    elif kind == "routing":
        rule = ("For each case, find the cheapest directed route from A to D. "
                "CLOSED edges cannot be used. Add edge costs; return the minimum "
                "total cost as an integer string. All costs are nonnegative.")
        for i, cid in enumerate(ids, 1):
            ab, bd, ac, cd, direct = 3+i, 12-i, 7+i, 5+i, 18-i
            rows += [f"{cid} | A -> B | OPEN | cost {ab}",
                     f"{cid} | B -> D | OPEN | cost {bd}",
                     f"{cid} | A -> C | OPEN | cost {ac}",
                     f"{cid} | C -> D | OPEN | cost {cd}",
                     f"{cid} | A -> D | CLOSED | cost 1",
                     f"{cid} | B -> C | OPEN | cost 2"]
            expected[cid] = str(min(ab+bd, ac+cd, ab+2+cd))
    elif kind == "inventory":
        rule = ("For each case, compute available units = OPENING + RECEIPT "
                "+ RETURN - SALE - DAMAGED. Return the integer count. "
                "Each row is an independent event; include every listed event.")
        for i, cid in enumerate(ids, 1):
            events = [("OPENING", 32+i), ("RECEIPT", 15+2*i),
                      ("SALE", 21+i), ("RETURN", i), ("DAMAGED", 3+i)]
            rows += [f"{cid} | {event} | units {number}" for event, number in events]
            expected[cid] = str(events[0][1]+events[1][1]-events[2][1]+events[3][1]-events[4][1])
    elif kind == "scheduling":
        rule = ("Count ACTIVE bookings for each case that overlap time 13:00. "
                "Intervals include their start and exclude their end. CANCELLED "
                "bookings do not count. Return the integer count.")
        for i, cid in enumerate(ids, 1):
            bookings = [("ACTIVE", 9, 13), ("ACTIVE", 12, 15),
                        ("ACTIVE", 13, 14), ("CANCELLED", 11, 16),
                        ("ACTIVE", 10+i%3, 13+i%2)]
            for j, (status, start, end) in enumerate(bookings, 1):
                rows.append(f"{cid} | booking {j} | {status} | {start:02d}:00 to {end:02d}:00")
            expected[cid] = str(sum(s == "ACTIVE" and a <= 13 < b for s, a, b in bookings))
    elif kind == "policy":
        rule = ("Choose the applicable policy with the highest numeric priority "
                "for each case. GLOBAL applies to all; a case-specific rule only "
                "applies to its named case. Ignore DISABLED rules. Return its ACTION code.")
        rows.append("GLOBAL | priority 5 | ENABLED | ACTION-A")
        for i, cid in enumerate(ids, 1):
            rows += [f"{cid} | priority {7+i} | ENABLED | ACTION-{i}",
                     f"{cid} | priority 99 | DISABLED | ACTION-X",
                     f"{cid} | priority 6 | ENABLED | ACTION-Y"]
            expected[cid] = f"ACTION-{i}"
    elif kind == "dependencies":
        rule = ("In each case, arrows mean the left task must finish before the "
                "right task. If task A is delayed, count all DISTINCT tasks reachable "
                "from A by one or more arrows. Do not count A itself. Return the count.")
        for i, cid in enumerate(ids, 1):
            edges = [("A", "B"), ("A", "C"), ("B", "D"), ("C", "D"),
                     ("D", "E"), ("E", "F")]
            if i % 2 == 0:
                edges.append(("F", "G"))
            if i % 3 == 0:
                edges.append(("C", "H"))
            rows += [f"{cid} | dependency {a} -> {b}" for a, b in edges]
            expected[cid] = str(5 + (i % 2 == 0) + (i % 3 == 0))
    elif kind == "quorum":
        rule = ("For each case, sum weights of YES and NO votes separately; ABSTAIN "
                "adds to neither. A motion PASSES if YES is at least 12 AND YES is "
                "strictly greater than NO. Return PASS or FAIL.")
        for i, cid in enumerate(ids, 1):
            votes = [("YES", 4+i), ("YES", 3+i%3),
                     ("NO", 5+i%4), ("ABSTAIN", 8),
                     ("YES" if i % 2 else "NO", 2)]
            rows += [f"{cid} | voter {j} | {vote} | weight {weight}"
                     for j, (vote, weight) in enumerate(votes, 1)]
            yes = sum(w for v, w in votes if v == "YES")
            no = sum(w for v, w in votes if v == "NO")
            expected[cid] = "PASS" if yes >= 12 and yes > no else "FAIL"
    elif kind == "pricing":
        rule = ("For each case, compute total CENTS = quantity * unit_price_cents "
                "minus discount_cents plus shipping_cents. Discount is applied "
                "once per order, not per unit. Return integer cents as a string.")
        for i, cid in enumerate(ids, 1):
            qty, price, discount, shipping = 2+i, 345+17*i, 31*i, 99+i
            rows += [f"{cid} | quantity {qty}", f"{cid} | unit_price_cents {price}",
                     f"{cid} | discount_cents {discount}", f"{cid} | shipping_cents {shipping}"]
            expected[cid] = str(qty*price-discount+shipping)
    elif kind == "timeline":
        rule = ("For each case, add ACTIVE minutes from OPEN to PAUSE and from "
                "RESUME to CLOSE. Time spent paused is excluded. Times are minutes "
                "after midnight on the same day. Return integer active minutes.")
        for i, cid in enumerate(ids, 1):
            events = [("OPEN", 100+i), ("PAUSE", 122+2*i),
                      ("RESUME", 140+3*i), ("CLOSE", 190+4*i)]
            rows += [f"{cid} | {event} | minute {minute}" for event, minute in events]
            expected[cid] = str(events[1][1]-events[0][1]+events[3][1]-events[2][1])
    else:
        raise ValueError(kind)
    return rule, rows, expected


def make_case(kind: str) -> tuple[str, dict[str, str]]:
    if kind not in KINDS:
        raise ValueError(kind)
    rng = random.Random(202600 + KINDS.index(kind) * 977)
    rule, rows, expected = build_puzzle(kind, rng)
    intro = (
        f"SCENARIO: {kind.upper()}\n{rule}\n"
        "Only rows beginning CASE-1 through CASE-6 (and GLOBAL, when relevant) "
        "belong to the requested cases. Other IDs are independent distractors. "
        "Return ONLY a JSON object with exactly CASE-1 through CASE-6 as keys, "
        "and string values. No explanation or markdown.\n\nRECORDS:\n"
    )
    filler = []
    for j in range(1, 251):
        filler.append(
            f"ARCHIVE-{j:04d} | unrelated {kind} file | observation {rng.randrange(1000,9999)} | "
            "this record belongs to a different case and has no bearing on any "
            "requested CASE identifier; keep each case's facts separate."
        )
    # Spread relevant rows through the full context rather than placing them together.
    rng.shuffle(rows)
    positions = [int((i + 0.5) * len(filler) / len(rows)) for i in range(len(rows))]
    for offset, (position, row) in enumerate(zip(positions, rows)):
        filler.insert(position + offset, row)
    try:
        import tiktoken
        encoding = tiktoken.get_encoding("cl100k_base")
        count = lambda s: len(encoding.encode(s))
    except Exception:
        count = lambda s: int(len(s.split()) * 1.5)
    suffix = "\n\nReturn the six requested answers as JSON only."
    while count(intro + "\n".join(filler) + suffix) < TARGET_TOKENS:
        j = len(filler) + 1000
        filler.append(
            f"ARCHIVE-{j:04d} | independent historical {kind} note | checksum {rng.randrange(10000,99999)} | "
            "the archive is unrelated to CASE-1 through CASE-6 and must not alter "
            "their calculations, rules, graph, dates, or answer values."
        )
    return intro + "\n".join(filler) + suffix, expected
