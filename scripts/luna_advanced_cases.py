"""Ten additional, deterministic long-context extraction and inference cases."""

from __future__ import annotations

import random


KINDS = (
    "supplier_awards", "shipment_chain", "revision_conflicts",
    "budget_reconcile", "sla_escalation", "inventory_recall",
    "policy_exceptions", "entity_aliases", "project_schedule", "mcq_50",
)
TARGET_TOKENS = 15_000


def build(kind: str, rng: random.Random):
    rows = []
    expected = {}
    ids = [f"CASE-{i}" for i in range(1, 7)]

    if kind == "supplier_awards":
        rule = ("Each case has three bids. A bid qualifies only if certification is VALID "
                "and risk is at most 3. Choose the qualified bid with highest score; "
                "break score ties by lower price. Return the supplier ID.")
        for i, cid in enumerate(ids, 1):
            bids = [(f"SUP-{i}-A", 72+i, 120-i, "VALID", 2),
                    (f"SUP-{i}-B", 91+i, 90+i, "EXPIRED", 1),
                    (f"SUP-{i}-C", 74+i%3, 100+i, "VALID", 3 if i%2 else 4)]
            for supplier, score, price, cert, risk in bids:
                rows += [f"{cid} | bid {supplier} | score {score} | price {price}",
                         f"{cid} | supplier {supplier} | certification {cert} | risk {risk}"]
            qualified = [b for b in bids if b[3] == "VALID" and b[4] <= 3]
            expected[cid] = max(qualified, key=lambda b: (b[1], -b[2]))[0]
    elif kind == "shipment_chain":
        rule = ("Follow each CASE order to its CURRENT lot, that lot to its assigned "
                "warehouse, and that warehouse to its zone. Ignore SUPERSEDED order "
                "links. Return the final zone code for each case.")
        for i, cid in enumerate(ids, 1):
            lot, warehouse = f"LOT-{i:02d}", f"WH-{i:02d}"
            rows += [f"{cid} | CURRENT lot {lot}", f"{cid} | SUPERSEDED lot LOT-X{i}",
                     f"{lot} | assigned warehouse {warehouse}",
                     f"{warehouse} | zone ZONE-{20+i}",
                     f"LOT-X{i} | assigned warehouse WH-X{i}",
                     f"WH-X{i} | zone ZONE-99"]
            expected[cid] = f"ZONE-{20+i}"
    elif kind == "revision_conflicts":
        rule = ("For each case, choose the signed memo with the latest EFFECTIVE day "
                "at or before day 30. Ignore unsigned memos and later effective dates. "
                "Return the chosen ACTION code.")
        for i, cid in enumerate(ids, 1):
            rows += [f"{cid} | memo M1 | SIGNED | effective day 12 | ACTION-A{i}",
                     f"{cid} | memo M2 | SIGNED | effective day {23+i} | ACTION-B{i}",
                     f"{cid} | memo M3 | UNSIGNED | effective day 29 | ACTION-X{i}",
                     f"{cid} | memo M4 | SIGNED | effective day 35 | ACTION-Y{i}"]
            expected[cid] = f"ACTION-B{i}"
    elif kind == "budget_reconcile":
        rule = ("For each case, start with BASE dollars, add all POSTED signed changes, "
                "ignore VOID changes, then return OVER if the resulting amount exceeds "
                "the LIMIT strictly; otherwise return WITHIN.")
        for i, cid in enumerate(ids, 1):
            base, limit = 90+5*i, 110+3*i
            changes = [("POSTED", 20+i), ("POSTED", -(6+i)), ("VOID", 90),
                       ("POSTED", 4 if i%2 else -4)]
            rows += [f"{cid} | BASE {base} | LIMIT {limit}"]
            rows += [f"{cid} | change {j} | {s} | dollars {v:+d}"
                     for j, (s, v) in enumerate(changes, 1)]
            final = base + sum(v for s, v in changes if s == "POSTED")
            expected[cid] = "OVER" if final > limit else "WITHIN"
    elif kind == "sla_escalation":
        rule = ("Compute active handling minutes = CLOSE minus OPEN minus the PAUSED "
                "interval (RESUME minus PAUSE). Return BREACH if active minutes are "
                "strictly greater than the case SLA limit, else ON_TIME.")
        for i, cid in enumerate(ids, 1):
            opened, paused, resumed, closed = 100+i, 119+2*i, 132+3*i, 175+5*i
            limit = 62+3*i+(2 if i%2 == 0 else -2)
            rows += [f"{cid} | OPEN minute {opened} | SLA limit {limit}",
                     f"{cid} | PAUSE minute {paused}",
                     f"{cid} | RESUME minute {resumed}",
                     f"{cid} | CLOSE minute {closed}"]
            active = closed-opened-(resumed-paused)
            expected[cid] = "BREACH" if active > limit else "ON_TIME"
    elif kind == "inventory_recall":
        rule = ("For each case, find its product lot. Count units at warehouses whose "
                "warehouse status is OPEN, but only when that lot's recall status is "
                "ACTIVE. If recall is CLEARED return 0. Return integer units as a string.")
        for i, cid in enumerate(ids, 1):
            lot = f"LOT-{i:02d}"
            status = "ACTIVE" if i%3 else "CLEARED"
            rows += [f"{cid} | product lot {lot}", f"{lot} | recall {status}",
                     f"{lot} | warehouse W-A{i} | units {10+i}",
                     f"{lot} | warehouse W-B{i} | units {7+2*i}",
                     f"W-A{i} | warehouse status OPEN",
                     f"W-B{i} | warehouse status {'CLOSED' if i%2 else 'OPEN'}"]
            expected[cid] = str((10+i)+(7+2*i if i%2 == 0 else 0)) if status == "ACTIVE" else "0"
    elif kind == "policy_exceptions":
        rule = ("Determine each case's applicable ACTION. A case-specific ACTIVE "
                "exception overrides its region's ACTIVE policy; otherwise use the "
                "region policy. Ignore REVOKED exceptions. Return the ACTION code.")
        for i, cid in enumerate(ids, 1):
            region = f"REG-{i}"
            exception_status = "ACTIVE" if i%2 else "REVOKED"
            rows += [f"{cid} | region {region}",
                     f"{region} | ACTIVE policy ACTION-R{i}",
                     f"{cid} | {exception_status} exception ACTION-E{i}",
                     f"{cid} | REVOKED exception ACTION-X{i}"]
            expected[cid] = f"ACTION-E{i}" if i%2 else f"ACTION-R{i}"
    elif kind == "entity_aliases":
        rule = ("Each case names an alias. Resolve the alias to a canonical entity, "
                "then sum that entity's POSTED signed transactions. VOID transactions "
                "count zero. Return the integer total as a string.")
        for i, cid in enumerate(ids, 1):
            alias, canonical = f"ALIAS-{i}", f"ENTITY-{i}"
            values = [20+3*i, -7-i, 100, 4+i]
            statuses = ["POSTED", "POSTED", "VOID", "POSTED"]
            rows += [f"{cid} | alias {alias}", f"{alias} | canonical {canonical}"]
            rows += [f"{canonical} | tx {j} | {s} | amount {v:+d}"
                     for j, (s, v) in enumerate(zip(statuses, values), 1)]
            expected[cid] = str(sum(v for s, v in zip(statuses, values) if s == "POSTED"))
    elif kind == "project_schedule":
        rule = ("Each case is a small project. Task A starts at day 0. Each other task "
                "starts when all its prerequisites finish. Finish = start + duration. "
                "Return the earliest finish day of task D as an integer string.")
        for i, cid in enumerate(ids, 1):
            a, b, c, d = 2+i, 3+i%3, 4+i%2, 1+i
            rows += [f"{cid} | task A | duration {a} | prerequisites NONE",
                     f"{cid} | task B | duration {b} | prerequisites A",
                     f"{cid} | task C | duration {c} | prerequisites A",
                     f"{cid} | task D | duration {d} | prerequisites B,C"]
            expected[cid] = str(a+max(b,c)+d)
    elif kind == "mcq_50":
        rule = ("Answer all 50 multiple-choice questions in one shot. For each item, "
                "use its dossier facts: ELIGIBLE means score at least 70, status ACTIVE, "
                "and hold NO. Each item states a claim that the entity IS ELIGIBLE or "
                "IS NOT ELIGIBLE. Choose A=True if the claim is correct, B=False "
                "otherwise. Return ONLY a JSON array containing the IDs of items "
                "whose correct choice is A=True. Do not include false item IDs.")
        expected = []
        true_claim_items = set(rng.sample(range(1, 51), 25))
        for i in range(1, 51):
            qid, entity = f"MCQ-{i:03d}", f"ENTITY-{i:03d}"
            score = 62 + (i*7)%23
            status = "ACTIVE" if i%4 else "SUSPENDED"
            hold = "YES" if i%7 == 0 else "NO"
            eligible = score >= 70 and status == "ACTIVE" and hold == "NO"
            claim_is_eligible = eligible if i in true_claim_items else not eligible
            rows += [f"{entity} | score {score}",
                     f"{entity} | status {status}",
                     f"{entity} | hold {hold}",
                     f"{qid} | entity {entity} | claim: entity IS {'ELIGIBLE' if claim_is_eligible else 'NOT ELIGIBLE'} | A=True B=False"]
            if claim_is_eligible == eligible:
                expected.append(qid)
    else:
        raise ValueError(kind)
    return rule, rows, expected


def make_case(kind: str):
    if kind not in KINDS:
        raise ValueError(kind)
    rng = random.Random(87000 + KINDS.index(kind)*139)
    rule, facts, expected = build(kind, rng)
    intro = f"SCENARIO: {kind.upper()}\n{rule}\n\nDOSSIER (records are deliberately out of order):\n"
    if kind != "mcq_50":
        intro += ("Answer CASE-1 through CASE-6 only. Return ONLY a JSON object with "
                  "those six keys and string values. No explanations or markdown.\n")
    filler = [
        f"ARCHIVE-{j:04d} | unrelated historical {kind} record | reference {rng.randrange(10000,99999)} | "
        "independent file; it does not change the facts, rules, entities, or answers "
        "for the requested IDs."
        for j in range(1, 201)
    ]
    rng.shuffle(facts)
    for index, fact in enumerate(facts):
        position = int((index+0.5)*len(filler)/len(facts))
        filler.insert(position+index, fact)
    try:
        import tiktoken
        encoding = tiktoken.get_encoding("cl100k_base")
        count = lambda text: len(encoding.encode(text))
    except Exception:
        count = lambda text: int(len(text.split())*1.5)
    suffix = "\nReturn only the requested JSON answer."
    while count(intro+"\n".join(filler)+suffix) < TARGET_TOKENS:
        j = len(filler)+1000
        filler.append(
            f"ARCHIVE-{j:04d} | archived {kind} correspondence | reference {rng.randrange(10000,99999)} | "
            "separate historical record; no relation to a requested case or MCQ item."
        )
    return intro+"\n".join(filler)+suffix, expected
