"""Builders for the structured output the model would have returned.

A test says only what matters to it; everything else takes the "the model said nothing"
value, exactly as ``client.empty_output`` does.
"""
from __future__ import annotations

from typing import Any

from src.llm.client import empty_output
from src.llm.schemas import SlotAnswer


def turn_output(**kwargs: Any):
    """A ChatbotTurnOutput with overrides.

    ``slots={"length": "20 ft"}`` is shorthand for the slot_answers list, since that is the
    field almost every qualification test needs.
    """
    output = empty_output()
    slots = kwargs.pop("slots", None)
    extracted = kwargs.pop("extracted", None)

    if slots:
        output.slot_answers = [
            SlotAnswer(slot_name=name, raw_answer=str(raw)) for name, raw in slots.items()
        ]
        if "intent" not in kwargs:
            kwargs["intent"] = "qualification_answer"
        if "answered_current_question" not in kwargs:
            kwargs["answered_current_question"] = True

    if extracted:
        for field, value in extracted.items():
            setattr(output.extracted, field, value)

    for field, value in kwargs.items():
        if field in {"name", "email", "phone", "declined"}:
            setattr(output.contact, field, value)
        else:
            setattr(output, field, value)
    return output


def raw_span(slot_name: str, raw_answer: str) -> SlotAnswer:
    return SlotAnswer(slot_name=slot_name, raw_answer=raw_answer)


def complete_welcome(fake_llm, session_id: str = "s1") -> None:
    """Get past the welcome turn.

    The first reply of every conversation is the greeting and the contact request, and no
    listings or qualification questions go out until contact is settled. A test about the
    qualification flow therefore has to start on turn two, exactly as a real customer does.
    """
    from src.graph.build import run_turn

    fake_llm.push(
        turn_output(intent="contact_info_provided", name="Dave", phone="979-555-0100")
    )
    run_turn(session_id, "hi, I'm Dave on 979-555-0100")
