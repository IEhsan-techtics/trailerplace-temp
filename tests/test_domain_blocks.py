"""The prompt blocks: they must describe real categories and never advertise absent stock."""
from __future__ import annotations

import dataclasses

from src.domain import company
from src.domain.categories import (
    CANONICAL_CATEGORIES,
    CATEGORY_BLURBS,
    category_menu_block,
)


def test_every_canonical_category_has_a_use_case_blurb():
    """Brief S5: categories are presented with what they are for, not as a name dump."""
    missing = [c for c in CANONICAL_CATEGORIES if c not in CATEGORY_BLURBS]
    assert missing == []


def test_no_blurb_describes_a_category_that_does_not_exist():
    """Brief S4/S39: nothing invented."""
    unknown = [c for c in CATEGORY_BLURBS if c not in CANONICAL_CATEGORIES]
    assert unknown == []


def test_the_menu_only_lists_categories_we_stock(monkeypatch):
    from src.domain import categories as categories_module

    monkeypatch.setattr(categories_module, "_advertised_categories", lambda: ("Dump", "Utility"))
    block = category_menu_block()
    assert "Dump" in block and "Utility" in block
    assert "Concession" not in block


def test_each_menu_line_carries_its_use_case():
    for line in category_menu_block().splitlines():
        assert ":" in line, f"category listed without a use case: {line!r}"


# --------------------------------------------------------------------------- company facts
def test_company_block_states_the_real_dealership_details():
    block = company.company_facts_block()
    assert "TrailerPlace" in block
    assert "Wharton, TX" in block
    assert "979-532-1486" in block


def test_the_website_line_always_carries_a_url(monkeypatch):
    """An unset env var must not produce "have a look at " with nothing after it.

    Settings is a frozen dataclass, so the module attribute is swapped rather than one of
    its fields - setattr on a frozen instance raises, and it raises again in teardown,
    which takes the rest of the session down with it.
    """
    monkeypatch.setattr(company, "settings", dataclasses.replace(company.settings,
                                                                trailerplace_website=""))
    assert "https://" in company.website_redirect_line()


def test_no_email_scenario_leaked_into_the_company_module():
    """Brief S27: the email half of the reference document is explicitly out of scope."""
    source = company.__doc__ + "".join(
        str(getattr(company, name)) for name in dir(company) if name.isupper()
    )
    for term in ("outbox", "faq_key", "escalation_alert", "team_request", "interested_listing"):
        assert term not in source.lower()
