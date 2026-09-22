""""How much does an Aluma 8218 cost?" found nothing, and we have one.

The model code is read off our own ``model`` column, where it is always the token carrying
letters ("8218ESA-TA-EL-R-RTD"), so ``extract_model_code`` skips bare numbers - the digits
in that column are trim and dimensions. Customers do not talk that way: half of Aluma's
range is named by its number alone, and that reading left the lookup with no code at all.
"Iron Bull DTB" worked the whole time, which is why it looked like an Aluma problem.
"""
from __future__ import annotations

import pandas as pd

from src.search.inventory_matcher import (
    extract_model_code,
    match_inventory,
    prepare_inventory,
    requested_model_code,
    _Identifiers,
)


def _lot() -> pd.DataFrame:
    return prepare_inventory(pd.DataFrame([
        {"title": "2027 Aluma 8218ESA TA EL R RTD - 13629", "url": "https://x/8218",
         "year": 2027, "make": "Aluma", "model": "8218ESA-TA-EL-R-RTD", "stock_number": "13629"},
        {"title": "2027 Aluma 8118W 10k TA EL R RTD - 18450", "url": "https://x/8118",
         "year": 2027, "make": "Aluma", "model": "8118W-10K-TA-EL-R-RTD", "stock_number": "18450"},
        {"title": "2026 Iron Bull Trailers DTB 15K 72 x 10 - 11154", "url": "https://x/dtb",
         "year": 2026, "make": "Iron Bull Trailers", "model": "DTB-15K", "stock_number": "11154"},
    ]))


def test_a_number_is_a_model_code_when_the_customer_says_it():
    assert requested_model_code("8218") == "8218"
    assert requested_model_code("8218ESA") == "8218esa"
    assert requested_model_code("DTB") == "dtb"


def test_a_year_is_not_a_model_code():
    """It has its own identifier, and as a code it would match every trailer of that year."""
    assert requested_model_code("2027") == ""


def test_our_own_column_is_read_the_same_way_as_before():
    assert extract_model_code("8218ESA-TA-EL-R-RTD") == "8218esa"
    assert extract_model_code("2027") == ""


def test_the_aluma_8218_is_found():
    result = match_inventory(
        _Identifiers(possible_make="Aluma", possible_model_code="8218", possible_model_text="8218"),
        df=_lot(),
    )

    assert [match["title"] for match in result["top_matches"]] == [
        "2027 Aluma 8218ESA TA EL R RTD - 13629"
    ]


def test_the_8118_is_not_offered_as_an_8218():
    """Fuzzy matching is right for letters - "Diamnod C" means Diamond C - but 8218 and
    8118 are two different trailers and a ratio cannot tell them apart."""
    result = match_inventory(
        _Identifiers(possible_make="Aluma", possible_model_code="8218", possible_model_text="8218"),
        df=_lot(),
    )

    assert all("8118" not in match["title"] for match in result["top_matches"])
    assert result["exact_match_count"] == 1


def test_a_letter_code_still_returns_every_trailer_that_carries_it():
    result = match_inventory(
        _Identifiers(possible_make="Iron Bull", possible_model_code="dtb", possible_model_text="DTB"),
        df=_lot(),
    )

    assert result["exact_match_count"] == 1
    assert "DTB" in result["top_matches"][0]["title"]
