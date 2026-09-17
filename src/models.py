from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class TrailerListing:
    title: str
    url: str
    listing_id: str | None = None
    condition: str | None = None
    price: str | float | int | None = None
    price_display: str | None = None
    payments_from: str | float | int | None = None
    category_subcategory: str | None = None
    make: str | None = None
    color: str | None = None
    hitch_type: str | None = None
    year: Any = None
    length: Any = None
    width: Any = None
    axles: Any = None
    gvwr: Any = None
    payload_capacity: Any = None
    trailer_material: str | None = None
    floor: str | None = None
    score: float | int | None = None
