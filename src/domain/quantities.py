"""Turn a quantity the model read into the number a slot stores. Pure, no I/O.

The split is deliberate. Reading "three and a half thousand pounds", "seven and a half feet"
or a unitless "144 x 72" is language, and the model reads language far better than a
regex - a live run had the regex parser store those as 1,000 lbs, 0.5 ft and 144 ft. But
arithmetic is not the model's strength, so it hands over the number as it was MEANT, in the
unit it was meant in, and the conversion happens here, from one table.

Two rules are enforced here rather than trusted to the model:

* a range resolves to its SMALLER end (brief S15) - the model was seen averaging;
* a value outside what a trailer could plausibly need is not stored (``check``), so a
  misread unit is asked about again instead of silently skewing the search.
"""
from __future__ import annotations

from typing import Any

from src.domain import slot_map

_TO_FEET = {"ft": 1.0, "in": 1 / 12, "yd": 3.0, "m": 3.280839895, "cm": 0.03280839895, "mm": 0.003280839895}
_TO_POUNDS = {"lb": 1.0, "kg": 2.2046226218, "ton": 2000.0, "tonne": 2204.6226218}
_TO_GALLONS = {"gal": 1.0, "l": 0.2641720524}
# Bin sizes are yardage, and they stay yardage (15 yd is stored as 15, see slot_map).
_BIN_YARDS = {"yd": 1.0, "cu_yd": 1.0}

# What each slot's number is measured in, and the table that gets it there.
_TABLE_BY_KIND = {
    "length_ft": _TO_FEET,
    "width_ft": _TO_FEET,
    "height_ft": _TO_FEET,
    "payload_lbs": _TO_POUNDS,
    "axle_capacity_lbs": _TO_POUNDS,
    "total_axle_capacity_lbs": _TO_POUNDS,
}
_TABLE_BY_SLOT = {"bin_size": _BIN_YARDS, "tank_capacity": _TO_GALLONS}

# The range a trailer requirement can sensibly take, in the stored unit. Wide on purpose -
# this is a guard against a misread UNIT (144 in taken as 144 ft), not a sales opinion. A
# value outside it is asked about again, never clamped.
PLAUSIBLE: dict[str, tuple[float, float]] = {
    "length_ft": (3.0, 60.0),        # the longest trailers are 53 ft
    "width_ft": (2.0, 12.0),         # trailers top out at 8.5 ft; a wide machine a little more
    "height_ft": (1.0, 15.0),
    "payload_lbs": (20.0, 100_000.0),
    "axle_capacity_lbs": (500.0, 30_000.0),
    "total_axle_capacity_lbs": (500.0, 100_000.0),
    "bin_size": (2.0, 60.0),         # yards
    "tank_capacity": (5.0, 15_000.0),  # gallons
}


def _field(quantity: Any, name: str) -> Any:
    return quantity.get(name) if isinstance(quantity, dict) else getattr(quantity, name, None)


def _kind(slot: str) -> str | None:
    if slot in _TABLE_BY_SLOT:
        return slot
    kind = slot_map.slot_value_kind(slot)
    return kind if kind in _TABLE_BY_KIND else None


def supports(slot: str) -> bool:
    """Whether a quantity for this slot can be converted here at all."""
    return _kind(slot) is not None


def to_canonical(slot: str, quantity: Any) -> float | None:
    """The stored number for this slot, or None when the unit does not fit the slot.

    A range is its smaller end, whatever the model put first. The sign is kept: a negative
    is for the caller to reject, not for this function to quietly flip.
    """
    kind = _kind(slot)
    if kind is None:
        return None
    table = _TABLE_BY_SLOT.get(slot) or _TABLE_BY_KIND[kind]
    factor = table.get(str(_field(quantity, "unit") or ""))
    if factor is None:
        return None
    low = _field(quantity, "low")
    if low is None:
        return None
    high = _field(quantity, "high")
    value = min(float(low), float(high)) if high is not None else float(low)
    return round(value * factor, 2)


def check(slot: str, value: float) -> str:
    """ "ok", or "implausible" when no trailer requirement looks like this."""
    kind = _kind(slot)
    bounds = PLAUSIBLE.get(slot) or PLAUSIBLE.get(kind or "")
    if bounds is None or value <= 0:
        # Zero and below are not ours to judge: zero is "no preference" and a negative is
        # re-asked, both by the existing rules downstream.
        return "ok"
    low, high = bounds
    return "ok" if low <= value <= high else "implausible"
