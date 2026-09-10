# 5.6 Luna TrailerPlace — repo overview

A trimmed extraction from the **New Prompt** chatbot, carrying only the inventory domain
model and the retrieval path. This is not the full chatbot: there is no graph build, no
LLM analyze/respond stage, no API and no conversation persistence.

What it does contain, end to end: a customer's stated requirements → normalized slots →
SQL filters → ranked trailer listings, plus the two tool calls that drive it.

---

## What lives where

### Root

| File | What it is |
|---|---|
| `pyproject.toml` | uv project. Dependencies pinned to New Prompt's exact freeze (124 packages). |
| `requirements.txt` | The same pins as a plain freeze. `uv.lock` is generated from `pyproject.toml`. |
| `.env` | Azure Postgres credentials, OpenAI key, LangSmith tracing, rerank tuning knobs. Read by `src/config.py`. |
| `listings.xlsx` | Fallback catalogue. Only read when the database is off or `trailer_listings` is empty. |
| `overview.md` | This file. |

### `src/` — infrastructure

| File | What it is |
|---|---|
| `config.py` | Every runtime setting, read from `.env` into one frozen `settings` object (~55 env keys): DB connection, search limits, rerank weights and thresholds, feature-rerank model and timeouts. |
| `db.py` | SQLAlchemy engine and session factory. `database_enabled()` is the switch every DB-touching call checks first, so the code degrades to the workbook instead of crashing. |
| `db_models.py` | ORM tables: `trailer_listings` (the catalogue search reads), plus `chatbot_leads`, `chatbot_conversations`, `chatbot_turns`, `chatbot_outbox`. Only `trailer_listings` is used by the code here. |
| `turn_status.py` | Small publish/read channel for the "let me check what we have" line, so a UI can show progress while a search is still running. |
| `tracing.py` | LangSmith wrappers. No-ops cleanly when tracing is off. |

### `src/domain/` — what a trailer *is*

| File | What it is |
|---|---|
| `trailer_fields.py` | The qualification questionnaire. Per category: which slots are required, which optional, the question text, and the guidance telling the extractor what counts as a valid answer. |
| `categories.py` | The 14 canonical categories and the two-tier synonym map that resolves free text to one. `_NAMING_TERMS` (an actual type name) outranks `_CARGO_TERMS` (cargo that merely implies one), so "tilt trailer to haul a tractor" resolves to Tilt, not Equipment. |
| `slot_map.py` | The slot engine. What kind of value each slot holds, how to parse it, which search filter it fills, and which slots are interchangeable. Also the feature sanitizer that keeps sizes, prices, hitches and axle numbers out of the free-text feature list. |
| `brands.py` | Which makes and categories we actually stock — read from `trailer_listings` so the prompts can never advertise inventory the search cannot return. Falls back to `listings.xlsx`. |
| `normalizer.py` | Canonicalizes category / make / hitch / subcategory strings coming from the catalogue or the customer. |
| `make_aliases.py` | Every spelling of every brand, folded onto its canonical short display name. |
| `units.py` | Measurement parsing: feet, inches, pounds, tons, ranges ("15–18 ft" → 15), and combined dimensions ("8x25x6.5" → width / length / height). |

### `src/search/` — finding the trailers

| File | What it is |
|---|---|
| `listing_search.py` | The main retrieval path: builds the SQL hard gates, fetches rows, runs the rerank stages, returns the final ordered list. |
| `inventory_matcher.py` | The other kind of lookup — a customer naming a *specific* unit by stock number, year, make or model. Returns `exact` / `ambiguous` / `no_exact` / `none`. |
| `feature_ranker.py` | LLM rerank for wants we hold no metadata field for ("sliding gates", "torsion axles"). Only runs when such features are present; costs tokens when it does. |

### `src/graph/nodes/` — the two tool calls

| File | What it is |
|---|---|
| `search.py` | `search_node` — the `TOOL search` call. Slots → filters → search, with two fallbacks if nothing comes back (drop the brand, then drop every gate but the category). Also picks the status line. |
| `inventory_lookup.py` | `inventory_lookup_node` — the `TOOL inventory_lookup` call. Rides along with any intent and asserts it never mutates qualification state. |

### `src/llm/`

| File | What it is |
|---|---|
| `schemas.py` | Pydantic contracts for LLM output: `ExtractedFields` (volunteered values), `InventoryLookup`, `TurnAnalysis`, `ReplyOutput`. |
| `usage.py` | Token and cost accounting. |

---

## The slot model

One name per concept, shared across every category:

| Slot | Holds | Fills the search filter |
|---|---|---|
| `length` | trailer / load / vehicle length in ft | `length_ft` |
| `width` | width in ft | `width_ft` |
| `height` | height in ft | `height_ft` |
| `payload_capacity` | load weight or payload requirement in lbs | `payload_lbs` |
| `axle_capacity` | **per-axle** rating in lbs | `axle_capacity_lbs` |
| `haul_item` | what they are hauling, free text | — (ranking only) |

Category-specific questions keep their own names: `hitch_type`, `base_category`,
`bin_size`, `tank_capacity`, `deck_style`, `dump_mechanism`, `tilt_style`,
`loading_style`, `open_vs_covered`, `gate_preferences`, `cdl_concern`, `fuel_type`,
`crew_size`, `sleeping_need`, `ac_windows_cabinets`, `finished_interior`,
`fiber_amenities`, `race_amenities`, `total_axle_capacity_lbs`, `axle_count`.

**Combined-size questions** — `trailer_size` ("length / width?"), `cargo_size`
("length × width × height") and `bin_size` (yardage) — also keep their own names, because
they ask for several numbers at once. Their answers still land on the
`length_ft` / `width_ft` / `height_ft` filters via `_SLOT_METADATA_FILTER_MAP`, and they
may *donate* a value to `length` but never *receive* one (`_NO_AUTOFILL_SLOTS`) —
otherwise a length carried over from another category would silently answer a question
the customer was never asked.

Slot names are **not** the search filter names. Slots are what the customer is asked;
`length_ft`, `payload_lbs` and `axle_capacity_lbs` are the DB-facing filter targets. The
translation happens in `slot_map.normalize_slot_targets`.

### Per-category slots

| Category | Required | Optional |
|---|---|---|
| Aluminum | `base_category`, `payload_capacity` | `sleeping_need` |
| Car Hauler | `haul_item`, `payload_capacity`, `length` | `open_vs_covered` |
| Concession | `length` | `ac_windows_cabinets`, `finished_interior` |
| Diesel Tank | `fuel_type`, `tank_capacity` | `deck_style`, `cdl_concern` |
| Dump | `haul_item`, `payload_capacity` | `dump_mechanism` |
| Enclosed | `haul_item`, `cargo_size` | `ac_windows_cabinets`, `finished_interior` |
| Equipment | `haul_item`, `payload_capacity`, `length`, `hitch_type` | `loading_style` |
| Fiber | `haul_item` | `crew_size`, `fiber_amenities` |
| Flatbed | `haul_item`, `payload_capacity` | `deck_style`, `cdl_concern` |
| Livestock | `length` | `gate_preferences` |
| Race Trailer | `haul_item`, `length` | `race_amenities` |
| Roll Off | `bin_size` | `deck_style`, `cdl_concern` |
| Tilt | `haul_item`, `payload_capacity` | `tilt_style` |
| Utility | `haul_item`, `payload_capacity` | `trailer_size` |
| Welding | `haul_item` | `payload_capacity` |

`Welding` is not a canonical category and is unreachable through normal qualification;
it is kept only for the ingest-side normalizer mapping.

---

## The three rerank stages

Ranking is three passes, and only one of them calls a model:

1. **Fit rerank** — `listing_search._rerank_listings_by_fit`. Pure arithmetic on length,
   payload (falling back to GVWR), width, height, per-axle capacity, total axle capacity
   and axle count. Each dimension gives a `candidate / required` ratio; under-spec is
   penalized steeply, over-spec mildly and escalating past the warn and extreme
   thresholds, and a dimension missing from the listing takes a fixed penalty. The axle
   terms never hard-cull, because most of the catalogue carries no axle rating.
2. **Feature rerank** — `feature_ranker.rank_non_metadata_features`. LLM. Skipped
   entirely when no non-metadata features were requested.
3. **Brand quota rerank** — `listing_search._apply_category_make_preference`.
   Deterministic; interleaves makes so one brand cannot fill the whole screen.

---

## Gotcha worth knowing

The fit reranker reads slot names by hand in `_required_*_from_filters`. An unrecognized
key does **not** raise — it yields
`rerank_debug={"applied": false, "reason": "missing_clear_requirements_or_no_listings"}`
and ranking silently falls back to fetch order. If you rename a slot, update those
readers *and* the SQL `min_length` block in `_listing_filters`, which resolves length
independently.

---

## Running it

    .venv/Scripts/python.exe -c "
    from src.graph.nodes.search import search_node
    state = {'session_id':'demo','qualification_complete':True,'category':'Dump',
             'slots':{'length':12,'payload_capacity':10000,'haul_item':'gravel'},
             'non_metadata_features':[],'shown_urls':[]}
    print(search_node(state)['turn_outcome']['result_count'])
    "

Both tool nodes hit the live Azure `trailer_listings` table using the credentials in
`.env`. Reads only — nothing here writes to the database.
