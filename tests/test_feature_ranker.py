"""The non-metadata feature path: sanitising, the gpt-5-nano assessment, and the 85/15 blend.

The model is faked. ``FakeNano`` answers every batch by looking for a keyword in each
candidate's evidence, which is what a correct assessment would do; individual tests then
bend it to return the failures Python has to catch (an invented quote, a skipped listing, a
batch that errors). The live behaviour - does the main model file the right words as
features, and does the real reranker move matching trailers up - is exercised by
``python scripts/scenario_run.py --scenarios "" --scripted features``.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from src.domain.slot_map import sanitize_non_metadata_features
from src.llm import usage
from src.search import feature_ranker, listing_search
from src.search.feature_ranker import (
    CandidateFeatureAssessment,
    FeatureMatch,
    FeatureRerankOutput,
    rank_non_metadata_features,
)


# ------------------------------------------------------------------------------ fixtures
def listing(n: int, *, length: float = 16.0, features: tuple[str, ...] = ()) -> dict:
    return {
        "url": f"https://www.trailerplace.com/inventory/{n}",
        "title": f"2026 Iron Bull Dump - {n}",
        "features": list(features),
        "match_evidence_text": "",
        "model": "", "trim": "", "make": "Iron Bull Trailers", "category": "Dump",
        "length": length, "width": 7.0, "height": None, "payload_capacity": 9000.0,
        "gvwr": 14000.0, "axle_capacity": None, "total_axle_capacity": None,
        "axle_count": None, "relevance_score": 0.0,
    }


class FakeNano:
    """Stands in for ``OpenAI().beta.chat.completions.parse``.

    ``keyword`` decides a match: a candidate whose evidence contains it is matched, quoting
    the line that contains it. ``tamper`` can rewrite one candidate's assessment to test the
    validator, and ``fail_on`` makes the batch holding that candidate raise.
    """

    def __init__(self, keyword: str, *, tamper=None, fail_on: str | None = None):
        self.keyword = keyword
        self.tamper = tamper
        self.fail_on = fail_on
        self.batches: list[list[str]] = []
        self.beta = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(parse=self.parse)))

    def parse(self, *, messages, **_kwargs):
        payload = json.loads(messages[-1]["content"])
        ids = [c["candidate_id"] for c in payload["candidates"]]
        self.batches.append(ids)
        if self.fail_on in ids:
            raise TimeoutError("nano timed out")
        assessments = []
        for candidate in payload["candidates"]:
            matches = []
            for feature in payload["requested_features"]:
                line = next((l for l in candidate["evidence"].splitlines()
                             if self.keyword in l.lower()), None)
                matches.append(FeatureMatch(
                    requested_feature=feature, matched=bool(line), evidence=line,
                    reason_code="explicit" if line else "not_found",
                    reason="The listing names it." if line else None,
                ))
            assessment = CandidateFeatureAssessment(candidate_id=candidate["candidate_id"],
                                                    feature_matches=matches)
            if self.tamper:
                assessment = self.tamper(assessment)
            if assessment is not None:
                assessments.append(assessment)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(
                parsed=FeatureRerankOutput(assessments=assessments), refusal=None))],
            usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20),
            _request_id=f"req_{len(self.batches)}",
        )


@pytest.fixture
def nano(monkeypatch):
    """Install a FakeNano; the test sets its keyword and failure modes."""
    def install(keyword: str, **kwargs) -> FakeNano:
        fake = FakeNano(keyword, **kwargs)
        monkeypatch.setattr(feature_ranker, "_openai_client", lambda: fake)
        return fake
    return install


@pytest.fixture
def catalogue(monkeypatch):
    """Serve search_listing_result from a list instead of Postgres."""
    def install(rows: list[dict]) -> None:
        monkeypatch.setattr(listing_search, "fetch_listings", lambda _filters: rows)
        monkeypatch.setattr(listing_search, "_row_to_listing", lambda row: dict(row))
    return install


# --------------------------------------------------------------- what counts as a feature
def test_real_equipment_is_kept_as_a_feature():
    kept, hitch = sanitize_non_metadata_features(["ramp door", "LED lights", "tarp system"])
    assert kept == ["ramp door", "LED lights", "tarp system"]
    assert hitch is None


def test_a_hitch_stated_as_a_feature_becomes_the_hitch_filter():
    kept, hitch = sanitize_non_metadata_features(["gooseneck hitch", "winch"])
    assert hitch == ["Gooseneck"] or hitch == "Gooseneck"
    assert kept == ["winch"]


def test_axle_counts_and_ratings_are_not_features():
    kept, _ = sanitize_non_metadata_features(["tandem axles", "7000 lb axles", "spare tire"])
    assert kept == ["spare tire"]


def test_the_axle_type_is_a_feature():
    """Torsion, unlike tandem or 7,000 lb, is construction - no field holds it."""
    kept, _ = sanitize_non_metadata_features(["torsion axles"])
    assert kept == ["torsion axles"]


def test_bare_sizes_and_prices_are_not_features():
    kept, _ = sanitize_non_metadata_features(["20 ft", "under $9,000", "winch"])
    assert kept == ["winch"]


def test_the_prompt_says_what_a_feature_is_and_is_not():
    from src.llm.prompt import system_prompt

    prompt = system_prompt()
    assert "non_metadata_features" in prompt
    assert "NEVER a feature" in prompt
    assert "scissor lift" in prompt and "mobile coffee business" in prompt


# -------------------------------------------------------------- the nano assessment itself
def test_every_candidate_is_assessed_and_grounded(nano):
    fake = nano("tarp")
    rows = [listing(1, features=("Tarp system",)), listing(2), listing(3, features=("Tarp kit",))]

    result = rank_non_metadata_features(rows, ["tarp"])

    assert set(result.assessments_by_id) == {"C001", "C002", "C003"}
    assert not result.fallback_candidate_ids
    matched = {cid for cid, a in result.assessments_by_id.items() if a.feature_matches[0].matched}
    assert matched == {"C001", "C003"}
    assert fake.batches == [["C001", "C002", "C003"]]


def test_a_quote_that_is_not_in_the_listing_is_rejected(nano):
    """The whole point of the evidence field: nano cannot claim a feature it did not read."""
    def invent(assessment):
        if assessment.candidate_id == "C002":
            assessment.feature_matches[0] = FeatureMatch(
                requested_feature="tarp", matched=True, evidence="Feature: Heavy duty tarp",
                reason_code="explicit", reason="Says tarp.",
            )
        return assessment

    nano("tarp", tamper=invent)
    result = rank_non_metadata_features([listing(1, features=("Tarp system",)), listing(2)], ["tarp"])

    assert result.fallback_candidate_ids == {"C002"}
    assert "ungrounded evidence" in result.validation_errors_by_id["C002"]
    assert "C001" in result.assessments_by_id


def test_a_match_with_a_non_match_reason_code_is_rejected(nano):
    def contradict(assessment):
        match = assessment.feature_matches[0]
        if match.matched:
            assessment.feature_matches[0] = match.model_copy(update={"reason_code": "uncertain"})
        return assessment

    nano("tarp", tamper=contradict)
    result = rank_non_metadata_features([listing(1, features=("Tarp system",))], ["tarp"])
    assert result.fallback_candidate_ids == {"C001"}


def test_a_skipped_candidate_falls_back_instead_of_vanishing(nano):
    nano("tarp", tamper=lambda a: None if a.candidate_id == "C002" else a)
    result = rank_non_metadata_features([listing(1), listing(2), listing(3)], ["tarp"])

    assert result.fallback_candidate_ids == {"C002"}
    assert result.validation_errors_by_id["C002"] == "model omitted candidate assessment"


def test_candidates_go_in_batches_of_twenty(nano):
    fake = nano("tarp")
    rows = [listing(n) for n in range(1, 46)]

    result = rank_non_metadata_features(rows, ["tarp"])

    assert sorted(len(batch) for batch in fake.batches) == [5, 20, 20]
    assert result.batch_count == 3
    assert len(result.assessments_by_id) == 45


def test_one_failed_batch_keeps_the_other_batches(nano):
    nano("tarp", fail_on="C021")
    rows = [listing(n) for n in range(1, 46)]

    result = rank_non_metadata_features(rows, ["tarp"])

    assert result.fallback_candidate_ids == {f"C{n:03d}" for n in range(21, 41)}
    assert len(result.assessments_by_id) == 25


def test_each_batch_is_counted_as_a_feature_rerank_call(nano):
    nano("tarp")
    with usage.usage_scope() as turn:
        rank_non_metadata_features([listing(n) for n in range(1, 26)], ["tarp"])
    assert turn.feature_reranks == 2
    assert turn.prompt_tokens == 200


# ----------------------------------------------------------------- the 85/15 blend, whole
def _search(features: list[str]) -> list[str]:
    result = listing_search.search_listing_result(
        category="Dump", slots={"length": 16.0}, requested_features=features,
        max_recommendations=5,
    )
    return [item["url"].rsplit("/", 1)[-1] for item in result.listings]


def test_a_matching_trailer_outranks_a_better_fitting_one(nano, catalogue):
    """85% feature coverage beats 15% fit: the 18 ft with a tarp goes above the 16 ft without."""
    nano("tarp")
    catalogue([listing(1, length=16.0), listing(2, length=18.0, features=("Tarp system",))])

    assert _search(["tarp"])[0] == "2"


def test_without_features_the_order_is_fit_and_nano_is_never_called(monkeypatch, catalogue):
    def boom(*_args, **_kwargs):
        raise AssertionError("nano must not run without requested features")

    monkeypatch.setattr(listing_search, "rank_non_metadata_features", boom)
    catalogue([listing(1, length=18.0), listing(2, length=16.0, features=("Tarp system",))])

    assert _search([])[0] == "2", "closest length first"


def test_when_nano_fails_the_keyword_matcher_still_ranks_by_feature(monkeypatch, catalogue):
    def down(*_args, **_kwargs):
        raise TimeoutError("nano unavailable")

    monkeypatch.setattr(listing_search, "rank_non_metadata_features", down)
    catalogue([listing(1, length=16.0), listing(2, length=18.0, features=("Tarp system",))])

    assert _search(["tarp"])[0] == "2"
