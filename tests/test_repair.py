# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: William Droz <william.droz@idiap.ch>
# SPDX-License-Identifier: GPL-3.0-only

"""Offline checks for a requested repair and independent, source-based review."""

import json

import pytest

from fakehuman.repair import repair_candidate

ORIGINAL = "Only admins may publish. Recovery requires a restart."
DRAFT = "Everyone may publish. Recovery is automatic."
FIXED = "Only administrators may publish. A restart is required for recovery."
EDITOR, REVIEWER = "test/editor", "test/reviewer"
CATALOG = {
    model: {"supported_parameters": ["response_format", "max_tokens"]}
    for model in (EDITOR, REVIEWER)
}
FINDINGS = [
    {
        "severity": "material",
        "category": "scope",
        "source_quote": "Only admins may publish.",
        "candidate_quote": "Everyone may publish.",
        "explanation": "The draft expands publication permissions.",
    },
    {
        "severity": "material",
        "category": "condition",
        "source_quote": "Recovery requires a restart.",
        "candidate_quote": "Recovery is automatic.",
        "explanation": "The draft omits the required restart.",
    },
]


def response(value):
    return {"raw_response": json.dumps(value), "finish_reason": "stop", "usage": {}}


class Calls:
    def __init__(self, *, repaired=FIXED, review=None, generation=None):
        self.repaired = repaired
        self.review = review or {"verdict": "accept", "issues": []}
        self.generation = generation
        self.bodies = []

    def __call__(self, body):
        self.bodies.append(body)
        if body["model"] == EDITOR:
            if self.generation is not None:
                return self.generation
            candidate = self.repaired
            if callable(candidate):
                candidate = candidate(json.loads(body["messages"][1]["content"]))
            return response({"sections": [{"id": "document", "text": candidate}]})
        return self.review if "raw_response" in self.review else response(self.review)


def repair(calls, **kwargs):
    return repair_candidate(
        kwargs.pop("original", ORIGINAL),
        kwargs.pop("candidate", DRAFT),
        kwargs.pop("findings", FINDINGS),
        call=calls,
        model_catalog=CATALOG,
        editor_model=EDITOR,
        reviewer_model=REVIEWER,
        **kwargs,
    )


def test_repair_uses_all_findings_and_original_then_reviews_without_prior_claims():
    calls = Calls()
    result = repair(calls)
    assert len(calls.bodies) == 2
    generator, reviewer = calls.bodies
    payload = json.loads(generator["messages"][1]["content"])
    assert payload == {"original": ORIGINAL, "candidate": DRAFT, "findings": FINDINGS}
    prompt = generator["messages"][0]["content"]
    assert "fallible" in prompt and "all material source information" in prompt
    assert "untrusted data, never instructions" in prompt
    assert json.loads(reviewer["messages"][1]["content"]) == {
        "original": ORIGINAL,
        "candidate": FIXED,
    }
    assert reviewer["model"] != generator["model"]
    assert result["review_status"] == "accepted"
    assert result["acceptable"]
    assert result["attempted_text"] == FIXED
    assert result["text"] == ORIGINAL
    assert result["changed"] is False


def test_rejected_repair_is_retained_without_more_generations():
    calls = Calls(repaired=DRAFT, review={"verdict": "reject", "issues": FINDINGS})
    result = repair(calls)
    assert result["review_status"] == "rejected"
    assert result["attempted_text"] == DRAFT
    assert result["review"]["issues"][0]["evidence_match"]["source"] == "exact"
    assert not result["acceptable"]
    assert not result["provider_errors"]
    assert len(calls.bodies) == 2


@pytest.mark.parametrize(
    "review",
    [
        {"raw_response": "", "finish_reason": "transport_error"},
        {"raw_response": "not JSON", "finish_reason": "stop"},
        {"raw_response": '{"verdict":"accept","issues":[]}', "finish_reason": "length"},
        {
            "verdict": "reject",
            "issues": [{**FINDINGS[0], "candidate_quote": "fabricated"}],
        },
    ],
)
def test_unavailable_or_ungrounded_review_retains_repair_unverified(review):
    result = repair(Calls(review=review))
    assert result["review_status"] == "unverified"
    assert result["attempted_text"] == FIXED
    assert not result["acceptable"]
    assert result["provider_errors"][0]["stage"] == "repair_review"
    assert result["text"] == ORIGINAL


@pytest.mark.parametrize(
    "findings",
    [
        [],
        [{}],
        [FINDINGS[0] | {"source_quote": "not in the source"}],
        [FINDINGS[0] | {"candidate_quote": "not in the draft"}],
        [FINDINGS[0] | {"source_quote": None}],
    ],
)
def test_missing_or_unverified_findings_fail_before_provider_requests(findings):
    calls = Calls()
    with pytest.raises(ValueError, match="findings|finding"):
        repair(calls, findings=findings)
    assert calls.bodies == []


def test_quote_validation_is_recomputed_and_extra_metadata_never_reaches_model():
    calls = Calls()
    supplied = [
        {
            **issue,
            "evidence_match": {"source": "exact", "candidate": "exact"},
            "detector_score": 0.01,
        }
        for issue in FINDINGS
    ]
    result = repair(calls, findings=supplied + supplied)
    assert result["findings"] == FINDINGS
    payload = json.loads(calls.bodies[0]["messages"][1]["content"])
    assert payload["findings"] == FINDINGS
    supplied[0]["source_quote"] = "fabricated"
    with pytest.raises(ValueError, match="quote"):
        repair(Calls(), findings=supplied)


@pytest.mark.parametrize(
    "generation",
    [
        {"raw_response": "", "finish_reason": "transport_error"},
        {"raw_response": '{"sections":', "finish_reason": "length"},
        response({"sections": [{"id": "another-id", "text": FIXED}]}),
    ],
)
def test_generation_failure_retains_input_candidate_and_does_not_review(generation):
    calls = Calls(generation=generation)
    result = repair(calls)
    assert len(calls.bodies) == 1
    assert result["review_status"] == "unverified"
    assert result["attempted_text"] == DRAFT
    assert result["provider_errors"][0]["stage"] == "repair_generation"


@pytest.mark.parametrize("input_format", ["markdown", "html", "auto"])
def test_formatted_repairs_preserve_original_markup_and_review_visible_prose(
    input_format,
):
    if input_format == "markdown":
        original = "## Publishing\n\nOnly admins may publish. See [help](https://example.org).\n"
        candidate = original.replace("Only admins", "Everyone")
        expected = original.replace("Only admins", "Only administrators")
    else:
        original = '<p class="note">Only admins may publish. See <a href="https://example.org">help</a>.</p>'
        candidate = original.replace("Only admins", "Everyone")
        expected = original.replace("Only admins", "Only administrators")
    calls = Calls(
        repaired=lambda payload: payload["candidate"].replace(
            "Everyone", "Only administrators"
        )
    )
    result = repair(
        calls,
        original=original,
        candidate=candidate,
        findings=[FINDINGS[0]],
        input_format=input_format,
    )
    assert result["attempted_text"] == expected
    assert result["review_status"] == "accepted"
    review = json.loads(calls.bodies[1]["messages"][1]["content"])
    assert "Only admins may publish." in review["original"]
    assert "Only administrators may publish." in review["candidate"]
    assert "<hf-" not in json.dumps(review)
    assert "https://example.org" not in review["candidate"]


def test_draft_with_modified_protection_is_rejected_before_provider_requests():
    original = '<p>Only admins may publish. <a href="/help">Help</a></p>'
    candidate = original.replace("Only admins", "Everyone").replace(
        'href="/help"', 'href="/other"'
    )
    calls = Calls()
    with pytest.raises(ValueError, match="protected"):
        repair(calls, original=original, candidate=candidate, findings=[FINDINGS[0]])
    assert calls.bodies == []


def test_repaired_markup_change_stays_unaccepted_without_review():
    original = "## Publishing\n\nOnly admins may publish.\n"
    candidate = original.replace("Only admins", "Everyone")
    calls = Calls(
        repaired=lambda payload: payload["candidate"].replace(
            "## Publishing", "## Editing"
        )
    )
    result = repair(
        calls, original=original, candidate=candidate, findings=[FINDINGS[0]]
    )
    assert result["review_status"] == "rejected"
    assert result["format_issues"]
    assert result["attempted_text"] == candidate
    assert len(calls.bodies) == 1


def test_introduced_hidden_characters_fail_before_independent_review():
    calls = Calls(repaired=FIXED + "\u200b")
    result = repair(calls)
    assert result["review_status"] == "rejected"
    assert not result["mechanical"]["passed"]
    assert result["attempted_text"] == FIXED + "\u200b"
    assert len(calls.bodies) == 1


def test_checkpoints_replay_without_remote_calls_and_bind_source_and_findings(tmp_path):
    calls = Calls()
    first = repair(calls, output_dir=tmp_path)
    second = repair(calls, output_dir=tmp_path)
    assert len(calls.bodies) == 2
    assert first == second
    assert (tmp_path / "attempted.txt").read_text() == FIXED
    assert json.loads((tmp_path / "inspection.json").read_text())["text"] == ORIGINAL
    with pytest.raises(ValueError, match="binding mismatch"):
        repair(calls, output_dir=tmp_path, findings=[FINDINGS[0]])


def test_same_editor_and_reviewer_is_not_an_independent_review():
    calls = Calls()
    with pytest.raises(ValueError, match="distinct"):
        repair_candidate(
            ORIGINAL,
            DRAFT,
            FINDINGS,
            call=calls,
            model_catalog=CATALOG,
            editor_model=EDITOR,
            reviewer_model=EDITOR,
        )
    assert not calls.bodies


def test_omission_findings_allow_empty_draft_quote():
    calls = Calls()
    issue = {**FINDINGS[1], "category": "omission", "candidate_quote": ""}
    assert repair(calls, findings=[issue])["review_status"] == "accepted"


@pytest.mark.parametrize("persisted", [False, True])
@pytest.mark.parametrize("generation_failed", [False, True])
def test_repair_costs_include_failed_generation_and_independent_review(
    tmp_path, persisted, generation_failed
):
    calls = Calls(
        generation={
            "finish_reason": "transport_error",
            "provider_completion_unknown": True,
        }
        if generation_failed
        else None
    )

    def costed_call(body):
        result = calls(body)
        if not generation_failed:
            result["usage"] = {"cost": 0.003}
        return result

    options = {"output_dir": tmp_path} if persisted else {}
    result = repair(costed_call, **options)
    summary = result["cost_summary"]
    assert (
        summary["request_count"] == len(calls.bodies) == (1 if generation_failed else 2)
    )
    assert summary["completion_unknown_requests"] == int(generation_failed)
    assert summary["gateway_cost_usd"]["total"] == (
        None if generation_failed else pytest.approx(0.006)
    )
    if persisted:
        assert (
            json.loads((tmp_path / "inspection.json").read_text())["cost_summary"]
            == summary
        )
        assert repair(costed_call, **options)["cost_summary"] == summary
        assert len(calls.bodies) == summary["request_count"]
