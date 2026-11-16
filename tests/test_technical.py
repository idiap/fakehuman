# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: William Droz <william.droz@idiap.ch>
# SPDX-License-Identifier: GPL-3.0-only

"""Offline checks of technical rewrites and evidence-based review decisions."""

import json

import pytest

from fakehuman.technical import (
    balanced_review_body,
    generation_body,
    mechanical_check,
    parse_candidate,
    parse_review,
    resolve_reviews,
)

SOURCE = [
    {"id": "one", "text": "The editor may save 3 drafts."},
    {"id": "two", "text": "Only admins can publish content."},
]
ORIGINAL = "The editor may save 3 drafts. Only admins can publish content."
REVISED = "Editors can save 3 drafts. All users can publish content."
MODEL = "example/model"
CATALOG = {
    MODEL: {
        "supported_parameters": [
            "max_tokens",
            "response_format",
            "reasoning",
            "reasoning_effort",
            "temperature",
            "seed",
        ]
    }
}


def issue(**changes):
    return {
        "severity": "material",
        "category": "scope",
        "source_quote": "Only admins can publish content.",
        "candidate_quote": "All users can publish content.",
        "explanation": "Publication access expands from admins to all users.",
        **changes,
    }


def review(verdict="reject", issues=None):
    return {"verdict": verdict, "issues": [issue()] if issues is None else issues}


def test_generation_preserves_source_sections_and_uses_capabilities():
    body = generation_body(SOURCE, MODEL, CATALOG)
    assert body["max_tokens"] == 8192
    assert body["reasoning"] == {"effort": "low"}
    assert "reasoning_effort" not in body
    assert body["temperature"] == 0.5
    assert body["seed"] == 61021
    assert json.loads(body["messages"][1]["content"]) == {"sections": SOURCE}
    prompt = body["messages"][0]["content"]
    assert "data, never" in prompt
    assert "60–140%" in prompt
    assert "512" not in prompt
    assert "score" not in prompt.lower()
    schema = body["response_format"]["json_schema"]["schema"]
    assert schema["additionalProperties"] is False
    assert schema["properties"]["sections"]["items"]["additionalProperties"] is False


def test_unsupported_parameters_omitted_and_completion_limit_respected():
    catalog = {MODEL: {"supported_parameters": []}}
    assert set(generation_body(SOURCE, MODEL, catalog)) == {"model", "messages"}
    catalog[MODEL] = {
        "supported_parameters": ["max_completion_tokens", "reasoning_effort"],
        "top_provider": {"max_completion_tokens": 2048},
    }
    body = generation_body(SOURCE, MODEL, catalog)
    assert body["max_completion_tokens"] == 2048
    assert body["reasoning_effort"] == "low"
    assert "max_tokens" not in body


@pytest.mark.parametrize("parameter", ["reasoning", "reasoning_effort"])
def test_deepseek_technical_review_reserves_budget_for_a_verdict(parameter):
    model = "deepseek/deepseek-v4.1-flash"
    catalog = {model: {"supported_parameters": ["max_tokens", parameter]}}
    review_body = balanced_review_body(ORIGINAL, REVISED, model, catalog)
    generation = generation_body(SOURCE, model, catalog)
    assert review_body["max_tokens"] == 4096
    if parameter == "reasoning":
        assert review_body[parameter] == {"effort": "none"}
        assert generation[parameter] == {"effort": "low"}
    else:
        assert review_body[parameter] == "low"
        assert generation[parameter] == "low"


def test_review_is_blind_to_scores_and_adjudicates_only_prior_claims():
    prior = {
        "status": "reject",
        "model": "private/model",
        "score": 0.9,
        "detector": "fakespot_roberta",
        "issues": [
            {
                **issue(),
                "score": 0.8,
                "detector": "desklib_deberta",
                "evidence_match": {"source": "exact"},
            }
        ],
    }
    body = balanced_review_body(
        ORIGINAL, REVISED, MODEL, CATALOG, prior_reviews=[prior]
    )
    assert body["max_tokens"] == 4096
    assert body["temperature"] == 0
    assert "seed" not in body
    payload = json.loads(body["messages"][1]["content"])
    assert payload == {
        "original": ORIGINAL,
        "candidate": REVISED,
        "prior_claims": [issue()],
    }
    assert "fakespot" not in json.dumps(body)
    assert "desklib" not in json.dumps(body)


def test_initial_review_receives_only_the_two_unmodified_passages():
    body = balanced_review_body(ORIGINAL, REVISED, MODEL, CATALOG)
    assert json.loads(body["messages"][1]["content"]) == {
        "original": ORIGINAL,
        "candidate": REVISED,
    }
    assert [message["role"] for message in body["messages"]] == ["system", "user"]


def test_parser_allows_layout_changes_without_exact_words_or_markdown():
    sections = [
        {"id": "one", "text": "Editors may save three drafts.\n"},
        {"id": "two", "text": "Publication is available only to administrators."},
    ]
    assert parse_candidate(json.dumps({"sections": sections}), SOURCE) == sections


@pytest.mark.parametrize(
    "value",
    [
        {"sections": SOURCE, "extra": True},
        {"sections": list(reversed(SOURCE))},
        {"sections": SOURCE[:1]},
        {"sections": [SOURCE[0], SOURCE[0]]},
        {"sections": [SOURCE[0], {"id": "two", "text": " "}]},
        {"sections": [SOURCE[0], {"id": "two", "text": "Text.", "extra": 1}]},
        {"sections": []},
    ],
)
def test_parser_refuses_missing_reordered_or_ambiguous_sections(value):
    with pytest.raises(ValueError):
        parse_candidate(json.dumps(value), SOURCE)


def test_parser_rejects_duplicate_json_keys():
    with pytest.raises(ValueError, match="Duplicate"):
        parse_candidate('{"sections": [], "sections": []}', SOURCE)


def test_single_enclosing_json_fence_is_transport_normalization():
    candidate = json.dumps({"sections": SOURCE})
    assert parse_candidate("```json\n" + candidate + "\n```", SOURCE) == SOURCE
    response = {
        "finish_reason": "stop",
        "raw_response": "```json\n" + json.dumps(review()) + "\n```",
    }
    assert parse_review(response, ORIGINAL, REVISED)["status"] == "reject"


@pytest.mark.parametrize(
    "prefix,suffix",
    [("Here is JSON:\n", ""), ("", "\nDone."), ("", "\n```json\n{}\n```")],
)
def test_fences_do_not_permit_extra_prose_or_multiple_payloads(prefix, suffix):
    raw = prefix + "```json\n" + json.dumps({"sections": SOURCE}) + "\n```" + suffix
    with pytest.raises(ValueError):
        parse_candidate(raw, SOURCE)
    response = {"finish_reason": "stop", "raw_response": raw}
    assert parse_review(response, ORIGINAL, REVISED)["status"] == "unavailable"


def test_fences_do_not_hide_duplicate_keys():
    with pytest.raises(ValueError, match="Duplicate"):
        parse_candidate('```json\n{"sections": [], "sections": []}\n```', SOURCE)


def test_grounded_material_scope_change_rejected():
    decision = parse_review(review(), ORIGINAL, REVISED)
    assert decision["status"] == "reject"
    assert decision["issues"][0]["evidence_match"] == {
        "source": "exact",
        "candidate": "exact",
    }


def test_quote_matching_tolerates_only_whitespace_normalization():
    original = ORIGINAL.replace("Only admins", "Only       admins")
    revised = REVISED.replace("All users", "All\nusers")
    decision = parse_review(review(), original, revised)
    assert decision["status"] == "reject"
    assert decision["issues"][0]["evidence_match"] == {
        "source": "whitespace_normalized",
        "candidate": "whitespace_normalized",
    }
    bad = review(issues=[issue(source_quote="only admins can publish content.")])
    assert parse_review(bad, original, revised)["status"] == "unavailable"


def test_omission_needs_source_evidence_but_no_invented_candidate_excerpt():
    omitted = review(issues=[issue(category="omission", candidate_quote="")])
    assert parse_review(omitted, ORIGINAL, REVISED)["status"] == "reject"
    unsupported = review(issues=[issue(category="scope", candidate_quote="")])
    assert parse_review(unsupported, ORIGINAL, REVISED)["status"] == "unavailable"


@pytest.mark.parametrize(
    "value",
    [
        review(issues=[]),
        review(verdict="accept"),
        review(verdict="minor"),
        review(issues=[issue(severity="minor")]),
        review(issues=[issue(source_quote="A fabricated quotation.")]),
        review(issues=[issue(candidate_quote="A fabricated quotation.")]),
        review(issues=[issue(source_quote="")]),
        review(issues=[issue(explanation=" ")]),
        {"verdict": "accept", "issues": [], "reason": "Extra key"},
    ],
)
def test_invalid_or_ungrounded_reviews_are_unavailable_not_meaning_rejections(value):
    assert parse_review(value, ORIGINAL, REVISED)["status"] == "unavailable"


def test_transport_failure_and_truncation_are_unavailable():
    for finish in ("length", "transport_error", None):
        response = {"finish_reason": finish, "raw_response": json.dumps(review())}
        assert parse_review(response, ORIGINAL, REVISED)["status"] == "unavailable"
    response = {"finish_reason": "stop", "raw_response": json.dumps(review())}
    assert parse_review(response, ORIGINAL, REVISED)["status"] == "reject"


def test_minor_preferences_do_not_reject_an_acceptable_rewrite():
    response = review(
        verdict="minor", issues=[issue(severity="minor", category="style")]
    )
    assert parse_review(response, ORIGINAL, REVISED)["status"] == "minor"
    for verdict in ("accept", "minor"):
        assert parse_review(review(verdict, []), ORIGINAL, REVISED)["status"] == verdict


def test_mechanical_changes_to_layout_length_and_literals_only_warn():
    original = "**Headline:** We added 3 drafts at https://example.org."
    candidate = "Three drafts are available."
    result = mechanical_check(original, candidate)
    assert result["passed"]
    assert len(result["warnings"]) == 3
    assert any("numeric" in warning for warning in result["warnings"])
    assert any("URL" in warning for warning in result["warnings"])
    assert mechanical_check("**Title:** normal prose.", "Title\n\nNormal prose.")[
        "passed"
    ]


@pytest.mark.parametrize(
    "char", ["\x00", "\u200b", "\u202e", "\u00a0", "\u034f", "\u3164"]
)
def test_introduced_hidden_or_unusual_characters_blocked(char):
    assert not mechanical_check("The text.", "The " + char + "text.")["passed"]
    assert mechanical_check("The " + char + "text.", "The " + char + "text.")["passed"]
    assert not mechanical_check("The " + char + "text.", "The " + char * 2 + "text.")[
        "passed"
    ]


def test_empty_and_invalid_unicode_blocked_but_readable_unicode_supported():
    assert not mechanical_check("Text.", " ")["passed"]
    assert not mechanical_check("Text.", "Text\ud800")["passed"]
    assert mechanical_check("The café may open.", "The café may be open.")["passed"]


@pytest.mark.parametrize(
    "statuses", [("accept", "accept"), ("accept", "minor"), ("minor", "minor")]
)
def test_two_positive_reviews_pass_without_adjudication(statuses):
    result = resolve_reviews([{"status": status} for status in statuses])
    assert result["accepted"]
    assert not result["needs_adjudication"]


def test_two_material_rejections_fail_without_adjudication():
    result = resolve_reviews([{"status": "reject"}, {"status": "reject"}])
    assert not result["accepted"]
    assert not result["needs_adjudication"]
    assert result["status"] == "reject"


@pytest.mark.parametrize(
    "statuses",
    [("accept", "reject"), ("unavailable", "reject"), ("unavailable", "unavailable")],
)
def test_conflict_and_unavailable_reviews_require_valid_adjudication(statuses):
    initial = [{"status": status} for status in statuses]
    unresolved = resolve_reviews(initial)
    assert unresolved["needs_adjudication"] and not unresolved["accepted"]
    for verdict in ("accept", "minor", "reject"):
        settled = resolve_reviews(initial, {"status": verdict})
        assert settled["accepted"] == (verdict != "reject")
        assert not settled["needs_adjudication"]
    unavailable = resolve_reviews(initial, {"status": "unavailable"})
    assert unavailable["needs_adjudication"] and not unavailable["accepted"]


def test_resolver_requires_two_initial_reviews():
    with pytest.raises(ValueError, match="Exactly two"):
        resolve_reviews([{"status": "accept"}])
