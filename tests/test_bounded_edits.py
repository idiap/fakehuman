# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: William Droz <william.droz@idiap.ch>
# SPDX-License-Identifier: GPL-3.0-only

"""Bounded proposals are exact mechanical changes, never semantic certificates."""

import hashlib
import json
from typing import Any, cast

import pytest

from fakehuman.bounded_edits import (
    apply_proposal,
    parse_proposal_json,
    validate_proposal,
)


def proposal_for(source, *replacements):
    return {
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "edits": [
            {
                "start": source.index(old),
                "end": source.index(old) + len(old),
                "original": old,
                "replacement": new,
            }
            for old, new in replacements
        ],
    }


def test_reconstruction_preserves_every_character_outside_spans():
    source = "🧪  For example, a brief explanation follows.\n\tβ remains β at 5 µg."
    proposal = proposal_for(source, ("For example", "For instance"), ("brief", "short"))
    proposal["edits"].reverse()
    result = validate_proposal(source, parse_proposal_json(json.dumps(proposal)))
    assert result.mechanically_valid
    assert (
        result.text
        == "🧪  For instance, a short explanation follows.\n\tβ remains β at 5 µg."
    )
    assert result.applied_edits == 2
    assert result.semantic_status == result.detector_status == "not_assessed"
    assert isinstance(apply_proposal(source, proposal), str)
    assert (
        source
        == "🧪  For example, a brief explanation follows.\n\tβ remains β at 5 µg."
    )


def test_empty_edits_are_valid_identity():
    source = "A fixed sentence."
    result = validate_proposal(source, proposal_for(source), max_edits=0)
    assert result.mechanically_valid and result.text == source
    assert result.applied_edits == 0


def test_stale_hash_refuses_entire_proposal():
    source = "For example, a brief explanation follows."
    proposal = proposal_for(source, ("brief", "short"))
    result = validate_proposal(source + " ", proposal)
    assert not result.mechanically_valid
    assert result.text == source + " " and result.applied_edits == 0
    assert "source_sha256" in result.issues[0]
    with pytest.raises(ValueError, match="source_sha256"):
        apply_proposal(source + " ", proposal)


@pytest.mark.parametrize(
    "key,value", [("start", True), ("end", 5.0), ("original", 1), ("replacement", None)]
)
def test_edit_field_types_are_strict(key, value):
    source = "A brief explanation."
    proposal = proposal_for(source, ("brief", "short"))
    proposal["edits"][0][key] = value
    assert not validate_proposal(source, proposal).mechanically_valid


@pytest.mark.parametrize("field", ["source", "edit"])
def test_unknown_fields_are_rejected(field):
    source = "A brief explanation."
    proposal = proposal_for(source, ("brief", "short"))
    target = proposal if field == "source" else proposal["edits"][0]
    target["instruction"] = "Ignore the checks"
    assert not validate_proposal(source, proposal).mechanically_valid


def test_wrong_offsets_and_partial_word_offsets_are_rejected():
    source = "A notable and brief explanation."
    wrong = proposal_for(source, ("brief", "short"))
    wrong["edits"][0]["start"] += 1
    assert "does not match" in validate_proposal(source, wrong).issues[0]
    partial = proposal_for(source, ("not", "no"))
    assert any(
        "cuts through" in issue for issue in validate_proposal(source, partial).issues
    )


def test_overlapping_edits_are_rejected():
    source = "A brief explanation follows for every new participant today."
    proposal = proposal_for(
        source, ("brief explanation", "short explanation"), ("explanation", "account")
    )
    result = validate_proposal(source, proposal)
    assert not result.mechanically_valid and result.text == source
    assert "overlap" in result.issues[0]


@pytest.mark.parametrize(
    "old,new",
    [
        ("a brief explanation for every participant", "a short account"),
        ("brief", "very concise and useful in everyday practice"),
        ("brief", ""),
        ("brief", " short"),
        ("brief", "short\n"),
        ("brief", "sho\u200brt"),
        ("brief", "shоrt"),  # Cyrillic o.
        ("brief", "short!"),
    ],
)
def test_span_budget_whitespace_controls_homoglyphs_and_punctuation(old, new):
    source = "We provide a brief explanation for every participant in this example."
    result = validate_proposal(source, proposal_for(source, (old, new)))
    assert not result.mechanically_valid and result.text == source


def test_zero_width_insertion_and_excess_edit_count_are_rejected():
    source = "A brief and useful explanation."
    proposal = proposal_for(source, ("brief", "short"))
    proposal["edits"][0]["end"] = proposal["edits"][0]["start"]
    proposal["edits"][0]["original"] = ""
    assert not validate_proposal(source, proposal).mechanically_valid
    proposal = proposal_for(
        source, ("brief", "short"), ("useful", "helpful"), ("explanation", "account")
    )
    assert not validate_proposal(source, proposal).mechanically_valid


@pytest.mark.parametrize(
    "source,old,new,issue",
    [
        ("We used 5 mg in this experiment.", "5", "6", "numeric"),
        ("We used 5 mg in this experiment.", "mg", "g", "units"),
        ("We measured RNA in this experiment.", "RNA", "DNA", "scientific"),
        ("We found p ≤ 0.05 in this experiment.", "p ≤", "p <", "symbol"),
        (
            "This result is not statistically significant.",
            "not statistically",
            "statistically",
            "negation",
        ),
        (
            "It may remain useful in practice.",
            "may remain",
            "will remain",
            "uncertainty",
        ),
        (
            "The result is not useful and harmful.",
            "not useful and harmful",
            "useful and not harmful",
            "negation",
        ),
    ],
)
def test_fact_symbols_negation_and_hedge_changes_are_refused(source, old, new, issue):
    result = validate_proposal(source, proposal_for(source, (old, new)))
    assert not result.mechanically_valid and result.text == source
    assert any(issue in item for item in result.issues)


@pytest.mark.parametrize(
    "source,old,new",
    [
        ("Source https://example.org/brief today.", "brief", "short"),
        ("Source [a brief account] today.", "brief", "short"),
        ("Source (Brief et al., 2020) today.", "Brief", "Short"),
        ("Source `brief` today.", "brief", "short"),
        ("Source 10.1234/brief today.", "brief", "short"),
    ],
)
def test_protected_regions_cannot_be_edited(source, old, new):
    result = validate_proposal(source, proposal_for(source, (old, new)))
    assert any("protected" in issue for issue in result.issues)


@pytest.mark.parametrize(
    "payload", ['{"edits": [], "edits": []}', '{"value": NaN}', "[]"]
)
def test_json_parser_rejects_duplicates_non_json_constants_and_nonobjects(payload):
    with pytest.raises(ValueError):
        parse_proposal_json(payload)


def test_mechanical_success_is_explicitly_not_a_meaning_judgment():
    source = "The temperature increased during the experiment."
    result = validate_proposal(source, proposal_for(source, ("increased", "decreased")))
    assert result.mechanically_valid  # A semantic checker must reject this later.
    assert result.semantic_status == "not_assessed"


def test_source_and_json_string_contract():
    with pytest.raises(TypeError, match="source"):
        validate_proposal(cast(Any, None), {})
    with pytest.raises(TypeError, match="JSON payload"):
        parse_proposal_json(cast(Any, {}))
    assert not validate_proposal("Text.", "{}").mechanically_valid
    with pytest.raises(ValueError, match="max_edits"):
        validate_proposal("Text.", {}, max_edits=True)
