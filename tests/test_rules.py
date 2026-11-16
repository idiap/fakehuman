# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: William Droz <william.droz@idiap.ch>
# SPDX-License-Identifier: GPL-3.0-only

"""Mechanical preservation and reproducibility for text transformations."""

import random
from typing import Any, cast

import pytest

from fakehuman.rules import (
    RULE_LABELS,
    RULES,
    apply_rules,
    quality_issues,
)


def test_all_named_rules_have_readable_labels_and_string_results():
    assert RULE_LABELS.keys() == RULES.keys()
    assert len(RULES) >= 10
    for name, rule in RULES.items():
        assert isinstance(rule("A small test.", random.Random(42)), str), name


def test_pipeline_is_deterministic_and_independent_of_batch_order():
    texts = [
        "Additionally, numerous helpful examples are available. However, check them.",
        "Furthermore, numerous results are helpful. Therefore, compare them.",
    ]
    pipeline = ["seeded_synonyms", "simplify_verbs"]
    forwards = {text: apply_rules(text, pipeline, seed=19) for text in texts}
    backwards = {text: apply_rules(text, pipeline, seed=19) for text in reversed(texts)}
    assert forwards == backwards
    assert forwards == {text: apply_rules(text, pipeline, seed=19) for text in texts}
    assert len({apply_rules(texts[0], pipeline, seed=seed) for seed in range(20)}) > 1


def test_random_choices_keep_same_stream_when_earlier_rule_changes_text():
    text = "Additionally, numerous examples utilize helpful notes. However, check them."
    synonyms_only = apply_rules(text, ["seeded_synonyms"], seed=7)
    with_preceding_rule = apply_rules(
        text, ["simplify_verbs", "seeded_synonyms"], seed=7
    )
    assert with_preceding_rule == synonyms_only.replace("utilize", "use")


def test_technical_values_symbols_and_negative_claim_survive_all_rules():
    text = (
        "Moreover, we utilize 5 µm of AX-6 at 37 °C and p ≤ 0.05. "
        "The mean was 2.5 ± 0.4 μV; these results are not statistically significant. "
        "The 10–20 mm range and −3.2 mV signal might require further review."
    )
    result = apply_rules(text, list(RULES))
    assert "5 µm" in result
    assert "AX-6" in result
    assert "37 °C" in result
    assert "p ≤ 0.05" in result
    assert "2.5 ± 0.4 μV" in result
    assert "10–20 mm" in result
    assert "−3.2 mV" in result
    assert "not statistically significant" in result
    assert quality_issues(text, result) == []


def test_only_prose_em_dashes_are_replaced():
    text = "The result—an estimate—is useful at 5—10 mm with −2 °C and β-modes."
    result = apply_rules(text, ["normalize_dashes"])
    assert (
        result
        == "The result, an estimate, is useful at 5—10 mm with −2 °C and β-modes."
    )
    assert quality_issues(text, result) == []


def test_capitalization_leaves_technical_identifiers_intact():
    text = "p ≤ 0.05 for eBook. Moreover, v53 was measured; eBook remained stable."
    result = apply_rules(
        text, ["trim_openers", "remove_stock_framing", "split_semicolons"]
    )
    assert result == "p ≤ 0.05 for eBook. v53 was measured. eBook remained stable."
    assert quality_issues(text, result) == []


def test_citations_urls_dois_and_code_are_preserved_byte_for_byte():
    protected = (
        "[Furthermore, 1–3] (Moreover et al., 2024) "
        "https://example.org/utilize?value=5 "
        "10.1234/utilize.2024 `utilize—helpful`"
    )
    text = "Moreover, we utilize helpful examples. Sources: " + protected
    result = apply_rules(text, list(RULES))
    assert protected in result
    assert quality_issues(text, result) == []


@pytest.mark.parametrize(
    ("original", "transformed", "issue"),
    [
        ("The result is 2.5 mm.", "The result is 25 mm.", "numeric values"),
        ("The result is 2.5 mm.", "The result is 2.5 m.", "measurement units"),
        ("The result is 2.5 mm.", "The result is 2.5.", "measurement units"),
        ("The result is ≤ 5 mm.", "The result is 5 mm.", "scientific symbols"),
        ("The mean is 3 ± 1 μV.", "The mean is 3 + 1 μV.", "scientific symbols"),
        ("The result is not significant.", "The result is significant.", "negation"),
        ("There was no evidence.", "There was evidence.", "negation"),
        ("The result might be useful.", "The result is useful.", "uncertainty"),
        (
            "We measured AX-6 output.",
            "We measured AX-8 output.",
            "scientific tokens",
        ),
        (
            "We measured TCP output.",
            "We measured UDP output.",
            "scientific tokens",
        ),
        (
            "We measured v53 output.",
            "We measured v54 output.",
            "scientific tokens",
        ),
        (
            "We measured eBook output.",
            "We measured TCP output.",
            "scientific tokens",
        ),
        ("Results are reported in [1].", "Results are reported in [2].", "citation"),
        ("Source: https://example.org/a", "Source: https://example.org/b", "URL"),
        ("Source: (Smith, 2024).", "Source: (Jones, 2024).", "citation"),
        ("An intact original statement.", "", "empty output"),
        (
            "The full explanation contains several relevant details.",
            "Details.",
            "excessive removal",
        ),
    ],
)
def test_quality_screen_flags_mechanical_damage(original, transformed, issue):
    assert any(issue in found for found in quality_issues(original, transformed))


@pytest.mark.parametrize(
    "text",
    [
        "The result is not significant and we cannot infer a causal effect.",
        "We do not know whether the samples are not representative.",
        "They will not replicate and the samples have not changed.",
        "We could not conclude that the values would not change.",
    ],
)
def test_contractions_preserve_explicit_negation_and_uncertainty(text):
    contracted = apply_rules(text, ["contractions"])
    assert contracted != text
    assert quality_issues(text, contracted) == []
    expanded = apply_rules(contracted, ["expand_contractions"])
    assert expanded == text


def test_unknown_rule_fails_clearly_and_empty_pipeline_is_identity():
    assert apply_rules("  Text remains.  ", []) == "  Text remains.  "
    with pytest.raises(ValueError, match="Unknown rule.*invented"):
        apply_rules("Text", ["invented"])
    with pytest.raises(TypeError, match="string"):
        apply_rules(cast(Any, None), [])


def test_quality_screen_is_explicitly_not_semantic_proof():
    # Identical counts do not establish that a rewritten technical claim is true.
    original = "Compression increases throughput in routers."
    opposite = "Compression decreases throughput in routers."
    assert quality_issues(original, opposite) == []


NEW_RULE_EXAMPLES = {
    "affirmative_contractions": "We are ready to begin and it is helpful to plan.",
    "auxiliary_have_contractions": "We have reviewed the notes and they have recorded the details.",
    "common_plain_verbs": "The activity requires a plan and provides a useful example.",
    "example_alternatives": "We can review a short note, for example, before the meeting.",
    "contrast_transitions": "However, the notes may omit a detail that needs further review.",
    "result_transitions": "Therefore, the group should review the available notes before deciding.",
    "two_sentence_paragraphs": "The notes are ready. The group can read them. The review starts tomorrow.",
    "ability_phrases": "We are able to review the records and discuss the limitations.",
    "recommendation_phrases": "My recommendation is to review the notes before the group meets.",
    "shorter_time_phrases": "We should distribute the notes in advance and discuss them together.",
    "direct_nominal_phrases": "We should keep a record of the points discussed during the meeting.",
}


@pytest.mark.parametrize(("rule", "example"), NEW_RULE_EXAMPLES.items())
def test_additional_rules_change_prose_and_preserve_protected_content(rule, example):
    suffix = (
        " No result is claimed for 5 mm at 37 °C with p ≤ 0.05 or AX-6. "
        "See [We are, 1–3] (Example et al., 2024), "
        "https://example.org/require and `for example`."
    )
    original = example + suffix
    transformed = apply_rules(original, [rule])
    assert transformed != original
    assert quality_issues(original, transformed) == []
    for protected in (
        "5 mm",
        "37 °C",
        "p ≤ 0.05",
        "AX-6",
        "[We are, 1–3]",
        "(Example et al., 2024)",
        "https://example.org/require",
        "`for example`",
    ):
        assert protected in transformed


def test_affirmative_contractions_leave_terminal_and_explanatory_auxiliaries():
    original = "We know what it is. That is, we know where we are. It is not a result."
    transformed = apply_rules(original, ["affirmative_contractions"])
    assert (
        transformed
        == "We know what it is. That is, we know where we are. It's not a result."
    )
    assert quality_issues(original, transformed) == []


def test_auxiliary_have_contractions_leave_possession_unchanged():
    original = "We have samples, they have time, and I have questions. We have reviewed the notes."
    transformed = apply_rules(original, ["auxiliary_have_contractions"])
    assert (
        transformed
        == "We have samples, they have time, and I have questions. We've reviewed the notes."
    )
    assert quality_issues(original, transformed) == []


def test_plain_verbs_keep_conditional_provided_and_passive_grammar():
    original = (
        "We provided notes. Samples were provided with labels. "
        "Proceed provided that approval is given. The rule provides for review."
    )
    transformed = apply_rules(original, ["common_plain_verbs"])
    assert transformed == original.replace("We provided", "We gave")
    assert quality_issues(original, transformed) == []


def test_transition_rules_do_not_rewrite_other_grammatical_roles():
    original = (
        "However carefully we work, errors remain. We therefore check every note."
    )
    assert (
        apply_rules(original, ["contrast_transitions", "result_transitions"])
        == original
    )
    assert (
        apply_rules(
            "However, errors remain. Therefore, we check notes.",
            ["contrast_transitions", "result_transitions"],
        )
        == "Still, errors remain. So we check notes."
    )


def test_paragraphs_change_only_whitespace_and_skip_abbreviations():
    original = "Dr. Smith reviewed 2.5 mm. The notes were clear. Prof. Jones read them. We waited."
    transformed = apply_rules(original, ["two_sentence_paragraphs"])
    assert "Dr. Smith" in transformed
    assert "Prof. Jones" in transformed
    assert "clear.\n\nProf." in transformed
    assert " ".join(original.split()) == " ".join(transformed.split())
    assert quality_issues(original, transformed) == []


def test_ability_phrases_preserve_qualified_and_past_ability():
    original = (
        "They might be able to help. They were able to help. They are not able to help. "
        "They may have the ability to help. They will have the ability to help. "
        "They want to have the ability to help. They don't have the ability to help."
    )
    assert apply_rules(original, ["ability_phrases"]) == original


def test_shorter_time_phrases_preserve_preposition_grammar():
    original = "Send notes in advance of the meeting, or prepare them in advance."
    transformed = apply_rules(original, ["shorter_time_phrases"])
    assert transformed == "Send notes before the meeting, or prepare them beforehand."
    assert quality_issues(original, transformed) == []


def test_broader_seeded_synonyms_are_paired_and_grammatical():
    original = (
        "An important step is to obtain feedback and ensure that a useful example "
        "is included. A brief note can provide additional information about a routine task."
    )
    outputs = [
        apply_rules(original, ["broader_seeded_synonyms"], seed=seed)
        for seed in range(12)
    ]
    assert len(set(outputs)) > 1
    assert outputs == [
        apply_rules(original, ["broader_seeded_synonyms"], seed=seed)
        for seed in range(12)
    ]
    assert any(text != original for text in outputs)
    assert all("an key" not in text.lower() for text in outputs)
    assert all(quality_issues(original, text) == [] for text in outputs)
    technical = (
        "Use simple random sampling for the classification task and a clear solution."
    )
    assert apply_rules(technical, ["broader_seeded_synonyms"]) == technical
