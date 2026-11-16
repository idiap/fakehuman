# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: William Droz <william.droz@idiap.ch>
# SPDX-License-Identifier: GPL-3.0-only

import random

import pytest

from fakehuman.candidates_structural import RULE_LABELS, RULES
from fakehuman.rules import quality_issues


def apply(name: str, text: str) -> str:
    return RULES[name](text, random.Random(7))


def test_registry_is_prefixed_and_labeled():
    assert len(RULES) == 25
    assert set(RULES) == set(RULE_LABELS)
    assert all(name.startswith("syn_") for name in RULES)


def test_abbreviation_replacements_work_with_terminal_punctuation():
    assert apply("syn_e_g_expansion", "Use e.g. this sample.") == (
        "Use for example this sample."
    )
    assert apply("syn_i_e_expansion", "This is i.e. the control.") == (
        "This is that is the control."
    )
    assert apply("syn_viz_expansion", "The set, viz. A and B, was tested.") == (
        "The set, namely A and B, was tested."
    )


def test_causal_and_conditional_phrases_keep_the_complement():
    assert apply("syn_due_to_equivalent", "The signal changed due to noise.") == (
        "The signal changed because of noise."
    )
    assert apply(
        "syn_as_long_as_to_provided_that", "It works as long as input is valid."
    ) == ("It works provided that input is valid.")
    assert apply(
        "syn_provided_that_to_as_long_as", "Provided that x is true, y follows."
    ) == ("As long as x is true, y follows.")
    assert apply(
        "syn_provided_that_to_as_long_as", "The method provided that result."
    ) == ("The method provided that result.")


def test_structural_edits_preserve_parenthetical_words_and_numbers():
    original = "The result (at 37 °C) was measured at 5 mV/ms."
    changed = apply("syn_parenthetical_to_dashes", original)
    assert changed == "The result — at 37 °C — was measured at 5 mV/ms."
    assert quality_issues(original, changed) == []
    assert apply("syn_parenthetical_to_commas", "The result (briefly) changed.") == (
        "The result, briefly, changed."
    )


def test_citations_urls_and_inline_code_are_protected():
    text = "See (Smith et al., 2024), https://example.org/a, and `i.e.` for details."
    assert apply("syn_i_e_expansion", text) == text
    assert apply("syn_parenthetical_to_dashes", text) == text


def test_technical_identifiers_and_measurements_are_preserved():
    original = "Roughly 2.5 µV on v53 was tested at 10°C; see Fig. 2."
    changed = apply("syn_roughly_to_about", original)
    assert changed == "About 2.5 µV on v53 was tested at 10°C; see Fig. 2."
    assert quality_issues(original, changed) == []


def test_sentence_initial_variants_require_comma_and_sentence_boundary():
    rule = RULES["syn_sentence_initial_in_contrast"]
    assert rule("In contrast, growth was stable.", random.Random(1)) == (
        "By contrast, growth was stable."
    )
    assert rule("The in contrast phrase is not initial.", random.Random(1)) == (
        "The in contrast phrase is not initial."
    )
    assert rule("An acronym. In contrast, growth was stable.", random.Random(1)) == (
        "An acronym. By contrast, growth was stable."
    )


def test_every_rule_is_a_safe_noop_on_empty_and_unmatched_text():
    for name, rule in RULES.items():
        assert rule("", random.Random(11)) == "", name
        assert rule("v53 was measured at 3 mV.", random.Random(11)) == (
            "v53 was measured at 3 mV."
        ), name


@pytest.mark.parametrize(
    ("name", "text"),
    [
        ("syn_due_to_equivalent", "The train is due to arrive soon."),
        ("syn_roughly_to_about", "The parcel was roughly handled."),
        ("syn_roughly_to_about", "They spoke roughly to the visitor."),
        ("syn_as_long_as_to_provided_that", "The cable is as long as the pipe."),
        ("syn_as_long_as_to_provided_that", "It takes as long as three hours."),
        ("syn_as_long_as_to_provided_that", "You can stay as long as you want."),
        ("syn_provided_that_to_as_long_as", "Provided that report to the team."),
        ("syn_along_with_equivalent", "They get along with their colleagues."),
        ("syn_along_with_equivalent", "We play along with the joke."),
        ("syn_along_with_equivalent", "They went along with the plan."),
        ("syn_comma_parenthetical_to_dashes", "We used red, green, blue and white."),
        ("syn_comma_parenthetical_to_dashes", "First, mix thoroughly, then wait."),
    ],
)
def test_ambiguous_structural_replacements_are_skipped(name, text):
    assert apply(name, text) == text


@pytest.mark.parametrize(
    "name", ["syn_parenthetical_to_commas", "syn_parenthetical_to_dashes"]
)
@pytest.mark.parametrize(
    "text",
    [
        "The result changed (briefly).",
        "(Briefly) the result changed.",
        "The range (x + y) was measured.",
        "The voltage (mV) was measured.",
        "The abbreviation (ABC) was defined.",
        "The result (an aside (still warm) inside) changed.",
        "The sample (red, green and blue) arrived.",
        "The result\n(briefly) changed.",
    ],
)
def test_parenthetical_edits_skip_notation_nesting_and_punctuation_hazards(name, text):
    assert apply(name, text) == text


def test_prose_aside_edits_keep_readable_spacing():
    assert (
        apply("syn_comma_parenthetical_to_dashes", "The sample, still warm, arrived.")
        == "The sample — still warm — arrived."
    )


def test_more_specifically_keeps_comparative_qualification():
    assert (
        apply(
            "syn_more_specifically_to_specifically",
            "More specifically, check the seal.",
        )
        == "More precisely, check the seal."
    )
