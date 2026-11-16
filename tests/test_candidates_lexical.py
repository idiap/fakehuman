# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: William Droz <william.droz@idiap.ch>
# SPDX-License-Identifier: GPL-3.0-only

import random

import pytest

from fakehuman.candidates_lexical import (
    _PAIRS,
    RULE_LABELS,
    RULES,
)
from fakehuman.rules import quality_issues


def test_candidate_catalog_has_distinct_named_rules_and_labels():
    names = [name for name, _, _ in _PAIRS]
    assert 75 <= len(names) <= 90
    assert len(names) == len(set(names))
    assert len(RULES) == len(RULE_LABELS) == len(names) + 1
    assert set(RULES) == set(RULE_LABELS)
    assert all(name.startswith("lex_") for name in RULES)


@pytest.mark.parametrize("name,source,target", _PAIRS)
def test_each_rule_replaces_only_its_exact_phrase(name, source, target):
    contexts = {
        "due to": ("We paused ", " the delay."),
        "for this reason": ("", ", the sample stayed here."),
        "in other words": ("", ", the sample stayed here."),
        "put differently": ("", ", the sample stayed here."),
        "that is to say": ("", ", the sample stayed here."),
        "as an example": ("", ", the sample stayed here."),
        "all the same": ("", ", the sample stayed here."),
        "even so": ("", ", the sample stayed here."),
        "as a result": ("", ", the sample stayed here."),
        "in addition": ("", ", the sample stayed here."),
        "in contrast": ("", ", the sample stayed here."),
        "by contrast": ("", ", the sample stayed here."),
        "in particular": ("", ", the sample stayed here."),
        "at least": ("Allow ", " three days."),
        "a short": ("We wrote ", " report."),
        "a brief": ("We wrote ", " report."),
        "at the same time": ("Both tests ran ", "."),
        "at the end of": ("We stopped ", " the trial."),
        "carry out": ("We ", " tests."),
        "carries out": ("The team ", " tests."),
        "carried out": ("The team ", " tests."),
    }
    before, after = contexts.get(source, ("The ", " results remain."))
    sentence = before + source + after
    output = RULES[name](sentence, random.Random(7))
    assert output == before + target + after
    assert RULES[name](source + "x", random.Random(7)) == source + "x"


def test_case_is_preserved_for_lowercase_and_sentence_initial_phrase():
    rule = RULES["lex_due_to"]
    assert rule("due to the protocol.", random.Random(0)) == "because of the protocol."
    assert rule("Due to the protocol.", random.Random(0)) == "Because of the protocol."
    assert rule("DUE TO the protocol.", random.Random(0)) == "BECAUSE OF the protocol."


def test_sentence_initial_transition_skips_internal_and_protected_text():
    rule = RULES["lex_initial_however_nevertheless"]
    text = "However, the result is limited. A limitation, however, remains. `However,`"
    assert rule(text, random.Random(0)) == (
        "Nevertheless, the result is limited. A limitation, however, remains. `However,`"
    )


def test_protected_citations_links_code_and_bracketed_text_are_unchanged():
    text = (
        "due to the result (Smith et al., 2020), https://example.org/due-to, "
        "`due to`, [due to], and doi:10.1234/due-to."
    )
    assert RULES["lex_due_to"](text, random.Random(0)) == (
        "because of the result (Smith et al., 2020), https://example.org/due-to, "
        "`due to`, [due to], and doi:10.1234/due-to."
    )


def test_causal_and_compositional_forms_keep_basic_grammar():
    assert (
        RULES["lex_gives_rise_to"](
            "The signal gives rise to a response.", random.Random(0)
        )
        == "The signal causes a response."
    )
    assert (
        RULES["lex_resulted_from"](
            "The outcome resulted from the update.", random.Random(0)
        )
        == "The outcome arose from the update."
    )
    assert (
        RULES["lex_is_composed_of"](
            "The assembly is composed of three sections.", random.Random(0)
        )
        == "The assembly consists of three sections."
    )
    assert (
        RULES["lex_are_composed_of"](
            "The assemblies are composed of three sections.", random.Random(0)
        )
        == "The assemblies consist of three sections."
    )


def test_frequent_wording_changes_preserve_articles_and_count_language():
    assert (
        RULES["lex_an_important_limitation_key"](
            "This is an important limitation.", random.Random(0)
        )
        == "This is a key limitation."
    )
    assert (
        RULES["lex_the_same_idea_identical"](
            "It communicated the same idea.", random.Random(0)
        )
        == "It communicated an identical idea."
    )
    assert (
        RULES["lex_at_least_minimum"]("Allow at least three days.", random.Random(0))
        == "Allow a minimum of three days."
    )


def test_changes_preserve_technical_values_symbols_citations_and_hedges():
    text = "Due to v53 output, the 2.5 µV signal may result in a response (Smith et al., 2020)."
    output = RULES["lex_due_to"](text, random.Random(0))
    assert (
        output
        == "Because of v53 output, the 2.5 µV signal may result in a response (Smith et al., 2020)."
    )
    assert quality_issues(text, output) == []


@pytest.mark.parametrize(
    ("name", "text"),
    [
        ("lex_due_to", "The train is due to arrive at noon."),
        ("lex_due_to", "The update is due to be released tomorrow."),
        ("lex_due_to", "The payment is due to the bank."),
        ("lex_due_to", "We paid the respect due to her."),
        ("lex_in_addition", "We need a cable in addition to the adapter."),
        ("lex_as_a_result_therefore", "The test failed as a result of noise."),
        ("lex_as_a_result_therefore", "They listed the measurement as a result."),
        ("lex_in_addition", "The exercise tests skill in addition."),
        ("lex_in_particular", "Did you have anything in particular in mind?"),
        ("lex_in_contrast", "In contrast to the first group, this group passed."),
        ("lex_in_advance_beforehand", "We arrived in advance of the meeting."),
        ("lex_all_the_same", "The samples are all the same size."),
        ("lex_even_so", "We could not detect even so small a change."),
        ("lex_as_an_example", "Use the diagram as an example."),
        ("lex_in_other_words", "Describe the result in other words."),
        ("lex_put_differently", "The question was put differently."),
        ("lex_a_short_brief", "A short circuit stopped the test."),
        ("lex_a_short_brief", "We saw a short person near the door."),
        ("lex_a_brief_short", "They filed a brief with the court."),
        ("lex_at_least_minimum", "At least we tried."),
        ("lex_at_the_same_time", "They arrived at the same time as us."),
        ("lex_at_the_end_of", "The label is at the end of the cable."),
        ("lex_carry_out", "Please carry out the box."),
        ("lex_carried_out", "They carried out the injured worker."),
    ],
)
def test_ambiguous_senses_and_incompatible_complements_are_unchanged(name, text):
    assert RULES[name](text, random.Random(7)) == text


@pytest.mark.parametrize(
    "text",
    [
        "They consist of two parts.",
        "It may consist of two parts.",
        "It does not consist of two parts.",
        "It needs to consist of two parts.",
    ],
)
def test_consist_replacement_keeps_finite_or_infinitive_grammar(text):
    assert RULES["lex_consist_of"](text, random.Random(7)) == text.replace(
        "consist of", "comprise"
    )


def test_initial_transition_keeps_capitalization_after_sentence_boundary():
    assert (
        RULES["lex_initial_however_nevertheless"](
            "The test ended. However, work continued.", random.Random(7)
        )
        == "The test ended. Nevertheless, work continued."
    )
