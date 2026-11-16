# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: William Droz <william.droz@idiap.ch>
# SPDX-License-Identifier: GPL-3.0-only

"""Regression cases from the rule audit, independent of detector model scores."""

import pytest

from fakehuman.rules import RULES, apply_rules, quality_issues


@pytest.mark.parametrize(
    "protected",
    [
        '```python\nutilize = "helpful"\n# Moreover, we are ready.\n```',
        "~~~~text\nFurthermore, we utilize helpful examples.\n~~~~",
        "``a `literal` with\nutilize and helpful``",
        "    Furthermore, we utilize helpful examples.\n    It is not code to edit.",
        "  \tFurthermore, we utilize helpful examples.",
        "> Furthermore, we utilize helpful examples.",
        "> ```text\n> Furthermore, we utilize helpful examples.\n> ```",
        '[utilize](../utilize/helpful.md "We are ready")',
        "[utilize][helpful]",
        '[utilize]: ../utilize/helpful.md "We are ready"',
        '<a title="utilize > helpful" data-note="we are ready">',
        "<code>Furthermore, we utilize helpful examples.</code>",
        "<pre>We are ready.\n    Utilize helpful examples.</pre>",
        '<script>const helpful = "We are ready";</script>',
        "<!-- Furthermore, we utilize helpful examples. -->",
        "utilize.helpful@example.org",
        "https://example.org/utilize?value=helpful",
        '"Furthermore, we utilize helpful examples."',
        "“Furthermore, we utilize helpful examples.”",
        "'Furthermore, we utilize helpful examples.'",
        "‘Furthermore, we utilize helpful examples.’",
    ],
)
def test_core_rules_preserve_literals_markup_and_quoted_wording(protected):
    text = (
        "Furthermore, we utilize helpful examples and provide context for the report.\n"
        + protected
        + "\nWe utilize helpful examples and retain the complete explanation."
    )
    rules = [name for name in RULES if name != "normalize_quotes"]
    result = apply_rules(text, rules)
    assert result != text
    assert protected in result
    assert quality_issues(text, result) == []


@pytest.mark.parametrize(
    "text",
    [
        "```text\nFurthermore, we utilize helpful examples.",
        "~~~text\nFurthermore, we utilize helpful examples.",
        "> ```text\n> Furthermore, we utilize helpful examples.",
        "<code>Furthermore, we utilize helpful examples.",
        "<!-- Furthermore, we utilize helpful examples.",
    ],
)
def test_unclosed_code_and_comments_are_conservatively_opaque(text):
    assert apply_rules(text, list(RULES)) == text


def test_fence_closers_must_match_the_opening_delimiter():
    text = "````text\nutilize\n```\nhelpful\n````\nWe utilize examples."
    assert apply_rules(text, ["simplify_verbs"]) == text.replace("We utilize", "We use")
    mismatched = "```text\nutilize\n```~\nWe utilize examples."
    assert apply_rules(mismatched, ["simplify_verbs"]) == mismatched


def test_quote_normalization_is_allowed_without_editing_quoted_words_or_code():
    original = "The form reads “complete” when it’s ready. `“utilize”`"
    normalized = 'The form reads "complete" when it\'s ready. `“utilize”`'
    assert apply_rules(original, ["normalize_quotes"]) == normalized
    assert quality_issues(original, normalized) == []
    changed_word = original.replace("complete", "incomplete")
    assert any(
        "quoted wording" in issue for issue in quality_issues(original, changed_word)
    )
    assert apply_rules("She said “utilize the tool.”", ["simplify_verbs"]) == (
        "She said “utilize the tool.”"
    )


@pytest.mark.parametrize(
    ("rule", "text"),
    [
        ("trim_openers", "[1]Moreover, the records are complete."),
        (
            "remove_stock_framing",
            "[1]It is worth noting that the records are complete.",
        ),
        ("contrast_transitions", "[1]However, the records are complete."),
        ("result_transitions", "[1]Therefore, the records are complete."),
        ("strip_heading_markup", "[1]## Results"),
    ],
)
def test_protected_span_end_does_not_manufacture_a_sentence_start(rule, text):
    assert apply_rules(text, [rule]) == text


def test_sentence_start_after_a_code_block_remains_a_real_boundary():
    original = "```text\nutilize\n```\nFurthermore, the records are complete."
    assert apply_rules(original, ["trim_openers"]) == original.replace(
        "Furthermore, the", "The"
    )


def test_space_normalization_preserves_indentation_tabs_and_markdown_hard_breaks():
    original = "  Please  read this.  \nNext\t\tcolumn.\n    utilize  tools.\n"
    expected = "  Please read this.  \nNext\t\tcolumn.\n    utilize  tools.\n"
    assert apply_rules(original, ["normalize_spaces"]) == expected
    assert quality_issues(original, expected) == []


@pytest.mark.parametrize(
    ("rule", "text"),
    [
        ("normalize_dashes", "The report—\n\nwhich we read—is useful."),
        ("trim_openers", "Moreover,\n\nthe records are complete."),
        (
            "remove_stock_framing",
            "It is worth noting that\n\nthe records are complete.",
        ),
        ("contrast_transitions", "However,\n\nthe records are complete."),
        ("result_transitions", "Therefore,\n\nthe records are complete."),
        ("split_semicolons", "The first test ended;\n\nanother began."),
    ],
)
def test_local_replacements_do_not_consume_paragraph_breaks(rule, text):
    assert "\n\n" in apply_rules(text, [rule])


def test_dash_normalization_skips_variable_and_acronym_pairs():
    original = "Use A—B, DNA—RNA and CoA—CoB. The report—an estimate—is ready."
    assert apply_rules(original, ["normalize_dashes"]) == (
        "Use A—B, DNA—RNA and CoA—CoB. The report, an estimate, is ready."
    )


def test_heading_markup_removes_closing_hashes_without_damaging_title_hashes():
    assert apply_rules("## Results ##\n### C#\n", ["strip_heading_markup"]) == (
        "Results\nC#\n"
    )


def test_semicolon_lists_remain_intact_while_explicit_clauses_can_split():
    original = (
        "We need apples; oranges; and pears. The first test ended; another began."
    )
    assert apply_rules(original, ["split_semicolons"]) == (
        "We need apples; oranges; and pears. The first test ended. Another began."
    )


def test_phrase_edits_do_not_rewrite_compounds_paths_or_possessives():
    original = "utilize-based pre-utilize utilize's /tmp/utilize/helpful foo\\utilize"
    assert apply_rules(original, ["simplify_verbs"]) == original
    nominal = "We use a decision-making tool and make a decision after the review."
    assert apply_rules(nominal, ["direct_nominal_phrases"]) == nominal.replace(
        "make a decision after", "decide after"
    )


def test_simplify_verbs_distinguishes_noun_endeavor_from_explicit_verb():
    original = (
        "An endeavor takes effort. We endeavor to finish and she endeavors to help."
    )
    assert apply_rules(original, ["simplify_verbs"]) == (
        "An endeavor takes effort. We try to finish and she tries to help."
    )


def test_affirmative_contractions_keep_stressed_relative_auxiliaries():
    original = "We know where we are now and how it is sometimes. We are ready."
    assert apply_rules(original, ["affirmative_contractions"]) == (
        "We know where we are now and how it is sometimes. We're ready."
    )


def test_auxiliary_contractions_preserve_all_caps_case():
    assert apply_rules(
        "WE HAVE REVIEWED THE NOTES.", ["auxiliary_have_contractions"]
    ) == ("WE'VE REVIEWED THE NOTES.")


@pytest.mark.parametrize(
    "text",
    [
        "Providing it works, the team will proceed.",
        "They are required to attend.",
        "What is required of us?",
        "The system provides users with support.",
        "They provided the team with support.",
    ],
)
def test_plain_verbs_skip_conditional_obligation_and_supply_with_contexts(text):
    assert apply_rules(text, ["common_plain_verbs"]) == text


def test_plain_verbs_can_still_edit_explicit_progressive_provision():
    assert apply_rules("We are providing support.", ["common_plain_verbs"]) == (
        "We are giving support."
    )


@pytest.mark.parametrize(
    "text",
    [
        "They could still have the ability to help.",
        "They do not actually have the ability to help.",
        "They want to eventually have the ability to help.",
        "Does she have the ability to help?",
    ],
)
def test_ability_phrases_do_not_erase_nonadjacent_qualifications(text):
    assert apply_rules(text, ["ability_phrases"]) == text


def test_seeded_synonyms_do_not_rewrite_noncontrast_however():
    original = "However carefully we work, errors remain. Choose however you prefer."
    for seed in range(20):
        assert apply_rules(original, ["seeded_synonyms"], seed=seed) == original


@pytest.mark.parametrize(
    "text",
    [
        "However, the company may still continue.",
        "However, the company, despite the delay, may still continue.",
        "HOWEVER, THE COMPANY MAY STILL CONTINUE.",
        "Nevertheless, the company may even so continue.",
        "Nonetheless, even so, the company may continue.",
    ],
)
def test_contrast_transitions_skip_targets_already_present_in_the_clause(text):
    assert apply_rules(text, ["contrast_transitions"]) == text


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "However, the report is ready. We still need approval.",
            "Still, the report is ready. We still need approval.",
        ),
        (
            "Nevertheless, the report is ready. Even so, we need approval.",
            "Even so, the report is ready. Even so, we need approval.",
        ),
        ("However, the company may continue.", "Still, the company may continue."),
        (
            "Nonetheless, the company may continue.",
            "Even so, the company may continue.",
        ),
        ("However, distillation may continue.", "Still, distillation may continue."),
    ],
)
def test_contrast_transitions_ignore_targets_in_later_sentences(text, expected):
    assert apply_rules(text, ["contrast_transitions"]) == expected
    assert quality_issues(text, expected) == []
