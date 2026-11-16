# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: William Droz <william.droz@idiap.ch>
# SPDX-License-Identifier: GPL-3.0-only

import pytest

from fakehuman.inspection import (
    inline_text_edits,
    refusal_reasons,
    trace_transformation,
)
from fakehuman.research_rules import RULES, apply_rules
from fakehuman.rule_examples import rule_examples


def test_rule_catalog_covers_custom_manifests_with_renderable_examples():
    names = list(RULES)
    examples = rule_examples(names, seed=17)
    assert [example["id"] for example in examples] == names
    for example in examples:
        assert example["label"] and example["description"]
        assert example["before"] != example["after"]
        assert example["after"] == apply_rules(example["before"], [example["id"]], 17)


def test_inline_edits_preserve_phrase_and_whitespace_changes_across_lines():
    original = "Éva waited due to the delay. Next came the report.\r\nKeep this line.\n"
    rewritten = (
        "Éva waited because of the delay.\n\nNext came the report.\r\nKeep this line."
    )
    edits = inline_text_edits(original, rewritten)
    assert [(edit["before"], edit["after"]) for edit in edits] == [
        ("due to", "because of"),
        (" ", "\n\n"),
        ("\n", ""),
    ]
    reconstructed = original
    for edit in reversed(edits):
        start, end = edit["before_start"], edit["before_end"]
        assert original[start:end] == edit["before"]
        assert rewritten[edit["after_start"] : edit["after_end"]] == edit["after"]
        reconstructed = reconstructed[:start] + edit["after"] + reconstructed[end:]
    assert reconstructed == rewritten


def test_inline_edits_keep_context_between_changes_in_a_long_repetitive_paragraph():
    context = "the sample remained intact. " * 3200
    edits = inline_text_edits(f"First. {context}Done.", f"Initial. {context}Finished.")
    assert [(edit["before"], edit["after"]) for edit in edits] == [
        ("First", "Initial"),
        ("Done", "Finished"),
    ]


@pytest.mark.parametrize("seed", [17, 42, 91])
@pytest.mark.parametrize(
    ("original", "names"),
    [
        (
            "Therefore, Éva prepared a brief report due to the delay. "
            "However, she did not change the 12 mm component.",
            [
                "lex_therefore_thus",
                "lex_initial_however_nevertheless",
                "lex_due_to",
                "lex_a_brief_short",
            ],
        ),
        (
            "Éva will utilize numerous helpful examples. "
            "Furthermore, she does not omit additional information.",
            [
                "simplify_verbs",
                "seeded_synonyms",
                "contractions",
                "broader_seeded_synonyms",
            ],
        ),
        ("A note (with an aside).\nAnother note.", ["syn_parenthetical_to_dashes"]),
        ("The sensor stays at 12 mm.", ["lex_therefore_thus"]),
        ("No rules are selected.", []),
    ],
)
def test_preview_edits_reconstruct_the_frozen_pipeline_with_original_rng(
    original, names, seed
):
    candidate, changes = trace_transformation(original, names, seed)
    assert candidate == apply_rules(original, names, seed)
    previous = original
    for change in changes:
        reconstructed = previous
        for edit in reversed(change["edits"]):
            assert previous[edit["before_start"] : edit["before_end"]] == edit["before"]
            reconstructed = (
                reconstructed[: edit["before_start"]]
                + edit["after"]
                + reconstructed[edit["before_end"] :]
            )
        for edit in change["edits"]:
            assert (
                reconstructed[edit["after_start"] : edit["after_end"]] == edit["after"]
            )
        assert reconstructed != previous
        assert change["rule"] in names
        assert change["label"]
        previous = reconstructed
    assert previous == candidate


@pytest.mark.parametrize(
    ("reason", "action"),
    [
        ("contradiction", "review the attempted changes"),
        ("low_entailment", "both directions"),
        ("alignment_failed", "shorter passage"),
        ("overlength", "shorter sentences"),
    ],
)
def test_semantic_refusals_explain_the_failed_check(reason, action):
    explanations = refusal_reasons([], {"passed": False, "reason": reason})
    assert len(explanations) == 1
    assert action in explanations[0]
