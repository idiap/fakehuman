# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: William Droz <william.droz@idiap.ch>
# SPDX-License-Identifier: GPL-3.0-only

"""Keep lexical demonstrations in sentence contexts rather than bare fragments."""

import pytest

from fakehuman.candidates_lexical import _PAIRS, RULES
from fakehuman.research_rules import apply_rules
from fakehuman.rule_examples import _LEXICAL_SENTENCES, rule_examples


def test_every_lexical_pair_has_an_explicit_sentence_example():
    assert set(_LEXICAL_SENTENCES) == {name for name, _, _ in _PAIRS}


@pytest.mark.parametrize(
    "example", rule_examples(tuple(RULES), 17), ids=lambda row: row["id"]
)
def test_lexical_examples_include_context_and_show_the_real_edit(example):
    before = example["before"].strip()
    after = example["after"].strip()
    assert before.endswith(".") and after.endswith(".")
    assert len(before.split()) >= 4
    assert len(after.split()) >= 4
    assert before != after
    assert example["after"] == apply_rules(example["before"], [example["id"]], 17)
    source = next(
        (source for name, source, _ in _PAIRS if name == example["id"]), "However"
    )
    assert source.lower() in before.lower()
    assert before.lower().rstrip(".") != source.lower()
