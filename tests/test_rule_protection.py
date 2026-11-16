# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: William Droz <william.droz@idiap.ch>
# SPDX-License-Identifier: GPL-3.0-only

"""Exercise every catalog rule's actual trigger inside protected content."""

import html
import json

import pytest

from fakehuman.research_rules import RULES, apply_rules
from fakehuman.rule_examples import rule_examples

_EXAMPLES = rule_examples(tuple(RULES), seed=17)
_SEEDS = (17, 42, 91)
_PLAIN_QUOTES = str.maketrans("“”‘’", "\"\"''")


def _protected_cases():
    for example in _EXAMPLES:
        rule = example["id"]
        sample = example["before"].strip()
        surfaces = {
            "backtick-fence": f"```text\n{sample}\n```\n",
            "tilde-fence": f"~~~text\n{sample}\n~~~\n",
            "html-code": f"<code>\n{html.escape(sample)}\n</code>",
            "html-pre": f"<pre>\n{html.escape(sample)}\n</pre>",
        }
        if "\n" not in sample and "\r" not in sample:
            # Keep inline fixtures syntactically valid. HTML attributes and
            # Markdown labels escape their delimiters; quotation fixtures use
            # ordinary nested quotation marks rather than unmatched delimiters.
            quoted_sample = sample.replace("“", "'").replace("”", "'")
            link_label = sample.replace("[", r"\[").replace("]", r"\]")
            surfaces.update(
                {
                    "html-attribute": (
                        f'<span title="{html.escape(sample, quote=True)}">label</span>'
                    ),
                    "straight-quotation": json.dumps(sample, ensure_ascii=False),
                    "curly-quotation": f"“{quoted_sample}”",
                    "markdown-link": f"[{link_label}](https://example.org/record)",
                }
            )
        for surface, original in surfaces.items():
            expected = original
            if rule == "normalize_quotes" and surface.endswith("quotation"):
                # This one rule explicitly normalizes quotation typography.
                # Every other byte of the quoted wording must survive.
                expected = original.translate(_PLAIN_QUOTES)
            yield pytest.param(rule, original, expected, id=f"{rule}-{surface}")


def test_protection_examples_cover_every_rule_with_a_working_trigger():
    assert {example["id"] for example in _EXAMPLES} == set(RULES)
    for example in _EXAMPLES:
        original = example["before"]
        assert any(
            apply_rules(original, [example["id"]], seed=seed) != original
            for seed in _SEEDS
        ), example["id"]


@pytest.mark.parametrize(("rule", "original", "expected"), _protected_cases())
def test_each_rule_preserves_its_trigger_inside_protected_content(
    rule, original, expected
):
    for seed in _SEEDS:
        assert apply_rules(original, [rule], seed=seed) == expected, f"seed={seed}"
