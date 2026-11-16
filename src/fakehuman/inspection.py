# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: William Droz <william.droz@idiap.ch>
# SPDX-License-Identifier: GPL-3.0-only

"""Explain frozen transformations without modifying their evaluated implementation."""

from __future__ import annotations

import difflib
import re
from collections.abc import Callable
from itertools import accumulate, groupby

from .research_rules import RULE_LABELS, apply_rules

_WORD_TOKENS = re.compile(r"\w+|[^\w\s]|\s+")
_MAX_INLINE_MATCH_CELLS = 2_000_000

type _EditRange = tuple[int, int, int, int]


def text_edits(before: str, after: str) -> list[dict]:
    """Return word-level edits with Python string offsets in each version."""
    return _word_edits(before, after)


def inline_text_edits(before: str, after: str) -> list[dict]:
    """Return display edits that retain line context and refine changed words."""
    if before == after:
        return []
    old_lines = before.splitlines(keepends=True)
    new_lines = after.splitlines(keepends=True)
    old_offsets = _offsets(old_lines)
    new_offsets = _offsets(new_lines)
    changes = []
    for i1, i2, j1, j2 in _inline_ranges(old_lines, new_lines):
        start, end = old_offsets[i1], old_offsets[i2]
        new_start, new_end = new_offsets[j1], new_offsets[j2]
        changes.extend(
            _word_edits(
                before[start:end],
                after[new_start:new_end],
                inline=True,
                before_offset=start,
                after_offset=new_start,
            )
        )
    return changes


def _offsets(parts: list[str]) -> list[int]:
    return list(accumulate(map(len, parts), initial=0))


def _matching_edits(
    old: list[str],
    new: list[str],
    *,
    autojunk: bool = True,
    isjunk: Callable[[str], bool] | None = None,
) -> list[_EditRange]:
    matcher = difflib.SequenceMatcher(isjunk, a=old, b=new, autojunk=autojunk)
    return [
        (i1, i2, j1, j2)
        for operation, i1, i2, j1, j2 in matcher.get_opcodes()
        if operation != "equal"
    ]


def _inline_ranges(
    old: list[str], new: list[str], *, isjunk: Callable[[str], bool] | None = None
) -> list[_EditRange]:
    # Isolate common edges before matching so an edit in a long, repetitive
    # passage does not consume its surrounding context or cause quadratic work.
    start = 0
    overlap = min(len(old), len(new))
    while start < overlap and old[start] == new[start]:
        start += 1
    old_end, new_end = len(old), len(new)
    while min(old_end, new_end) > start and old[old_end - 1] == new[new_end - 1]:
        old_end -= 1
        new_end -= 1
    if start == old_end == new_end:
        return []
    if (old_end - start) * (new_end - start) > _MAX_INLINE_MATCH_CELLS:
        return _large_inline_ranges(old, new, start, old_end, new_end)
    return [
        (i1 + start, i2 + start, j1 + start, j2 + start)
        for i1, i2, j1, j2 in _matching_edits(
            old[start:old_end], new[start:new_end], autojunk=False, isjunk=isjunk
        )
    ]


def _large_inline_ranges(
    old: list[str], new: list[str], start: int, old_end: int, new_end: int
) -> list[_EditRange]:
    # Sparse in-place edits can be recovered in linear time. Use a single span
    # when the two large regions do not have a clear positional alignment.
    if old_end == new_end:
        equal = [old[index] == new[index] for index in range(start, old_end)]
        if sum(equal) >= 0.8 * len(equal):
            ranges = []
            for same, group in groupby(equal):
                end = start + sum(1 for _ in group)
                if not same:
                    ranges.append((start, end, start, end))
                start = end
            return ranges
    return [(start, old_end, start, new_end)]


def _word_edits(
    before: str,
    after: str,
    *,
    inline: bool = False,
    before_offset: int = 0,
    after_offset: int = 0,
) -> list[dict]:
    old = _WORD_TOKENS.findall(before)
    new = _WORD_TOKENS.findall(after)
    old_offsets = _offsets(old)
    new_offsets = _offsets(new)
    # Spaces alone should not split a phrase replacement into individual words.
    ranges = (
        _inline_ranges(old, new, isjunk=str.isspace)
        if inline
        else _matching_edits(old, new)
    )
    changes = []
    for i1, i2, j1, j2 in ranges:
        start, end = old_offsets[i1], old_offsets[i2]
        new_start, new_end = new_offsets[j1], new_offsets[j2]
        changes.append(
            {
                "before_start": before_offset + start,
                "before_end": before_offset + end,
                "after_start": after_offset + new_start,
                "after_end": after_offset + new_end,
                "before": before[start:end],
                "after": after[new_start:new_end],
            }
        )
    return changes


def trace_transformation(
    text: str, names: list[str], seed: int = 17
) -> tuple[str, list[dict]]:
    """Return the candidate and changed rules with reconstructable passage edits.

    Every stage uses the frozen implementation with the original text and a
    prefix of the rule sequence. This preserves its original-document random
    seeds, including when a previous rule changed the text. Edit offsets are
    Unicode codepoint indices relative to that rule's input and output, rather
    than the complete pipeline's original and final text. Unchanged rules are
    omitted.
    """
    previous = text
    changes = []
    for index, name in enumerate(names):
        current = apply_rules(text, names[: index + 1], seed)
        if current != previous:
            changes.append(
                {
                    "rule": name,
                    "label": RULE_LABELS.get(name, name),
                    "edits": text_edits(previous, current),
                }
            )
        previous = current
    return previous, changes


def refusal_reasons(issues: list[str], meaning: dict) -> list[str]:
    """Explain the failed checks and what a caller can review or shorten."""
    reasons = [f"Mechanical preservation check failed: {issue}." for issue in issues]
    if not meaning["passed"]:
        reason = meaning.get("reason", "unknown")
        explanations = {
            "contradiction": (
                "The meaning screen detected a possible contradiction; "
                "review the attempted changes."
            ),
            "low_entailment": (
                "The meaning screen did not establish sufficient entailment in "
                "both directions; review the attempted changes."
            ),
            "alignment_failed": (
                "The meaning screen could not verify the changed text because "
                "sentence alignment failed; try a shorter passage."
            ),
            "overlength": (
                "The changed comparison exceeds the meaning screen's token limit "
                "and was not scored; try a smaller edit or shorter sentences."
            ),
            "empty_input": "The meaning screen received empty text; review the candidate.",
        }
        reasons.append(
            explanations.get(
                reason, f"The meaning screen refused the candidate: {reason}."
            )
        )
    return reasons
