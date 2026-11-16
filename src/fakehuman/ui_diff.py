# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: William Droz <william.droz@idiap.ch>
# SPDX-License-Identifier: GPL-3.0-only

"""Render prose changes for the workbench."""

import re
from html import escape

from .inspection import inline_text_edits

INLINE_DIFF_CSS = """
.workbench-inline-diff {
    max-height: 24rem;
    overflow: auto;
    overflow-wrap: anywhere;
    white-space: pre-wrap;
    padding: .75rem 1rem;
    border: 1px solid #e2e8f0;
    border-radius: .5rem;
    line-height: 1.75;
}
.workbench-inline-diff del {
    color: #881337;
    background: #ffe4e6;
    text-decoration: line-through;
}
.workbench-inline-diff ins {
    color: #115e59;
    background: #ccfbf1;
    text-decoration: underline;
}
.workbench-inline-diff .diff-replacement { margin-inline-start: .2em; }
.workbench-inline-diff .diff-omitted {
    display: block;
    color: #475569;
    font-size: .85em;
    white-space: normal;
    margin-block: .3em;
}
"""
_CONTEXT_CHARS = 180
_WHITESPACE_MARKS = {" ": "␠", "\t": "⇥", "\n": "↵\n", "\r": "␍", "\xa0": "⍽"}


def _context(text: str, *, at_start: bool, at_end: bool) -> str:
    if len(text) <= 2 * _CONTEXT_CHARS:
        return escape(text)
    head = "" if at_start else re.sub(r"\s+\S{0,24}$", "", text[:_CONTEXT_CHARS])
    tail = "" if at_end else re.sub(r"^\S{0,24}\s+", "", text[-_CONTEXT_CHARS:])
    return (
        escape(head)
        + '<span class="diff-omitted">… unchanged text …</span>'
        + escape(tail)
    )


def _marked_text(text: str) -> str:
    if text.isspace():
        # Make edits like a trailing newline or a second space visible.
        text = "".join(_WHITESPACE_MARKS.get(char, "␠") for char in text)
    return escape(text)


def inline_diff_html(before: str, after: str) -> str:
    """Show changed words inline and fold distant unchanged context."""
    edits = inline_text_edits(before, after)
    if not edits:
        return ""

    fragments = []
    cursor = 0
    for index, edit in enumerate(edits):
        fragments.append(
            _context(
                before[cursor : edit["before_start"]], at_start=index == 0, at_end=False
            )
        )
        if removed := edit["before"]:
            fragments.append(f"<del>{_marked_text(removed)}</del>")
        if added := edit["after"]:
            css_class = ' class="diff-replacement"' if removed else ""
            fragments.append(f"<ins{css_class}>{_marked_text(added)}</ins>")
        cursor = edit["before_end"]
    fragments.append(_context(before[cursor:], at_start=False, at_end=True))
    return "".join(fragments)
