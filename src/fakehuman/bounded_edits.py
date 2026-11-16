# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: William Droz <william.droz@idiap.ch>
# SPDX-License-Identifier: GPL-3.0-only

"""Offline mechanical validation of proposed edits to one to five words.

This prototype neither generates text nor changes the frozen research pipeline.
Offsets are Python Unicode character offsets, not UTF-8 bytes or JavaScript
UTF-16 code units. A proposal is bound to the exact UTF-8 source SHA-256.

Passing these checks does NOT establish equivalent meaning or fluent writing.
After validation, inspect changed sentences in context, run the meaning screen,
and score the reconstructed full text with both detectors. Never score JSON
wrappers or isolated replacement fragments as evidence about the full text.
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from dataclasses import dataclass

from .rules import _HEDGES, _NEGATIONS, _PROTECTED, _WORD, quality_issues

MAX_SPAN_WORDS = 5
_PROPOSAL_KEYS = {"source_sha256", "edits"}
_EDIT_KEYS = {"start", "end", "original", "replacement"}


@dataclass(frozen=True)
class ProposalValidation:
    """Mechanical result; invalid proposals return the unchanged source text."""

    mechanically_valid: bool
    text: str
    issues: tuple[str, ...]
    applied_edits: int
    semantic_status: str = "not_assessed"
    detector_status: str = "not_assessed"


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _invalid_constant(value: str) -> object:
    raise ValueError(f"Non-JSON numeric constant: {value}")


def parse_proposal_json(payload: str) -> dict[str, object]:
    """Parse JSON without duplicate keys, NaN, or non-object top-level values.

    Schema and source checks still require :func:`validate_proposal`.
    """
    if not isinstance(payload, str):
        raise TypeError("JSON payload must be a string")
    proposal = json.loads(
        payload, object_pairs_hook=_unique_object, parse_constant=_invalid_constant
    )
    if not isinstance(proposal, dict):
        raise ValueError("Proposal JSON must be an object")
    return proposal


def _punctuation(text: str) -> str:
    # Symbols are included: scientific operators must not disappear either.
    return "".join(char for char in text if not char.isalnum() and not char.isspace())


def _span_issues(source: str, edit: dict, index: int) -> list[str]:
    prefix = f"edit {index}: "
    start, end = edit["start"], edit["end"]
    old, new = edit["original"], edit["replacement"]
    issues = []
    if source[start:end] != old:
        return [prefix + "original does not match source at these offsets"]
    words = list(_WORD.finditer(source))
    if any(
        match.start() < edge < match.end() for match in words for edge in (start, end)
    ):
        issues.append(prefix + "offset cuts through a word")
    if any(not 1 <= len(_WORD.findall(span)) <= MAX_SPAN_WORDS for span in (old, new)):
        issues.append(prefix + "source and replacement must each contain 1–5 words")
    if any(span != span.strip() for span in (old, new)):
        issues.append(prefix + "spans must not begin or end with whitespace")
    if any(char.isspace() and char != " " for char in old + new):
        issues.append(prefix + "only ordinary spaces are allowed inside edited spans")
    if any(unicodedata.category(char).startswith("C") for char in old + new):
        issues.append(prefix + "control, format, or surrogate characters are forbidden")
    # Restrict edited words to ASCII English. Unicode elsewhere remains exact.
    # This deliberately refuses accented/scientific words rather than guessing
    # which visually similar characters are interchangeable.
    if any(char.isalnum() and not char.isascii() for char in old + new):
        issues.append(prefix + "edited words must use ASCII letters and digits")
    if _punctuation(old) != _punctuation(new):
        issues.append(prefix + "punctuation and symbol sequence must remain identical")
    if any(
        start < match.end() and end > match.start()
        for match in _PROTECTED.finditer(source)
    ):
        issues.append(prefix + "span overlaps a protected URL, DOI, citation, or code")
    if old != new and any(
        pattern.search(span) for pattern in (_NEGATIONS, _HEDGES) for span in (old, new)
    ):
        issues.append(
            prefix + "negation or uncertainty inside an edit requires a narrower span"
        )
    return issues


def validate_proposal(
    source: str, proposal: object, *, max_edits: int = 2
) -> ProposalValidation:
    """Validate a strict proposal object and reconstruct its text mechanically.

    Schema: ``{"source_sha256": str, "edits": [{"start": int, "end": int,
    "original": str, "replacement": str}]}``. Unknown fields are errors. Integer
    offsets exclude booleans. Edits use original-source coordinates, cannot
    overlap, and cannot insert/delete whole spans. Empty edits mean identity.
    Invalid proposals fail atomically: none of their changes are returned.
    """
    if not isinstance(source, str):
        raise TypeError("source must be a string")
    if type(max_edits) is not int or max_edits < 0:
        raise ValueError("max_edits must be a nonnegative integer")

    def failed(issues: list[str]) -> ProposalValidation:
        return ProposalValidation(False, source, tuple(issues), 0)

    try:
        source_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
    except UnicodeEncodeError:
        return failed(["source must be valid UTF-8 text"])
    if not isinstance(proposal, dict) or proposal.keys() != _PROPOSAL_KEYS:
        return failed(["proposal must contain exactly source_sha256 and edits"])
    if (
        not isinstance(proposal["source_sha256"], str)
        or proposal["source_sha256"] != source_hash
    ):
        return failed(["source_sha256 does not match the exact source"])
    edits = proposal["edits"]
    if not isinstance(edits, list) or len(edits) > max_edits:
        return failed([f"edits must be a list containing at most {max_edits} entries"])
    issues = []
    for index, edit in enumerate(edits):
        if not isinstance(edit, dict) or edit.keys() != _EDIT_KEYS:
            issues.append(
                f"edit {index}: expected exactly start, end, original, replacement"
            )
            continue
        if any(type(edit[key]) is not int for key in ("start", "end")):
            issues.append(f"edit {index}: offsets must be integers, excluding booleans")
            continue
        if not 0 <= edit["start"] < edit["end"] <= len(source):
            issues.append(f"edit {index}: offsets must identify a nonempty source span")
            continue
        if any(not isinstance(edit[key], str) for key in ("original", "replacement")):
            issues.append(f"edit {index}: original and replacement must be strings")
            continue
        issues.extend(_span_issues(source, edit, index))
    if issues:
        return failed(issues)
    ordered = sorted(edits, key=lambda edit: edit["start"])
    if any(left["end"] > right["start"] for left, right in zip(ordered, ordered[1:])):
        return failed(["edits overlap in original-source coordinates"])

    # Reverse application preserves source offsets and every character outside
    # the explicitly bound spans, including all whitespace and Unicode.
    result = source
    for edit in reversed(ordered):
        result = result[: edit["start"]] + edit["replacement"] + result[edit["end"] :]
    issues = quality_issues(source, result)
    if issues:
        return failed(issues)
    return ProposalValidation(
        True, result, (), sum(edit["original"] != edit["replacement"] for edit in edits)
    )


def apply_proposal(source: str, proposal: object, *, max_edits: int = 2) -> str:
    """Return a mechanically validated string or raise; semantics remain unchecked."""
    result = validate_proposal(source, proposal, max_edits=max_edits)
    if not result.mechanically_valid:
        raise ValueError("Invalid bounded edit proposal: " + "; ".join(result.issues))
    return result.text
