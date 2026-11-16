# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: William Droz <william.droz@idiap.ch>
# SPDX-License-Identifier: GPL-3.0-only

"""Meaning preservation checks for transformed prose."""

from __future__ import annotations

from difflib import SequenceMatcher

from .meaning import MeaningGuard


def _can_split_overlength(full: dict, before: str, after: str) -> bool:
    if (full.get("reason"), full.get("scope")) != ("overlength", "changed_sentences"):
        return False
    old = MeaningGuard._sentence_chunks(before)
    new = MeaningGuard._sentence_chunks(after)
    # Global sentence counts can match despite an insertion and a deletion in
    # different places. Require matching counts within every changed span.
    return all(
        i2 - i1 == j2 - j1
        for _, i1, i2, j1, j2 in SequenceMatcher(
            a=old, b=new, autojunk=False
        ).get_opcodes()
    )


def _changed_sentence_pairs(before: str, after: str) -> list[tuple[str, str]] | None:
    old = MeaningGuard._sentence_chunks(before)
    new = MeaningGuard._sentence_chunks(after)
    if len(old) != len(new):
        return None
    return [
        (left, right) for left, right in zip(old, new, strict=True) if left != right
    ]


def check_document_pair(guard, before: str, after: str) -> dict:
    """Check a text pair, splitting oversized aligned groups of changed sentences."""
    full = guard.check_pairs([(before, after)])[0]
    if not _can_split_overlength(full, before, after):
        return full
    pairs = _changed_sentence_pairs(before, after)
    if not pairs or len(pairs) < 2:
        return full
    results = guard.check_pairs(pairs)
    if len(results) != len(pairs):
        return full
    failed = next((item for item in results if not item["passed"]), None)
    return {
        "passed": failed is None,
        "reason": failed["reason"] if failed else "bidirectional_entailment",
        "scope": "individual_changed_sentences",
        "min_entailment": min(item["min_entailment"] for item in results),
        "max_contradiction": max(item["max_contradiction"] for item in results),
    }


def screen_candidate(
    guard, before: str, after: str, *, allow_overlength: bool = False
) -> dict:
    """Check whole prose and each changed sentence, optionally splitting long spans."""
    if before == after:
        return {
            "passed": True,
            "reason": "identity",
            "full": {"passed": True},
            "sentences": [],
        }
    sentence_pairs = _changed_sentence_pairs(before, after)
    if sentence_pairs is None or not sentence_pairs:
        return {
            "passed": False,
            "reason": "sentence_alignment_failed",
            "full": None,
            "sentences": [],
        }
    try:
        full = guard.check_pairs([(before, after)])[0]
        sentence_results = guard.check_pairs(sentence_pairs)
    except Exception:  # NLI failures are refusals in this safety gate.
        return {
            "passed": False,
            "reason": "nli_error",
            "full": None,
            "sentences": [],
        }
    sentence_fallback = (
        allow_overlength
        and not full.get("passed")
        and len(sentence_pairs) > 1
        and _can_split_overlength(full, before, after)
    )
    passed = (bool(full.get("passed")) or sentence_fallback) and all(
        item.get("passed") for item in sentence_results
    )
    return {
        "passed": passed,
        "reason": "bidirectional_entailment" if passed else "nli_refusal",
        **({"scope": "individual_changed_sentences"} if sentence_fallback else {}),
        "full": full,
        "sentences": [
            {"before": left, "after": right, "result": result}
            for (left, right), result in zip(
                sentence_pairs, sentence_results, strict=True
            )
        ],
    }
