# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: William Droz <william.droz@idiap.ch>
# SPDX-License-Identifier: GPL-3.0-only

"""Secondary detector diagnostics over complete, overlapping token windows.

Offsets refer to each detector's prepared text, not the source Markdown. Window
means describe this segmentation; they are not whole-document probabilities.
Overlapping scores are dependent observations of the same local classifier.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class WindowSpan:
    token_start: int
    token_end: int
    start: int
    end: int


def window_spans(
    offsets: Sequence[Sequence[int]], *, budget: int, stride: int = 256
) -> list[WindowSpan]:
    """Cover all tokens with full windows, including an aligned final window.

    Token and character bounds are half open. Shared character offsets (e.g.
    byte-level tokens representing one Unicode character) are permitted. No
    tokenizer decoding or approximation from character counts is performed.
    """
    if type(budget) is not int or budget < 1:
        raise ValueError("budget must be a positive integer")
    if type(stride) is not int or not 1 <= stride <= budget:
        raise ValueError("stride must be between one and budget to prevent gaps")
    previous = (0, 0)
    for pair in offsets:
        if (
            len(pair) != 2
            or any(type(value) is not int for value in pair)
            or pair[0] < 0
            or pair[1] < pair[0]
            or pair[0] < previous[0]
            or pair[1] < previous[1]
        ):
            raise ValueError("offsets must be ordered nonnegative character spans")
        previous = (pair[0], pair[1])
    count = len(offsets)
    if not count:
        return []
    if count <= budget:
        starts = [0]
    else:
        starts = sorted(set(range(0, count - budget + 1, stride)) | {count - budget})
    return [
        WindowSpan(
            token_start=start,
            token_end=min(start + budget, count),
            start=offsets[start][0],
            end=offsets[min(start + budget, count) - 1][1],
        )
        for start in starts
    ]


def _summary(scores: Sequence[float]) -> dict:
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in scores):
        raise ValueError("Detector scores must be finite values between zero and one")
    return {
        "mean": math.fsum(scores) / len(scores),
        "min": min(scores),
        "max": max(scores),
    }


def _unavailable(base: dict, reason: str) -> dict:
    return {
        **base,
        "available": False,
        "reason": reason,
        "windows": [],
        "mean": None,
        "min": None,
        "max": None,
        "token_coverage_complete": False,
    }


def _score_one(text: str, detector: Any, counter: Any, stride: int) -> dict:
    prepared = counter.prepare(text)
    base = {
        "method": "descriptive_overlapping_window_mean",
        "offset_reference": "detector_prepared_text",
        "source_characters": len(text),
        "prepared_characters": len(prepared),
        "prepared_sha256": hashlib.sha256(prepared.encode()).hexdigest(),
        "preprocessing": counter.metadata(),
        "stride_tokens": stride,
    }
    if not prepared.strip():
        return _unavailable(base, "No scorable content after detector preprocessing")

    # TokenCounter currently has no public tokenizer accessor. Initialize its
    # pinned tokenizer through its public counting API, then use the cached
    # tokenizer solely for offsets. This never accesses classifier weights.
    count_prepared = getattr(counter, "count_prepared", None)
    score_prepared = getattr(detector, "score_prepared", None)
    full_count = count_prepared(prepared) if callable(count_prepared) else counter(text)
    tokenizer = counter._tokenizer
    if tokenizer is None or not getattr(tokenizer, "is_fast", False):
        return _unavailable(base, "A fast tokenizer with offsets is required")
    encoded = tokenizer(
        prepared,
        add_special_tokens=False,
        truncation=False,
        return_offsets_mapping=True,
        verbose=False,
    )
    offsets = encoded["offset_mapping"]
    if len(offsets) != len(encoded["input_ids"]):
        raise ValueError("Token IDs and offset mapping have different lengths")
    if any(end > len(prepared) for _, end in offsets):
        raise ValueError("Tokenizer offsets extend beyond the prepared text")
    max_tokens = counter.spec.max_tokens
    special_tokens = tokenizer.num_special_tokens_to_add(False)
    budget = max_tokens - special_tokens
    base.update(
        {
            "max_tokens": max_tokens,
            "special_tokens": special_tokens,
            "full_token_count": full_count,
            "content_token_count": len(offsets),
        }
    )
    if not offsets:
        return _unavailable(base, "Tokenizer returned no content tokens")
    if full_count <= max_tokens:
        scores = list(detector.score([text]))
        if len(scores) != 1:
            raise ValueError("Detector did not return one score per window")
        return {
            **base,
            "available": True,
            "method": "single_window_native",
            "content_budget": budget,
            "budget_reductions": 0,
            "token_coverage_complete": True,
            "covered_token_count": len(offsets),
            "windows": [
                {
                    "index": 0,
                    "token_start": 0,
                    "token_end": len(offsets),
                    "start": 0,
                    "end": len(prepared),
                    "token_count": full_count,
                    "score": scores[0],
                }
            ],
            **_summary(scores),
        }

    reductions = 0
    while budget >= stride:
        spans = window_spans(offsets, budget=budget, stride=stride)
        texts = [prepared[span.start : span.end] for span in spans]
        # Prepared APIs preserve the exact slices even when cleanup is not
        # idempotent at a new Markdown boundary. Legacy injected adapters still
        # need the guard, since their counting or scoring may clean once more.
        if (not callable(count_prepared) or not callable(score_prepared)) and any(
            counter.prepare(part) != part for part in texts
        ):
            return _unavailable(
                base, "Prepared window changes under repeated detector preprocessing"
            )
        count_window = count_prepared if callable(count_prepared) else counter
        counts = [count_window(part) for part in texts]
        if all(count <= max_tokens for count in counts):
            break
        budget -= 8
        reductions += 1
    else:
        return _unavailable(
            base, "No safe window budget at the frozen stride; coverage would have gaps"
        )
    covered = 0
    for span in spans:
        if span.token_start > covered:
            raise AssertionError("Window plan has an uncovered token gap")
        covered = max(covered, span.token_end)
    if covered != len(offsets):
        raise AssertionError("Window plan does not cover the complete token sequence")
    score_windows = score_prepared if callable(score_prepared) else detector.score
    if any(not part.strip() for part in texts):
        # A tokenizer can assign tokens to long runs of Unicode whitespace
        # even though the complete document contains prose.
        if any(not part for part in texts):
            return _unavailable(base, "Tokenizer returned an empty window slice")
        score_windows = getattr(detector, "_score_prepared_windows", None)
        if not callable(score_windows):
            return _unavailable(base, "Detector cannot score whitespace-only windows")
    scores = list(score_windows(texts))
    if len(scores) != len(spans):
        raise ValueError("Detector did not return one score per window")
    return {
        **base,
        "available": True,
        "prepared_scoring": callable(score_prepared),
        "content_budget": budget,
        "budget_reductions": reductions,
        "token_coverage_complete": True,
        "covered_token_count": covered,
        "windows": [
            {
                "index": index,
                **asdict(span),
                "token_count": count,
                "score": score,
            }
            for index, (span, count, score) in enumerate(zip(spans, counts, scores))
        ],
        **_summary(scores),
    }


def score_document_windows(
    text: str,
    *,
    detectors: Mapping[str, Any],
    counters: Mapping[str, Any],
    stride: int = 256,
) -> dict:
    """Score each detector's complete prepared document without truncation.

    ``detectors`` supplies objects exposing ``score(list[str])`` and preferably
    ``score_prepared(list[str])``. ``counters`` supplies TokenCounter-compatible
    objects, preferably with ``count_prepared(str)``. Legacy adapters are used
    only when cleaning their prepared slices again leaves them unchanged.
    No remote generation or classifier download is performed here;
    passed detectors retain responsibility for loading their existing models.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if type(stride) is not int or stride < 1:
        raise ValueError("stride must be a positive integer")
    if not detectors or set(detectors) != set(counters):
        raise ValueError("detectors and counters must have matching nonempty names")
    return {
        "method": "descriptive_overlapping_window_mean",
        "stride_tokens": stride,
        "detectors": {
            name: _score_one(text, detector, counters[name], stride)
            for name, detector in detectors.items()
        },
        "limitations": [
            "Window means are descriptive, not native whole-document probabilities.",
            "Overlapping windows are dependent observations, not independent samples.",
            "This is a consistency diagnostic of the same detectors, not a holdout.",
            "Offsets refer to detector-prepared text; preprocessing can remove content.",
        ],
    }
