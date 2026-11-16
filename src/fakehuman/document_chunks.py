# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: William Droz <william.droz@idiap.ch>
# SPDX-License-Identifier: GPL-3.0-only

"""Lossless document sections and explicitly described section-score summaries.

Sections always reference contiguous slices of the original document. Token
budgets include special tokens and each detector's own preprocessing. The
original sections and word weights remain fixed when comparing rewritten text.
"""

from __future__ import annotations

import math
import re
from bisect import bisect_right
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from .detectors import (
    DEFAULT_DETECTOR_NAMES,
    DETECTOR_SPECS,
    DetectorSpec,
    fakespot_clean_text,
)

TokenCounters = Mapping[str, Callable[[str], int]]
MAX_DOCUMENT_CHARACTERS = 100_000
MAX_TECHNICAL_CHARACTERS = 24_000


class TokenCounter:
    """Load only a pinned tokenizer, never classifier weights or remote code."""

    def __init__(self, spec: DetectorSpec) -> None:
        self.spec = spec
        self._tokenizer = None

    def prepare(self, text: str) -> str:
        return (
            fakespot_clean_text(text)
            if self.spec.preprocessing == "fakespot_clean_text"
            else text
        )

    def __call__(self, text: str) -> int:
        return self.count_prepared(self.prepare(text))

    def count_prepared(self, text: str) -> int:
        """Count exact prepared text, including special tokens, without cleaning."""
        if self._tokenizer is None:
            try:
                from transformers import AutoTokenizer
            except ImportError as exc:
                raise RuntimeError(
                    "Install tokenizer dependencies: uv sync --locked --extra cpu"
                ) from exc
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.spec.model_id,
                revision=self.spec.revision,
                trust_remote_code=False,
            )
        return len(
            self._tokenizer(
                text, add_special_tokens=True, truncation=False, verbose=False
            )["input_ids"]
        )

    def metadata(self) -> dict:
        return {
            "model_id": self.spec.model_id,
            "revision": self.spec.revision,
            "preprocessing": self.spec.preprocessing,
            "add_special_tokens": True,
            "truncation": False,
            "max_tokens": self.spec.max_tokens,
            "trust_remote_code": False,
        }


def build_token_counters(
    names: Sequence[str] | None = None,
) -> dict[str, TokenCounter]:
    names = list(DEFAULT_DETECTOR_NAMES if names is None else names)
    if not names or len(names) != len(set(names)):
        raise ValueError("Choose distinct detector names")
    unknown = set(names).difference(DETECTOR_SPECS)
    if unknown:
        raise ValueError("Unknown detector(s): " + ", ".join(sorted(unknown)))
    return {name: TokenCounter(DETECTOR_SPECS[name]) for name in names}


@dataclass(frozen=True)
class DocumentChunk:
    index: int
    start: int
    end: int
    text: str
    weight: int
    token_counts: dict[str, int]
    warnings: tuple[str, ...] = ()


def _token_counts(text: str, counters: TokenCounters) -> dict[str, int]:
    values = {name: counter(text) for name, counter in counters.items()}
    if any(type(value) is not int or value < 0 for value in values.values()):
        raise ValueError("Token counters must return nonnegative integers")
    return values


def split_document(
    text: str,
    token_counters: TokenCounters,
    *,
    target_tokens: int = 360,
) -> list[DocumentChunk]:
    """Split at paragraphs, list items, sentences, then words or characters.

    A balanced target reduces short final sections. Boundary search is a
    heuristic: tokenization and Markdown cleanup are not necessarily monotonic
    as a prefix grows. Every selected section is nevertheless measured and
    checked independently, so this never relies on a monotonicity assumption
    for length safety. Character fallback always makes progress or raises.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("The document must be a nonempty string")
    if not token_counters:
        raise ValueError("At least one token counter is required")
    if type(target_tokens) is not int or target_tokens < 1:
        raise ValueError("target_tokens must be a positive integer")

    # End positions retain separators in the preceding original section.
    paragraphs = {match.end() for match in re.finditer(r"(?:\r?\n[ \t]*){2,}", text)}
    list_items = {
        match.start()
        for match in re.finditer(r"(?m)^[ \t]*(?:[-*+]\s+|\d+[.)]\s+)", text)
    }
    sentences = {match.end() for match in re.finditer(r"[.!?][\"'”’)\]]*\s+", text)}
    whitespace = {match.end() for match in re.finditer(r"\s+", text)}
    natural = paragraphs | list_items | sentences
    families = [
        sorted(paragraphs | list_items | {len(text)}),
        sorted(sentences | {len(text)}),
        sorted(whitespace | {len(text)}),
    ]
    chunks: list[DocumentChunk] = []
    start = 0
    previous_split_inside_sentence = False
    while start < len(text):
        cache: dict[int, dict[str, int]] = {}

        def counts(end: int) -> dict[str, int]:
            if end not in cache:
                cache[end] = _token_counts(text[start:end], token_counters)
            return cache[end]

        def size(end: int) -> int:
            return max(counts(end).values())

        remaining_size = size(len(text))
        if remaining_size <= target_tokens:
            end = len(text)
        else:
            ideal = remaining_size / math.ceil(remaining_size / target_tokens)

            def nearby(boundaries: Sequence[int]) -> list[int]:
                """Measure a logarithmic selection of possible boundaries."""
                low = bisect_right(boundaries, start)
                high = len(boundaries)
                visited: set[int] = set()
                while low < high:
                    middle = (low + high) // 2
                    candidate = boundaries[middle]
                    visited.add(candidate)
                    if size(candidate) < ideal:
                        low = middle + 1
                    else:
                        high = middle
                first = bisect_right(boundaries, start)
                for position in (first, low - 1, low):
                    if first <= position < len(boundaries):
                        visited.add(boundaries[position])
                return [
                    candidate
                    for candidate in visited
                    if size(candidate) <= target_tokens
                ]

            end = 0
            fallback: list[int] = []
            for family in families:
                candidates = nearby(family)
                fallback.extend(candidates)
                substantive = [
                    candidate
                    for candidate in candidates
                    if size(candidate) >= ideal * 0.65 and text[start:candidate].strip()
                ]
                if substantive:
                    end = min(
                        substantive,
                        key=lambda value: (abs(size(value) - ideal), -value),
                    )
                    break
            if not end:
                # range is lazy even for very long whitespace-free strings.
                characters = nearby(range(start + 1, len(text) + 1))
                candidates = characters + fallback
                if not candidates:
                    raise ValueError(
                        f"No source section at character {start} fits "
                        f"target_tokens={target_tokens}; increase the token budget"
                    )
                end = min(
                    candidates, key=lambda value: (abs(size(value) - ideal), -value)
                )

        assert end > start
        chunk_text = text[start:end]
        measured = counts(end)
        assert all(value <= target_tokens for value in measured.values())
        inside_sentence = end < len(text) and end not in natural
        warnings = []
        if previous_split_inside_sentence:
            warnings.append("Section begins inside a sentence.")
        if inside_sentence:
            warnings.append("Section ends inside a sentence to satisfy token limits.")
            if end not in whitespace:
                warnings.append("A word was split at a Unicode character boundary.")
        if not chunk_text.strip():
            warnings.append("Whitespace-only section; preserve without rewriting.")
        chunks.append(
            DocumentChunk(
                index=len(chunks),
                start=start,
                end=end,
                text=chunk_text,
                weight=max(1, len(re.findall(r"\S+", chunk_text))),
                token_counts=measured,
                warnings=tuple(warnings),
            )
        )
        start = end
        previous_split_inside_sentence = inside_sentence

    assert "".join(chunk.text for chunk in chunks) == text
    assert all(text[chunk.start : chunk.end] == chunk.text for chunk in chunks)
    return chunks


def aggregate_chunk_scores(
    chunks: Sequence[DocumentChunk],
    original_scores: Mapping[str, Sequence[float | None]],
    rewritten_scores: Mapping[str, Sequence[float | None]],
    *,
    rewritten_texts: Sequence[str],
    token_counters: TokenCounters,
) -> dict:
    """Compare sections with fixed ORIGINAL whitespace-delimited word weights.

    For one section this preserves the detector's native score. For several it
    is a descriptive weighted mean, never a calibrated document probability.
    None means a section has no scorable content after detector preprocessing.
    Missing sections make the corresponding aggregate unavailable; their
    original weights remain in place and are never redistributed.
    """
    if not chunks or len(rewritten_texts) != len(chunks):
        raise ValueError("Provide one rewritten text for every nonempty section list")
    names = set(token_counters)
    if not names or names != set(original_scores) or names != set(rewritten_scores):
        raise ValueError(
            "Original scores, rewritten scores and token counters must match"
        )
    for scores in (original_scores, rewritten_scores):
        for values in scores.values():
            if len(values) != len(chunks) or any(
                value is not None and (not math.isfinite(value) or not 0 <= value <= 1)
                for value in values
            ):
                raise ValueError("Provide one finite score in [0, 1] per section")
    if any(chunk.weight < 1 for chunk in chunks):
        raise ValueError("Original section weights must be positive")
    total_weight = sum(chunk.weight for chunk in chunks)

    def weighted_mean(values):
        if any(value is None for value in values):
            return None
        return (
            sum(
                chunk.weight * score
                for chunk, score in zip(chunks, values, strict=True)
            )
            / total_weight
        )

    def comparison(original, rewritten, missing_reasons):
        scores = {
            "original": original,
            "rewritten": rewritten,
            "delta": (
                rewritten - original
                if original is not None and rewritten is not None
                else None
            ),
        }
        if missing_reasons:
            scores["unavailable_reasons"] = {
                **missing_reasons,
                "delta": "Both original and rewritten scores are required.",
            }
        return scores

    detectors = {}
    for name in token_counters:
        missing_reasons = {}
        for side, values in (
            ("original", original_scores[name]),
            ("rewritten", rewritten_scores[name]),
        ):
            missing = [
                chunk.index
                for chunk, score in zip(chunks, values, strict=True)
                if score is None
            ]
            if missing:
                missing_reasons[side] = (
                    f"Sections {missing} have no scorable content after detector "
                    "preprocessing. No sections were excluded or weights redistributed."
                )
        detectors[name] = comparison(
            weighted_mean(original_scores[name]),
            weighted_mean(rewritten_scores[name]),
            missing_reasons,
        )
    return {
        "method": (
            "single-section-native"
            if len(chunks) == 1
            else "original-word-weighted-section-mean"
        ),
        "score_interpretation": (
            "Classifier outputs, not calibrated authorship probabilities. "
            "For multiple sections, means use fixed original section word counts; "
            "they are descriptive section summaries, not full-document probabilities."
        ),
        "weight_definition": "max(1, original whitespace-delimited word count)",
        "total_original_weight": total_weight,
        "detectors": detectors,
        "sections": [
            {
                "index": chunk.index,
                "start": chunk.start,
                "end": chunk.end,
                "original_weight": chunk.weight,
                "warnings": list(chunk.warnings),
                "original_token_counts": _token_counts(chunk.text, token_counters),
                "rewritten_token_counts": _token_counts(rewritten, token_counters),
                "scores": {
                    name: comparison(
                        original_scores[name][position],
                        rewritten_scores[name][position],
                        {
                            side: "No scorable content after detector preprocessing."
                            for side, score in (
                                ("original", original_scores[name][position]),
                                ("rewritten", rewritten_scores[name][position]),
                            )
                            if score is None
                        },
                    )
                    for name in token_counters
                },
            }
            for position, (chunk, rewritten) in enumerate(
                zip(chunks, rewritten_texts, strict=True)
            )
        ],
    }
