# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: William Droz <william.droz@idiap.ch>
# SPDX-License-Identifier: GPL-3.0-only

import json
from typing import Any, cast

import pytest

from fakehuman.meaning import (
    CONTRADICTION_THRESHOLD,
    ENTAILMENT_THRESHOLD,
    MODEL_ID,
    REVISION,
    MeaningGuard,
)


class StubGuard(MeaningGuard):
    def __init__(self, scores=None, *, max_tokens=512):
        super().__init__()
        self.scores = scores or {}
        self.max_tokens = max_tokens
        self.scored = []

    def _token_count(self, premise, hypothesis):
        # Force multi-sentence test inputs through the chunk alignment path.
        if self.max_tokens == 1:
            long_markers = (
                "Unchanged first sentence.",
                "Removed sentence.",
                "A long single sentence.",
                "A changed single sentence.",
            )
            return 600 if any(marker in premise for marker in long_markers) else 10
        return len((premise + " " + hypothesis).split())

    def _score_pairs(self, pairs):
        self.scored.extend(pairs)
        return [self.scores.get(pair, (0.95, 0.02)) for pair in pairs]


def test_identity_skips_model_and_returns_fixed_result():
    guard = StubGuard()
    result = guard.check_pairs([("same text", "same text")])
    assert result == [
        {
            "passed": True,
            "reason": "identical",
            "min_entailment": 1.0,
            "max_contradiction": 0.0,
            "scope": "identity",
        }
    ]
    assert guard.scored == []


def test_empty_input_fails_closed_without_running_the_model():
    guard = StubGuard()
    result = guard.check_pairs([("some text", " ")])[0]
    assert result["passed"] is False
    assert result["reason"] == "empty_input"
    assert guard.scored == []


def test_requires_both_directions_to_meet_both_fixed_thresholds():
    before, after = "The signal was stable.", "The signal remained stable."
    guard = StubGuard(
        {
            (before, after): (0.92, 0.03),
            (after, before): (0.79, 0.01),
        }
    )
    result = guard.check_pairs([(before, after)])[0]
    assert result["passed"] is False
    assert result["reason"] == "low_entailment"
    assert result["min_entailment"] == 0.79
    assert result["max_contradiction"] == 0.03
    assert len(guard.scored) == 2
    assert ENTAILMENT_THRESHOLD == 0.8
    assert CONTRADICTION_THRESHOLD == 0.1


def test_contradiction_threshold_fails_even_when_entailment_is_high():
    before, after = "The signal was stable.", "The signal remained stable."
    guard = StubGuard(
        {
            (before, after): (0.95, 0.11),
            (after, before): (0.95, 0.02),
        }
    )
    result = guard.check_pairs([(before, after)])[0]
    assert result["passed"] is False
    assert result["reason"] == "contradiction"
    assert result["max_contradiction"] == 0.11


def test_long_pair_checks_changed_aligned_sentences_only_without_truncation():
    guard = StubGuard(max_tokens=1)
    result = guard.check_pairs(
        [
            (
                "Unchanged first sentence. Old conclusion.",
                "Unchanged first sentence. New conclusion.",
            )
        ]
    )[0]
    assert result["passed"] is True
    assert result["scope"] == "changed_sentences"
    assert guard.scored == [
        ("Old conclusion.", "New conclusion."),
        ("New conclusion.", "Old conclusion."),
    ]


def test_pair_length_preflight_counts_all_tokens_quietly():
    class Tokenizer:
        def encode(self, premise, hypothesis, **kwargs):
            assert (premise, hypothesis) == ("Premise.", "Hypothesis.")
            assert kwargs == {
                "add_special_tokens": True,
                "truncation": False,
                "verbose": False,
            }
            return range(731)

    guard = MeaningGuard(tokenizer=Tokenizer(), model=object())
    assert guard._token_count("Premise.", "Hypothesis.") == 731


def test_long_insertions_and_unbounded_changed_sentences_fail_closed():
    deleted = StubGuard(max_tokens=1).check_pairs(
        [("Unchanged. Removed sentence.", "Unchanged.")]
    )[0]
    assert deleted["passed"] is False
    assert deleted["reason"] == "alignment_failed"

    long_span = StubGuard(max_tokens=1).check_pairs(
        [("A long single sentence.", "A changed single sentence.")]
    )[0]
    assert long_span["passed"] is False
    assert long_span["reason"] == "overlength"


def test_pair_results_are_cached_symmetrically():
    before, after = "A stable result.", "A consistent result."
    guard = StubGuard()
    first, second = guard.check_pairs([(before, after), (after, before)])
    assert first == second
    assert len(guard.scored) == 2


def test_input_validation_and_empty_batch():
    guard = StubGuard()
    assert guard.check_pairs([]) == []
    with pytest.raises(TypeError, match="list"):
        guard.check_pairs(cast(Any, ("before", "after")))
    with pytest.raises(TypeError, match="tuple"):
        guard.check_pairs(cast(Any, [["before", "after"]]))


def test_metadata_pins_model_and_documents_limits_without_loading():
    metadata = MeaningGuard().metadata()
    assert metadata["model_id"] == MODEL_ID == "cross-encoder/nli-deberta-v3-small"
    assert metadata["revision"] == REVISION
    assert len(REVISION) == 40
    assert metadata["license"] == "Apache-2.0"
    assert metadata["trust_remote_code"] is False
    assert metadata["use_safetensors"] is True
    assert metadata["entailment_threshold"] == 0.8
    assert metadata["contradiction_threshold"] == 0.1
    assert metadata["loaded"] is False
    assert "fail closed" in metadata["limitations"]
    json.dumps(metadata)
