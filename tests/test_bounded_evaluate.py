# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: William Droz <william.droz@idiap.ch>
# SPDX-License-Identifier: GPL-3.0-only

import pytest

from fakehuman.bounded_evaluate import check_document_pair, screen_candidate
from fakehuman.meaning import MeaningGuard


class PassingGuard:
    def __init__(self):
        self.pairs = []

    def check_pairs(self, pairs):
        self.pairs.extend(pairs)
        return [{"passed": True, "reason": "bidirectional_entailment"} for _ in pairs]

    def metadata(self):
        return {"model_id": "injected-test-guard"}


class RejectingGuard(PassingGuard):
    def check_pairs(self, pairs):
        self.pairs.extend(pairs)
        return [{"passed": False, "reason": "low_entailment"} for _ in pairs]


def test_screen_requires_full_and_each_changed_sentence():
    guard = PassingGuard()
    result = screen_candidate(
        guard,
        "The initial measured value seems fairly stable. A second result remains unchanged.",
        "The value seems fairly stable. A second result remains unchanged.",
    )
    assert result["passed"]
    assert len(guard.pairs) == 2
    assert guard.pairs[0][0].startswith("The initial measured")
    assert guard.pairs[1] == (
        "The initial measured value seems fairly stable.",
        "The value seems fairly stable.",
    )


def test_sentence_alignment_and_nli_errors_fail_closed():
    mismatch = screen_candidate(
        PassingGuard(),
        "A short original sentence.",
        "A short original sentence. Another one.",
    )
    assert not mismatch["passed"] and mismatch["reason"] == "sentence_alignment_failed"
    rejected = screen_candidate(
        RejectingGuard(),
        "The old wording seems fairly stable.",
        "The new wording seems fairly stable.",
    )
    assert not rejected["passed"]


@pytest.mark.parametrize("reject", [False, True])
def test_dense_edits_require_all_individual_sentence_checks(reject):
    class Guard(MeaningGuard):
        def _token_count(self, premise, hypothesis):
            return 30 * len((premise + " " + hypothesis).split())

        def _score_pairs(self, pairs):
            return [
                (0.89, 0.2 if reject and "delayed" in right else 0.02)
                for _, right in pairs
            ]

    before = "The first report was filed. The second report was late."
    after = "The first report was submitted. The second report was delayed."
    guard = Guard()
    assert guard.check_pairs([(before, after)])[0]["reason"] == "overlength"

    local = check_document_pair(guard, before, after)
    assert local["scope"] == "individual_changed_sentences"
    assert local["passed"] is not reject
    assert local["reason"] == (
        "contradiction" if reject else "bidirectional_entailment"
    )
    assert not screen_candidate(guard, before, after)["passed"]
    final = screen_candidate(guard, before, after, allow_overlength=True)
    assert final["passed"] is not reject
    assert final["scope"] == "individual_changed_sentences"
    assert len(final["sentences"]) == 2


def test_overlength_fallback_refuses_misaligned_changed_blocks():
    class OverflowGuard:
        def check_pairs(self, pairs):
            assert len(pairs) == 1
            return [
                {"passed": False, "reason": "overlength", "scope": "changed_sentences"}
            ]

    before = "First report. Second report. The audit began. Third report."
    after = "First summary. The audit began. Second summary. Third summary."
    result = check_document_pair(OverflowGuard(), before, after)
    assert result["reason"] == "overlength"
    assert not result["passed"]
