# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: William Droz <william.droz@idiap.ch>
# SPDX-License-Identifier: GPL-3.0-only

"""Recorded billing coverage, checkpoint accounting and summary composition."""

import json

import pytest

from fakehuman.adaptive import _checkpoint
from fakehuman.costs import (
    combine_cost_summaries,
    summarize_responses,
    summarize_run_costs,
)


def response(gateway=None, upstream=None, **kwargs):
    return {
        "finish_reason": "stop",
        "usage": {
            "cost": gateway,
            "cost_details": {"upstream_inference_cost": upstream},
        },
        **kwargs,
    }


def test_zero_gateway_and_nonzero_upstream_remain_separate():
    summary = summarize_responses([response(0, 0.0000929), response(0, 0.0001832)])
    assert summary["currency"] == "USD"
    assert summary["request_count"] == 2
    assert summary["gateway_cost_usd"] == {
        "total": 0.0,
        "reported_requests": 2,
        "missing_requests": 0,
    }
    assert summary["upstream_cost_usd"]["total"] == pytest.approx(0.0002761)


def test_missing_usage_is_unknown_and_partial_values_keep_their_coverage():
    summary = summarize_responses(
        [response(0.1), {"usage": None, "provider_completion_unknown": True}, {}]
    )
    assert summary["request_count"] == 3
    assert summary["completion_unknown_requests"] == 1
    assert summary["gateway_cost_usd"] == {
        "total": 0.1,
        "reported_requests": 1,
        "missing_requests": 2,
    }
    assert summary["upstream_cost_usd"] == {
        "total": None,
        "reported_requests": 0,
        "missing_requests": 3,
    }


@pytest.mark.parametrize(
    "value", [None, True, False, "0.5", -0.1, float("nan"), float("inf"), 10**400]
)
def test_invalid_cost_values_are_missing(value):
    summary = summarize_responses([response(value, value)])
    assert summary["gateway_cost_usd"]["total"] is None
    assert summary["upstream_cost_usd"]["missing_requests"] == 1


def test_skipped_calls_do_not_count_as_billable_requests():
    assert summarize_responses([response(api_call_skipped=True)]) == {
        "currency": "USD",
        "request_count": 0,
        "completion_unknown_requests": 0,
        "gateway_cost_usd": {
            "total": 0.0,
            "reported_requests": 0,
            "missing_requests": 0,
        },
        "upstream_cost_usd": {
            "total": 0.0,
            "reported_requests": 0,
            "missing_requests": 0,
        },
    }


def test_only_request_checkpoints_count_not_mirrored_candidates_and_results(tmp_path):
    generation = response(0, 0.05)
    review = response(0, 0.03)
    for name, payload in (
        ("search/section/round_1_candidate_1/generation.json", generation),
        ("search/section/round_1_candidate_1/review.json", review),
        # Malformed responses still represent a request with missing usage.
        ("search/section/round_1_candidate_2/generation.json", {}),
        (
            "final_context_response.json",
            {"finish_reason": "transport_error", "provider_completion_unknown": True},
        ),
    ):
        _checkpoint(tmp_path / name, {"request": {"model": "fixture"}}, lambda: payload)
    mirrored = {"generation": generation, "review": {"response": review}}
    _checkpoint(tmp_path / "candidate.json", {}, lambda: mirrored)
    _checkpoint(tmp_path / "search/section/result.json", {}, lambda: [mirrored])
    (tmp_path / "result.json").write_text(json.dumps(mirrored))
    summary = summarize_run_costs(tmp_path)
    assert summary["request_count"] == 4
    assert summary["completion_unknown_requests"] == 1
    assert summary["upstream_cost_usd"] == {
        "total": 0.08,
        "reported_requests": 2,
        "missing_requests": 2,
    }
    assert summarize_run_costs(tmp_path) == summary


def test_corrupt_request_checkpoint_cannot_supply_costs(tmp_path):
    path = tmp_path / "generation.json"
    _checkpoint(path, {"request": {}}, lambda: response(0.01))
    stored = json.loads(path.read_text())
    stored["payload"]["usage"]["cost"] = 1000
    path.write_text(json.dumps(stored))
    with pytest.raises(ValueError, match="Cost checkpoint integrity mismatch"):
        summarize_run_costs(tmp_path)


def test_combining_stages_and_repairs_preserves_unknown_and_partial_costs():
    known = summarize_responses([response(0, 0.06)])
    unknown = summarize_responses([{"provider_completion_unknown": True}])
    local = summarize_responses([])
    summary = combine_cost_summaries([known, unknown, local])
    assert summary["request_count"] == 2
    assert summary["completion_unknown_requests"] == 1
    assert summary["gateway_cost_usd"] == {
        "total": 0.0,
        "reported_requests": 1,
        "missing_requests": 1,
    }
    assert summary["upstream_cost_usd"]["total"] == 0.06
    assert combine_cost_summaries([unknown])["gateway_cost_usd"]["total"] is None
    assert combine_cost_summaries([]) == local


def test_combining_different_currencies_is_rejected():
    summary = {**summarize_responses([]), "currency": "EUR"}
    with pytest.raises(ValueError, match="USD"):
        combine_cost_summaries([summary])
