# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: William Droz <william.droz@idiap.ch>
# SPDX-License-Identifier: GPL-3.0-only

"""Summarize recorded provider costs without treating missing usage as free."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable
from pathlib import Path

_COST_FIELDS = ("gateway_cost_usd", "upstream_cost_usd")


def _amount(value: object) -> float | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    try:
        amount = float(value)
    except OverflowError:
        return None
    if math.isfinite(amount) and amount >= 0:
        return amount
    return None


def summarize_responses(responses: Iterable[dict]) -> dict:
    """Count each supplied request once, including refused and failed requests.

    Gateway ``usage.cost`` and upstream inference cost are separate provider
    reports. They must not be added together: their billing scopes can overlap.
    """
    values = {field: [] for field in _COST_FIELDS}
    request_count = completion_unknown = 0
    for response in responses:
        if response.get("api_call_skipped") is True:
            continue
        request_count += 1
        completion_unknown += response.get("provider_completion_unknown") is True
        usage = response.get("usage")
        usage = usage if isinstance(usage, dict) else {}
        details = usage.get("cost_details")
        details = details if isinstance(details, dict) else {}
        for field, value in (
            ("gateway_cost_usd", usage.get("cost")),
            ("upstream_cost_usd", details.get("upstream_inference_cost")),
        ):
            amount = _amount(value)
            if amount is not None:
                values[field].append(amount)
    return {
        "currency": "USD",
        "request_count": request_count,
        "completion_unknown_requests": completion_unknown,
        **{
            field: {
                "total": math.fsum(amounts) if amounts or not request_count else None,
                "reported_requests": len(amounts),
                "missing_requests": request_count - len(amounts),
            }
            for field, amounts in values.items()
        },
    }


def summarize_run_costs(directory: Path) -> dict:
    """Read request checkpoints once, excluding their candidate/result mirrors.

    Resuming a run therefore preserves its recorded total without counting the
    same requests again. Costs from requests interrupted before checkpointing
    are unavailable, as is any provider usage omitted from a response.
    """

    def responses():
        for path in sorted(Path(directory).rglob("*.json")):
            stored = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(stored, dict) or set(stored) != {
                "binding",
                "payload",
                "payload_sha256",
            }:
                continue
            binding, payload = stored["binding"], stored["payload"]
            if (
                not isinstance(binding, dict)
                or not isinstance(binding.get("request"), dict)
                or not isinstance(payload, dict)
            ):
                continue
            serialized = json.dumps(
                payload, ensure_ascii=False, sort_keys=True, allow_nan=False
            )
            if (
                hashlib.sha256(serialized.encode("utf-8")).hexdigest()
                != stored["payload_sha256"]
            ):
                raise ValueError(f"Cost checkpoint integrity mismatch: {path}")
            yield payload

    return summarize_responses(responses())


def combine_cost_summaries(summaries: Iterable[dict]) -> dict:
    """Combine disjoint run or repair summaries, preserving partial coverage."""
    summaries = list(summaries)
    if any(summary["currency"] != "USD" for summary in summaries):
        raise ValueError("Cost summaries must use USD")
    request_count = sum(summary["request_count"] for summary in summaries)
    result = {
        "currency": "USD",
        "request_count": request_count,
        "completion_unknown_requests": sum(
            summary["completion_unknown_requests"] for summary in summaries
        ),
    }
    for field in _COST_FIELDS:
        reported = sum(summary[field]["reported_requests"] for summary in summaries)
        result[field] = {
            "total": math.fsum(
                summary[field]["total"]
                for summary in summaries
                if summary[field]["total"] is not None
            )
            if reported or not request_count
            else None,
            "reported_requests": reported,
            "missing_requests": request_count - reported,
        }
    return result
