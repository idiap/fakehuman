# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: William Droz <william.droz@idiap.ch>
# SPDX-License-Identifier: GPL-3.0-only

import json
import subprocess
import sys

import pytest

from fakehuman import adaptive_transport as transport


@pytest.fixture(autouse=True)
def configured_provider(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "http://127.0.0.1:9999/v1")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.delenv("FAKEHUMAN_OFFLINE", raising=False)


def test_passes_only_request_to_child_and_preserves_response(monkeypatch, tmp_path):
    response = {"raw_response": "real output", "usage": {"total_tokens": 12}}
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, json.dumps(response), "")

    monkeypatch.setattr(transport.subprocess, "run", run)
    assert (
        transport.DeadlineClient(root=tmp_path).call({"model": "example"}) == response
    )
    assert len(calls) == 1
    assert calls[0][1]["timeout"] == 240
    assert json.loads(calls[0][1]["input"]) == {"model": "example"}
    assert "cwd" not in calls[0][1]
    assert calls[0][0] == [sys.executable, "-m", "fakehuman.llm_client"]
    assert "test-key" not in json.dumps(calls)


@pytest.mark.parametrize("failure", ["timeout", "nonzero", "malformed", "array"])
def test_failure_is_not_retried_and_does_not_leak_output(
    monkeypatch, tmp_path, failure
):
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if failure == "timeout":
            raise subprocess.TimeoutExpired(
                command, 240, output="PRIVATE", stderr="SECRET"
            )
        return subprocess.CompletedProcess(
            command,
            1 if failure == "nonzero" else 0,
            "[]" if failure == "array" else "PRIVATE",
            "SECRET",
        )

    monkeypatch.setattr(transport.subprocess, "run", run)
    result = transport.DeadlineClient(root=tmp_path).call({"model": "example"})
    assert len(calls) == 1
    assert result["finish_reason"] == "transport_error"
    assert result["usage"] is None
    assert result["provider_completion_unknown"] is True
    assert "PRIVATE" not in json.dumps(result)
    assert "SECRET" not in json.dumps(result)


def test_real_child_is_terminated_at_deadline(monkeypatch, tmp_path):
    real_run = transport.subprocess.run

    def sleeping_child(command, **kwargs):
        return real_run([command[0], "-c", "import time; time.sleep(30)"], **kwargs)

    monkeypatch.setattr(transport.subprocess, "run", sleeping_child)
    result = transport.DeadlineClient(root=tmp_path, timeout_seconds=0.05).call({})
    assert result["transport_errors"][0]["type"] == "wall_clock_deadline"
    assert result["latency_seconds"] < 5


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf"), True, "1"])
def test_invalid_deadlines_fail_before_spawning(timeout):
    with pytest.raises(ValueError, match="positive and finite"):
        transport.DeadlineClient(timeout_seconds=timeout)


def test_model_catalog_uses_hard_deadline_and_sanitizes_timeout(monkeypatch):
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        raise subprocess.TimeoutExpired(
            command, 0.05, output="PRIVATE", stderr="SECRET"
        )

    monkeypatch.setattr(transport.subprocess, "run", run)
    with pytest.raises(transport.LLMModelCatalogError) as error:
        transport.DeadlineClient(timeout_seconds=0.05).models()
    assert len(calls) == 1
    assert calls[0][0][-1] == "--models"
    assert calls[0][1]["timeout"] == 0.05
    assert "PRIVATE" not in str(error.value)
    assert "SECRET" not in str(error.value)
