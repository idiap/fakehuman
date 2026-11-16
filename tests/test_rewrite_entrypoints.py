# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: William Droz <william.droz@idiap.ch>
# SPDX-License-Identifier: GPL-3.0-only

"""Exercise explicit provider access, CLI dispatch and MCP rewriting contracts."""

import asyncio
import json
import sys

import pytest
from fastmcp import Client

from fakehuman import cli, llm_client, process, research_mcp
from fakehuman.meaning import MeaningGuard
from fakehuman.service import TextService


@pytest.fixture
def configured(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("FAKEHUMAN_OFFLINE", raising=False)
    monkeypatch.setenv("LLM_BASE_URL", "https://provider.example/v1")
    monkeypatch.setenv("LLM_API_KEY", "fixture-private-key")


@pytest.mark.parametrize("repetitions", [1, 3_000])
def test_service_rewrite_uses_shared_models_and_unique_local_runs(
    configured, monkeypatch, tmp_path, repetitions
):
    monkeypatch.setenv("FAKEHUMAN_RUNS_DIR", str(tmp_path))
    calls = []

    def rewrite(text, **options):
        calls.append((text, options))
        return {"text": "Rewritten passage.", "status": "changed"}

    monkeypatch.setattr(process, "process_text", rewrite)
    guard, detectors = object(), {}
    service = TextService(cli.load_manifest(), guard=guard, detectors=detectors)
    original = "Original passage. " * repetitions
    for _ in range(2):
        assert service.rewrite_text(original, rounds=1)["status"] == "changed"
    assert all(text == original for text, _ in calls)
    assert calls[0][1]["guard"] is guard
    assert calls[0][1]["detectors"] is detectors
    assert calls[0][1]["rounds"] == 1
    assert calls[0][1]["output_dir"].parent == tmp_path
    assert calls[0][1]["output_dir"] != calls[1][1]["output_dir"]


def test_frozen_workflow_transforms_all_sentences_in_a_long_document():
    class Guard(MeaningGuard):
        def _token_count(self, premise, hypothesis):
            return len((premise + " " + hypothesis).split())

        def _score_pairs(self, pairs):
            return [(0.95, 0.02) for _ in pairs]

    service = TextService(cli.load_manifest(), guard=Guard(), detectors={})
    original = "Therefore, the team recorded the result.\n" * 1_000
    report = service.inspect_transformation(original)
    assert report["meaning"]["scope"] == "individual_changed_sentences"
    assert not report["refused"]
    assert report["text"] == original.replace("Therefore", "Thus")
    assert service.humanify(original) == report["text"]


@pytest.mark.parametrize("offline", [False, True])
def test_service_configuration_errors_precede_processing(
    monkeypatch, tmp_path, offline
):
    monkeypatch.setenv("FAKEHUMAN_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("FAKEHUMAN_OFFLINE", "1" if offline else "0")
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.setattr(
        process, "process_text", lambda *a, **kw: pytest.fail("Must not process")
    )
    service = TextService(cli.load_manifest(), guard=object(), detectors={})
    with pytest.raises(llm_client.LLMConfigurationError):
        service.rewrite_text("A passage.")
    assert not (tmp_path / "runs").exists()


def test_cli_rewrite_dispatches_file_and_model_options(
    configured, monkeypatch, tmp_path, capsys
):
    source = tmp_path / "input.txt"
    source.write_bytes(b"A passage.\r\n")
    calls = []

    def rewrite(text, **options):
        calls.append((text, options))
        return {"text": text, "status": "unchanged", "scores": {}}

    monkeypatch.setattr(process, "process_text", rewrite)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fakehuman",
            "rewrite",
            "--input",
            str(source),
            "--output",
            str(tmp_path / "run"),
            "--rounds",
            "1",
            "--editor-model",
            "provider/editor",
            "--reviewer-model",
            "provider/reviewer",
            "--review-policy",
            "contextual",
        ],
    )
    cli.main()
    assert json.loads(capsys.readouterr().out)["text"] == "A passage.\r\n"
    assert calls[0][1]["editor_model"] == "provider/editor"
    assert calls[0][1]["reviewer_model"] == "provider/reviewer"
    assert calls[0][1]["review_policy"] == "contextual"
    assert calls[0][1]["output_dir"] == tmp_path / "run"


def test_cli_rewrite_offline_fails_without_processing(configured, monkeypatch, capsys):
    monkeypatch.setattr(
        process, "process_text", lambda *a, **kw: pytest.fail("Must not process")
    )
    monkeypatch.setattr(
        sys, "argv", ["fakehuman", "rewrite", "--offline", "--text", "A passage."]
    )
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 1
    assert "offline" in capsys.readouterr().err


def test_cli_provider_failure_keeps_report_and_returns_failure(
    configured, monkeypatch, capsys
):
    monkeypatch.setattr(
        process,
        "process_text",
        lambda text, **options: {
            "text": text,
            "status": "provider_error",
            "provider_errors": [{"type": "http_error", "http_status": 401}],
        },
    )
    monkeypatch.setattr(sys, "argv", ["fakehuman", "rewrite", "--text", "A passage."])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 1
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "provider_error"
    assert report["text"] == "A passage."


def test_mcp_rewrite_schema_dispatch_and_provider_annotations(monkeypatch):
    calls = []

    def rewrite(self, text, **options):
        calls.append((text, options))
        return {"text": "Rewritten.", "status": "changed"}

    monkeypatch.setattr(TextService, "rewrite_text", rewrite)
    server = research_mcp.make_server(cli.load_manifest(), guard=object(), detectors={})

    async def exercise():
        async with Client(server) as client:
            tools = {tool.name: tool for tool in await client.list_tools()}
            tool = tools["rewrite_text"]
            assert tool.annotations.open_world_hint
            assert not tool.annotations.read_only_hint
            result = await client.call_tool(
                "rewrite_text", {"text": "Original.", "rounds": 1}
            )
            assert result.data == {"text": "Rewritten.", "status": "changed"}
            invalid = await client.call_tool_mcp(
                "rewrite_text", {"text": "Original.", "strategy": "unknown"}
            )
            assert invalid.is_error

    asyncio.run(exercise())
    assert calls == [
        ("Original.", {"strategy": "adaptive", "rounds": 1, "input_format": "auto"})
    ]


def test_contextual_review_dispatch_and_policy_validation(configured, monkeypatch):
    calls = []
    monkeypatch.setattr(
        process,
        "process_text",
        lambda text, **options: calls.append(options) or {"text": text},
    )
    service = TextService(cli.load_manifest(), guard=object(), detectors={})
    service.rewrite_text("A passage.", review_policy="contextual")
    assert calls[0]["review_policy"] == "contextual"
    with pytest.raises(ValueError, match="review_policy"):
        service.rewrite_text("A passage.", review_policy="skip")
    with pytest.raises(ValueError, match="sentence editing"):
        service.rewrite_text(
            "A passage.",
            strategy="technical-update-experimental",
            review_policy="contextual",
        )
    assert len(calls) == 1


def test_mcp_contextual_review_and_manual_repair_dispatch(monkeypatch):
    calls = []
    monkeypatch.setattr(
        TextService,
        "rewrite_text",
        lambda self, text, **options: (
            calls.append(("rewrite", options)) or {"text": text}
        ),
    )
    monkeypatch.setattr(
        TextService,
        "repair_candidate",
        lambda self, original, candidate, findings, **options: (
            calls.append((original, candidate, findings, options))
            or {
                "text": original,
                "attempted_text": candidate,
                "review_status": "unverified",
            }
        ),
    )
    server = research_mcp.make_server(cli.load_manifest(), guard=object(), detectors={})
    findings = [{"severity": "material", "category": "omission"}]

    async def exercise():
        async with Client(server) as client:
            tools = {tool.name: tool for tool in await client.list_tools()}
            repair = tools["repair_candidate"]
            assert repair.annotations.open_world_hint
            assert not repair.annotations.read_only_hint
            assert not repair.annotations.idempotent_hint
            await client.call_tool(
                "rewrite_text", {"text": "Original.", "review_policy": "contextual"}
            )
            result = await client.call_tool(
                "repair_candidate",
                {"original": "Original.", "candidate": "Draft.", "findings": findings},
            )
            assert result.data["text"] == "Original."
            assert result.data["review_status"] == "unverified"
            invalid = await client.call_tool_mcp(
                "rewrite_text", {"text": "Original.", "review_policy": "skip"}
            )
            assert invalid.is_error

    asyncio.run(exercise())
    assert calls == [
        (
            "rewrite",
            {
                "strategy": "adaptive",
                "rounds": 3,
                "input_format": "auto",
                "review_policy": "contextual",
            },
        ),
        ("Original.", "Draft.", findings, {"input_format": "auto"}),
    ]
