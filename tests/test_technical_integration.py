# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: William Droz <william.droz@idiap.ch>
# SPDX-License-Identifier: GPL-3.0-only

"""Offline coverage of the public text processor's experimental strategy route."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from fakehuman import technical_pipeline as pipeline
from fakehuman.llm_client import LLMConfigurationError
from fakehuman.process import process_text

STRATEGY = "technical-update-experimental"
SOURCE = "The export keeps all sessions. Authors can review the history.\n"


def catalog_for(*models):
    return {
        model: {"id": model, "supported_parameters": ["response_format", "max_tokens"]}
        for model in models
    }


def forbidden(*args, **kwargs):
    raise AssertionError("No provider, catalog network, or model load expected")


@pytest.mark.parametrize("models", [None, ["test/chosen-generator"]])
def test_public_route_selects_requested_or_default_generators_and_default_reviewers(
    tmp_path, monkeypatch, models
):
    chosen = list(pipeline.DEFAULT_GENERATORS) if models is None else models
    catalog = catalog_for(
        *chosen, *pipeline.DEFAULT_REVIEWERS, "gpt-6-luna", "test/unused"
    )
    loaded, dispatched = [], []

    def load(directory, *, timeout_seconds):
        loaded.append((directory, timeout_seconds))
        return catalog

    def process(text, **kwargs):
        dispatched.append((text, kwargs))
        return {"sentinel": True}

    monkeypatch.setattr(pipeline, "load_catalog", load)
    monkeypatch.setattr(pipeline, "process_technical_text", process)
    deps = {
        name: object() for name in ("detectors", "token_counters", "call", "progress")
    }
    result = process_text(
        SOURCE,
        output_dir=tmp_path,
        strategy=STRATEGY,
        models=models,
        workers=1,
        request_timeout=37,
        detectors=deps["detectors"],
        token_counters=deps["token_counters"],
        call=deps["call"],
        progress=deps["progress"],
    )
    assert result == {"sentinel": True}
    assert loaded == [(tmp_path, 37)]
    text, kwargs = dispatched[0]
    assert text == SOURCE
    assert kwargs["models"] == chosen
    assert kwargs["reviewers"] == list(pipeline.DEFAULT_REVIEWERS)
    assert set(kwargs["catalog"]) == set(chosen) | set(pipeline.DEFAULT_REVIEWERS)
    assert kwargs["output_dir"] == tmp_path
    assert kwargs["workers"] == 1
    assert kwargs["request_timeout"] == 37
    for name, dependency in deps.items():
        assert kwargs[name] is dependency


def test_public_route_resolves_catalog_aliases_for_default_technical_models(
    tmp_path, monkeypatch
):
    luna, deepseek = "gpt-6-luna", "deepseek/deepseek-v4.1-flash"
    reviewers = [deepseek, luna, "gemini-3.8-flash", "anthropic/claude-haiku-4.5"]
    catalog = catalog_for(*reviewers, "test/unused")
    monkeypatch.setattr(pipeline, "load_catalog", forbidden)
    monkeypatch.setattr(pipeline, "process_technical_text", lambda text, **kw: kw)

    result = process_text(
        SOURCE,
        output_dir=tmp_path,
        strategy=STRATEGY,
        model_catalog=catalog,
        call=forbidden,
    )
    assert result["models"] == [luna, deepseek]
    assert result["reviewers"] == reviewers
    assert result["catalog"] == {model: catalog[model] for model in reviewers}

    with pytest.raises(LLMConfigurationError, match="openai/gpt-6-luna"):
        process_text(
            SOURCE,
            output_dir=tmp_path,
            strategy=STRATEGY,
            models=["openai/gpt-6-luna"],
            model_catalog=catalog,
            call=forbidden,
        )


def test_public_route_honors_explicit_reviewers_and_avoids_catalog_fetch(
    tmp_path, monkeypatch
):
    models, reviewers = ["test/generator"], ["test/a", "test/b", "test/c"]
    catalog = catalog_for(*models, *reviewers)
    monkeypatch.setattr(pipeline, "load_catalog", forbidden)
    seen = []

    def process(text, **kwargs):
        seen.append(kwargs)
        return {"text": text}

    monkeypatch.setattr(pipeline, "process_technical_text", process)
    result = process_text(
        SOURCE,
        output_dir=tmp_path,
        strategy=STRATEGY,
        models=models,
        reviewers=reviewers,
        model_catalog=catalog,
        call=forbidden,
    )
    assert result["text"] == SOURCE
    assert seen[0]["reviewers"] == reviewers
    assert seen[0]["catalog"] == catalog


@pytest.mark.parametrize("missing_role", ["generator", "reviewer"])
def test_missing_catalog_model_fails_before_dispatch(
    tmp_path, monkeypatch, missing_role
):
    models, reviewers = ["test/generator"], ["test/a", "test/b", "test/c"]
    catalog = catalog_for(*models, *reviewers)
    missing = models[0] if missing_role == "generator" else reviewers[0]
    del catalog[missing]
    monkeypatch.setattr(pipeline, "process_technical_text", forbidden)
    monkeypatch.setattr(pipeline, "load_catalog", forbidden)
    with pytest.raises(ValueError, match="Missing .*model metadata: " + missing):
        process_text(
            SOURCE,
            output_dir=tmp_path,
            strategy=STRATEGY,
            models=models,
            reviewers=reviewers,
            model_catalog=catalog,
            call=forbidden,
        )
    assert list(tmp_path.iterdir()) == []


def test_unknown_strategy_fails_before_loading_dependencies(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "load_catalog", forbidden)
    with pytest.raises(ValueError, match="Unknown text processing strategy"):
        process_text(
            SOURCE, output_dir=tmp_path, strategy="technical-maybe", call=forbidden
        )


@pytest.mark.parametrize(
    "override", [{"models": []}, {"reviewers": []}, {"model_catalog": {}}]
)
def test_model_overrides_require_experimental_opt_in(tmp_path, override):
    with pytest.raises(ValueError, match="Model overrides require the experimental"):
        process_text(SOURCE, output_dir=tmp_path, call=forbidden, **override)
    assert list(tmp_path.iterdir()) == []


def test_catalog_resume_never_contacts_provider(tmp_path, monkeypatch):
    catalog = catalog_for(*pipeline.DEFAULT_GENERATORS, *pipeline.DEFAULT_REVIEWERS)
    (tmp_path / "protocol.json").write_text(
        json.dumps({"strategy": "technical_update", "catalog": catalog})
    )
    monkeypatch.setattr(pipeline, "DeadlineClient", forbidden)
    assert pipeline.load_catalog(tmp_path) == catalog


def test_saved_other_strategy_catalog_is_rejected_without_network(
    tmp_path, monkeypatch
):
    (tmp_path / "protocol.json").write_text(json.dumps({"strategy": "adaptive"}))
    monkeypatch.setattr(pipeline, "DeadlineClient", forbidden)
    with pytest.raises(ValueError, match="belongs to another strategy"):
        pipeline.load_catalog(tmp_path)


def test_fresh_catalog_uses_configured_provider_metadata(tmp_path, monkeypatch):
    calls = []
    metadata = {"id": "test/model", "supported_parameters": ["seed"]}

    class Provider:
        def __init__(self, *, timeout_seconds):
            calls.append(("deadline", timeout_seconds))

        def models(self):
            calls.append("models")
            return {"test/model": metadata}

    monkeypatch.setattr(pipeline, "DeadlineClient", Provider)
    assert pipeline.load_catalog(tmp_path, timeout_seconds=7) == {
        "test/model": metadata
    }
    assert calls == [("deadline", 7), "models"]


def test_catalog_deadline_failure_leaves_no_run_files(tmp_path, monkeypatch):
    deadlines = []

    class ExpiredProvider:
        def __init__(self, *, timeout_seconds):
            deadlines.append(timeout_seconds)

        def models(self):
            raise RuntimeError("Provider model catalog request exceeded its deadline")

    monkeypatch.setattr(pipeline, "DeadlineClient", ExpiredProvider)
    with pytest.raises(RuntimeError, match="exceeded its deadline"):
        pipeline.load_catalog(tmp_path, timeout_seconds=0.25)
    assert deadlines == [0.25]
    assert list(tmp_path.iterdir()) == []


def test_source_snapshots_use_package_files_outside_a_repository(tmp_path, monkeypatch):
    package = tmp_path / "installed" / "fakehuman"
    package.mkdir(parents=True)
    content = b"# Installed source\n"
    (package / "technical_pipeline.py").write_bytes(content)
    monkeypatch.setattr(pipeline, "PACKAGE_DIR", package)
    hashes = pipeline.source_hashes()
    assert set(hashes) == {"technical_pipeline.py"}
    destination = tmp_path / "run" / "source_snapshot"
    pipeline.snapshot_sources(destination, hashes)
    assert (destination / "technical_pipeline.py").read_bytes() == content
    (package / "technical_pipeline.py").write_bytes(b"# Changed\n")
    with pytest.raises(ValueError, match="Source changed"):
        pipeline.snapshot_sources(destination, hashes)


def test_snapshot_paths_cannot_escape_the_package(tmp_path):
    with pytest.raises(ValueError, match="Invalid source snapshot path"):
        pipeline.snapshot_sources(tmp_path, {"../outside.py": "unused"})


class Detector:
    def __init__(self, name):
        self.name = name
        self.calls = []

    def metadata(self):
        return {"model_id": "offline:" + self.name, "revision": "integration-test"}

    def score(self, texts):
        self.calls.append(list(texts))
        return [0.2 if "retains" in text else 0.8 for text in texts]


def test_public_processor_exports_experimental_result_and_resumes_offline(
    tmp_path, monkeypatch
):
    generator = pipeline.DEFAULT_GENERATORS[0]
    catalog = catalog_for(generator, *pipeline.DEFAULT_REVIEWERS)
    requests, window_calls = [], []
    detectors = {name: Detector(name) for name in pipeline.NAMES}
    counters = {name: lambda text: len(text.split()) + 2 for name in pipeline.NAMES}

    def call(body):
        requests.append(body)
        payload = json.loads(body["messages"][1]["content"])
        if "sections" in payload:
            answer = {
                "sections": [
                    {"id": row["id"], "text": row["text"].replace("keeps", "retains")}
                    for row in payload["sections"]
                ]
            }
        else:
            answer = {"verdict": "accept", "issues": []}
        return {
            "raw_response": json.dumps(answer),
            "finish_reason": "stop",
            "usage": {},
        }

    def windows(text, **kwargs):
        window_calls.append(text)
        return {
            "method": "offline-test",
            "detectors": {
                name: {"available": True, "mean": 0.2 if "retains" in text else 0.8}
                for name in pipeline.NAMES
            },
        }

    monkeypatch.setattr(pipeline, "source_hashes", lambda: {})
    monkeypatch.setattr(pipeline, "score_document_windows", windows)
    monkeypatch.setattr(pipeline, "DeadlineClient", forbidden)
    result = process_text(
        SOURCE,
        output_dir=tmp_path,
        strategy=STRATEGY,
        models=[generator],
        model_catalog=catalog,
        workers=1,
        call=call,
        detectors=detectors,
        token_counters=counters,
    )
    assert result["text"] == SOURCE.replace("keeps", "retains")
    assert result["experimental"] is True
    assert result["promotion_gate_passed"] is False
    assert "gate failed" in result["validation_status"]
    assert result["human_review_performed"] is False
    assert result["pangram_submissions"] == 0
    assert result["selected_model"] == generator
    assert [body["model"] for body in requests] == [
        generator,
        *[model for model in pipeline.DEFAULT_REVIEWERS if model != generator][:2],
    ]
    for scores in result["scores"]["detectors"].values():
        assert scores["original"] == 0.8
        assert scores["rewritten"] == 0.2
        assert scores["delta"] == pytest.approx(-0.6)
    for path in result["files"].values():
        assert Path(path).is_file()
    assert (tmp_path / "input.txt").read_bytes() == SOURCE.encode()
    assert (tmp_path / "output.txt").read_bytes() == result["text"].encode()
    assert "-The export keeps" in (tmp_path / "changes.diff").read_text()
    assert json.loads((tmp_path / "result.json").read_text()) == result
    counts = (len(window_calls), [len(model.calls) for model in detectors.values()])
    resumed = process_text(
        SOURCE,
        output_dir=tmp_path,
        strategy=STRATEGY,
        models=[generator],
        workers=1,
        call=forbidden,
        detectors=detectors,
        token_counters=counters,
    )
    assert resumed == result
    assert counts == (
        len(window_calls),
        [len(model.calls) for model in detectors.values()],
    )


def test_cli_help_exposes_opt_in_strategy_and_model_choice_without_api():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-m", "fakehuman", "rewrite", "--help"],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    assert "--strategy {adaptive,technical-update-experimental}" in result.stdout
    assert "--model MODEL" in result.stdout
