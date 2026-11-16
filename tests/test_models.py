# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: William Droz <william.droz@idiap.ch>
# SPDX-License-Identifier: GPL-3.0-only

"""Verify pinned prefetch files and CLI offline setup without downloading weights."""

import json
import os
import subprocess
import sys

import pytest

from fakehuman import cli, models
from fakehuman.detectors import DEFAULT_DETECTOR_NAMES, DETECTOR_SPECS
from fakehuman.meaning import MODEL_ID, REVISION


def test_prefetch_uses_the_runtime_revisions_and_exact_assets(monkeypatch, tmp_path):
    import huggingface_hub

    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        return str(tmp_path / kwargs["revision"] / kwargs["filename"])

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    report = models.download_models(offline=True)
    expected_revisions = {
        MODEL_ID: REVISION,
        **{
            DETECTOR_SPECS[name].model_id: DETECTOR_SPECS[name].revision
            for name in DEFAULT_DETECTOR_NAMES
        },
    }
    assert {call["repo_id"]: call["revision"] for call in calls} == expected_revisions
    assert all(call["local_files_only"] for call in calls)
    assert all(not call["filename"].endswith((".bin", ".py")) for call in calls)
    assert len(report) == 3
    for model in models.model_assets():
        assert {
            call["filename"] for call in calls if call["repo_id"] == model["model_id"]
        } == set(model["files"])
        assert "model.safetensors" in model["files"]
        assert "tokenizer.json" in model["files"]
    assert all(item["files"] == 7 for item in report)
    assert report[0]["cache_path"] == str(tmp_path / REVISION)


def test_missing_offline_file_fails_with_recovery_instructions(monkeypatch):
    import huggingface_hub
    from huggingface_hub.errors import LocalEntryNotFoundError

    files = []

    def download(**kwargs):
        files.append(kwargs["filename"])
        if kwargs["filename"] == "model.safetensors":
            raise LocalEntryNotFoundError("absent")
        return "/cache/" + kwargs["filename"]

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    with pytest.raises(RuntimeError, match="model.safetensors.*network access first"):
        models.download_models(offline=True)
    assert len(files) == 5


@pytest.mark.parametrize(
    "arguments",
    [
        ["--offline", "download-models"],
        ["download-models", "--offline"],
    ],
)
def test_cli_offline_is_set_before_download_and_does_not_start_server(
    monkeypatch, capsys, tmp_path, arguments
):
    def download(*, offline):
        assert offline
        assert os.environ["HF_HUB_OFFLINE"] == "1"
        assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
        assert os.environ["FASTMCP_CHECK_FOR_UPDATES"] == "off"
        return [{"name": "fixture"}]

    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "FASTMCP_CHECK_FOR_UPDATES"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["fakehuman", *arguments])
    monkeypatch.setattr(models, "download_models", download)
    cli.main()
    assert json.loads(capsys.readouterr().out) == {"models": [{"name": "fixture"}]}


def test_prefetch_does_not_import_torch_or_transformers():
    code = """
import sys
import huggingface_hub
from fakehuman.models import download_models
huggingface_hub.hf_hub_download = lambda **kw: '/cache/' + kw['filename']
download_models()
assert 'torch' not in sys.modules
assert 'transformers' not in sys.modules
"""
    subprocess.run(
        [sys.executable, "-c", code], check=True, capture_output=True, timeout=15
    )


def test_cli_offline_download_fails_cleanly_with_an_empty_cache(tmp_path):
    env = dict(os.environ, HF_HUB_CACHE=str(tmp_path / "empty cache"))
    result = subprocess.run(
        [sys.executable, "-m", "fakehuman", "download-models", "--offline"],
        env=env,
        text=True,
        capture_output=True,
        timeout=15,
    )
    assert result.returncode == 1
    assert "network access first" in result.stderr
    assert "Traceback" not in result.stderr
    assert not result.stdout


def test_cli_help_stays_lightweight():
    code = """
import sys
from fakehuman.cli import main
sys.argv = ['fakehuman', '--help']
try:
    main()
except SystemExit as exc:
    assert exc.code == 0
assert 'torch' not in sys.modules
assert 'transformers' not in sys.modules
assert 'huggingface_hub' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert "download-models" in result.stdout
    assert "--offline" in result.stdout
