# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: William Droz <william.droz@idiap.ch>
# SPDX-License-Identifier: GPL-3.0-only

"""Cache exactly the runtime assets for the three pinned model checkpoints."""

from __future__ import annotations

from pathlib import Path

from .detectors import DEFAULT_DETECTOR_NAMES, DETECTOR_SPECS
from .meaning import MODEL_ID, REVISION

_COMMON_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "model.safetensors",
)
_DEBERTA_FILES = ("added_tokens.json", "spm.model")
_ROBERTA_FILES = ("merges.txt", "vocab.json")


def model_assets() -> list[dict]:
    """List the files at the same immutable revisions used by the adapters.

    These checkpoints use single-file safetensors weights. Explicit filenames
    avoid downloading alternate weights or remote Python code, and allow an
    offline check to detect partially populated snapshots.
    """
    models = [
        {
            "name": "meaning_guard",
            "model_id": MODEL_ID,
            "revision": REVISION,
            "files": _COMMON_FILES + _DEBERTA_FILES,
        }
    ]
    for name in DEFAULT_DETECTOR_NAMES:
        spec = DETECTOR_SPECS[name]
        models.append(
            {
                "name": name,
                "model_id": spec.model_id,
                "revision": spec.revision,
                "files": _COMMON_FILES
                + (
                    _DEBERTA_FILES
                    if spec.architecture == "desklib_mean_pool"
                    else _ROBERTA_FILES
                ),
            }
        )
    return models


def download_models(*, offline: bool = False) -> list[dict]:
    """Download runtime assets, or check every required file in offline mode.

    Uses the normal Hugging Face cache and its HF_HOME/HF_HUB_CACHE overrides.
    No tokenizer or model is instantiated, and no weights are loaded into RAM.
    """
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError

    report = []
    for model in model_assets():
        cached = []
        for filename in model["files"]:
            try:
                cached.append(
                    hf_hub_download(
                        repo_id=model["model_id"],
                        revision=model["revision"],
                        filename=filename,
                        local_files_only=offline,
                    )
                )
            except (HfHubHTTPError, LocalEntryNotFoundError, OSError) as exc:
                hint = (
                    "Run 'fakehuman download-models' with network access first."
                    if offline
                    else "Check network access and the Hugging Face cache, then retry."
                )
                raise RuntimeError(
                    f"Could not cache {model['name']} file {filename} at "
                    f"{model['model_id']}@{model['revision']}. {hint}"
                ) from exc
        report.append(
            {
                "name": model["name"],
                "model_id": model["model_id"],
                "revision": model["revision"],
                "files": len(cached),
                "cache_path": str(Path(cached[0]).parent),
            }
        )
    return report
