# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: William Droz <william.droz@idiap.ch>
# SPDX-License-Identifier: GPL-3.0-only

"""Check the packaged pipeline against its runtime implementation."""

import pytest

from fakehuman.cli import load_manifest
from fakehuman.research_rules import source_hashes, validate_manifest


def test_default_pipeline_matches_current_implementation():
    current = load_manifest()
    assert current["source_hashes"] == source_hashes()
    assert "evaluation_source_hashes" not in current


def test_pipeline_refuses_source_fingerprints_from_another_implementation():
    manifest = load_manifest()
    manifest["source_hashes"]["rules.py"] = "0" * 64
    with pytest.raises(ValueError, match="Implementation changed"):
        validate_manifest(manifest)
