# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: William Droz <william.droz@idiap.ch>
# SPDX-License-Identifier: GPL-3.0-only

"""Extended registry, kept separate so the original experiment remains reproducible."""

from __future__ import annotations

import hashlib
import random
from pathlib import Path

from . import candidates_lexical, candidates_structural, rules

RULES = {**rules.RULES, **candidates_lexical.RULES, **candidates_structural.RULES}
RULE_LABELS = {
    **rules.RULE_LABELS,
    **candidates_lexical.RULE_LABELS,
    **candidates_structural.RULE_LABELS,
}


def apply_rules(text: str, names: list[str], seed: int = 17) -> str:
    original = text
    for name in names:
        # Same stable, original-document RNG convention as the first experiment.
        material = f"{seed}\0{name}\0{original}".encode()
        rng = random.Random(int.from_bytes(hashlib.sha256(material).digest(), "big"))
        text = RULES[name](text, rng)
    return text


def source_hashes() -> dict[str, str]:
    return {
        name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in (
            "rules.py",
            "candidates_lexical.py",
            "candidates_structural.py",
            "research_rules.py",
            "meaning.py",
        )
    }


def validate_manifest(manifest: dict) -> None:
    if manifest.get("schema_version") != 2:
        raise ValueError("Expected research manifest version 2")
    if manifest.get("source_hashes") != source_hashes():
        raise ValueError(
            "Implementation changed since evaluation; run a new experiment"
        )
    if any(name not in RULES for name in manifest["rules"]):
        raise ValueError("Unknown rule in manifest")
