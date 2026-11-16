# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: William Droz <william.droz@idiap.ch>
# SPDX-License-Identifier: GPL-3.0-only

"""Describe measured detector changes without inferring unmeasured rule effects."""

from __future__ import annotations

import json
import math
from importlib.resources import files
from statistics import fmean

from .research_rules import RULES, source_hashes

_COLORS = {
    "Not yet measured": "grey-8",
    "No AI matches": "grey-8",
    "All edits refused": "brown-8",
    "Score increase": "red-8",
    "Negligible change": "grey-8",
    "Small reduction": "teal-8",
    "Moderate reduction": "teal-9",
    "Large reduction": "green-9",
}


def impact_label(mean_delta: float | None) -> str:
    """Label mean after-minus-before detector change on the 0–1 score scale.

    These are display bands, not significance or efficacy claims. Absolute
    changes through 0.001 (0.1 percentage points) are negligible. Reductions
    below 0.01 are small; reductions from 0.01 to below 0.05 are moderate;
    reductions of at least 0.05 are large. Increases above 0.001 are adverse.
    Missing and nonfinite values carry no measured label.
    """
    if mean_delta is None:
        return "Not yet measured"
    if isinstance(mean_delta, bool) or not isinstance(mean_delta, (int, float)):
        raise TypeError("mean_delta must be a number or None")
    if not math.isfinite(mean_delta):
        return "Not yet measured"
    if not -1 <= mean_delta <= 1:
        raise ValueError("mean_delta must be between -1 and 1")
    if abs(mean_delta) <= 0.001:
        return "Negligible change"
    if mean_delta > 0:
        return "Score increase"
    if mean_delta > -0.01:
        return "Small reduction"
    if mean_delta > -0.05:
        return "Moderate reduction"
    return "Large reduction"


def impact_color(label: str) -> str:
    """Return a NiceGUI badge color, including a neutral fallback."""
    return _COLORS.get(label, "grey-8")


def recorded_impact_label(record: dict | None) -> str:
    """Distinguish coverage and preservation failures from measured effects."""
    if record is None:
        return "Not yet measured"
    if record["attempted"] == 0:
        return "No AI matches"
    if record["accepted"] == 0:
        return "All edits refused"
    return impact_label(record["mean_delta"])


def recorded_rule_impacts() -> dict[str, dict]:
    """Read packaged, independent-rule measurements matching current sources.

    The archive separates general-corpus AI results, human controls and authored
    matching examples. Only accepted-output changes on the AI corpus determine
    an impact band. Measurements require the evaluated source fingerprints or
    an explicitly recorded behavior-equivalent revision.
    """
    resource = files("fakehuman").joinpath("data", "rule_impacts.json")
    archive = json.loads(resource.read_text(encoding="utf-8"))
    if archive.get("schema_version") != 2 or source_hashes() not in (
        archive["implementation_sha256"],
        archive.get("compatible_implementation_sha256"),
    ):
        return {}
    source = archive["source"]

    def changes(deltas: dict) -> str:
        return "; ".join(
            f"{archive['detectors'][detector]['label']} {delta * 100:+.4f} pp"
            for detector, delta in deltas.items()
        )

    result = {}
    for name, recorded in archive["rules"].items():
        if name not in RULES:
            continue
        detector_deltas = recorded["detector_deltas"]
        human = recorded["human"]
        probe = recorded["matching_example"]
        crossings = "; ".join(
            f"{metadata['label']} {human['crossed_up_0_5'][name]} upward / "
            f"{human['crossed_down_0_5'][name]} downward"
            for name, metadata in archive["detectors"].items()
        )
        detail = (
            f"Recorded benchmark: {recorded['n']} AI passages from "
            f"{source['corpus_label']} ({source['date']}, seed {source['seed']}). "
            f"Rule tested alone: {recorded['attempted']} matched, "
            f"{recorded['accepted']} accepted and {recorded['refused']} refused. "
            "Accepted-output mean changes (after minus before): "
            f"{changes(detector_deltas)}. The badge averages both detectors across "
            f"all {recorded['n']} AI passages. Unchanged and refused edits count "
            "as zero. Raw candidates before preservation checks: "
            f"{changes(recorded['raw_detector_deltas'])}.\n\n"
            f"Human controls: {human['n']} passages, {human['attempted']} matched, "
            f"{human['accepted']} accepted and {human['refused']} refused. "
            f"Accepted-output mean changes: {changes(human['detector_deltas'])}. "
            f"Threshold crossings at 0.5: {crossings}.\n\n"
            f"Matching example (separate diagnostic): {probe['outcome']}; "
            f"accepted-output changes {changes(probe['detector_deltas'])}; "
            f"raw-candidate changes {changes(probe['raw_detector_deltas'])}. "
            "The matching example does not determine the badge.\n\n"
            "This reused development corpus is exploratory evidence, not a fresh "
            "holdout, an estimate for your text or a measure of authorship. "
            f"Source: {source['label']}, {recorded['trial']} trial."
        )
        if recorded["attempted"] == 0:
            detail += " No AI passage matched this rule, so its corpus effect was not observed."
        elif recorded["accepted"] == 0:
            detail += (
                " Every matching AI edit was refused; no changed output was accepted."
            )
        result[name] = {
            **recorded,
            "mean_delta": fmean(detector_deltas.values()),
            "source_label": source["label"],
            "source_path": source["path"],
            "source_sha256": source["sha256"],
            "detail": detail,
        }
    return result
