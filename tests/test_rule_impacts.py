# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: William Droz <william.droz@idiap.ch>
# SPDX-License-Identifier: GPL-3.0-only

"""Keep impact bands explicit and historical evidence attributable."""

import json
import math
import re
from importlib.resources import files
from statistics import fmean

import pytest

from fakehuman import rule_impacts
from fakehuman.research_rules import RULES, source_hashes


@pytest.mark.parametrize(
    ("delta", "label"),
    [
        (None, "Not yet measured"),
        (math.nan, "Not yet measured"),
        (math.inf, "Not yet measured"),
        (-math.inf, "Not yet measured"),
        (0, "Negligible change"),
        (-0.001, "Negligible change"),
        (0.001, "Negligible change"),
        (math.nextafter(0.001, math.inf), "Score increase"),
        (math.nextafter(-0.001, -math.inf), "Small reduction"),
        (math.nextafter(-0.01, 0), "Small reduction"),
        (-0.01, "Moderate reduction"),
        (math.nextafter(-0.05, 0), "Moderate reduction"),
        (-0.05, "Large reduction"),
        (-1, "Large reduction"),
        (1, "Score increase"),
    ],
)
def test_impact_bands_include_the_documented_boundaries(delta, label):
    assert rule_impacts.impact_label(delta) == label


@pytest.mark.parametrize("delta", [True, False, "0.01", object()])
def test_impact_label_rejects_non_numeric_values(delta):
    with pytest.raises(TypeError, match="number or None"):
        rule_impacts.impact_label(delta)


@pytest.mark.parametrize("delta", [-1.01, 1.01])
def test_impact_label_rejects_out_of_range_score_differences(delta):
    with pytest.raises(ValueError, match="between -1 and 1"):
        rule_impacts.impact_label(delta)


@pytest.mark.parametrize(
    ("record", "label"),
    [
        (None, "Not yet measured"),
        ({"attempted": 0, "accepted": 0, "mean_delta": -0.9}, "No AI matches"),
        ({"attempted": 10, "accepted": 0, "mean_delta": 0}, "All edits refused"),
        ({"attempted": 10, "accepted": 4, "mean_delta": 0}, "Negligible change"),
        ({"attempted": 10, "accepted": 4, "mean_delta": -0.03}, "Moderate reduction"),
        ({"attempted": 10, "accepted": 4, "mean_delta": 0.01}, "Score increase"),
    ],
)
def test_recorded_labels_separate_coverage_preservation_and_impact(record, label):
    assert rule_impacts.recorded_impact_label(record) == label
    assert rule_impacts.impact_color(label)


@pytest.fixture
def measured_archive(monkeypatch, tmp_path):
    record = {
        "trial": "contractions",
        "n": 60,
        "attempted": 10,
        "accepted": 6,
        "refused": 4,
        "errors": 0,
        "detector_deltas": {"first": -0.03, "second": -0.01},
        "detector_intervals": {"first": [-0.05, -0.01], "second": [-0.02, 0]},
        "raw_detector_deltas": {"first": -0.4, "second": -0.2},
        "human": {
            "n": 60,
            "attempted": 12,
            "accepted": 10,
            "refused": 2,
            "errors": 0,
            "detector_deltas": {"first": 0.004, "second": -0.003},
            "crossed_up_0_5": {"first": 1, "second": 0},
            "crossed_down_0_5": {"first": 0, "second": 2},
        },
        "matching_example": {
            "outcome": "refused",
            "issues": ["fixture preservation issue"],
            "meaning": {"passed": False},
            "detector_deltas": {"first": 0, "second": 0},
            "raw_detector_deltas": {"first": -0.8, "second": -0.6},
        },
    }
    archive = {
        "schema_version": 2,
        "source": {
            "label": "Fixture full catalog benchmark",
            "date": "2026-11-06",
            "path": "research/fixture/summary.json",
            "sha256": "a" * 64,
            "corpus_label": "HC3 and DACTYL development",
            "seed": 17,
        },
        "implementation_sha256": source_hashes(),
        "detectors": {
            "first": {"label": "Fakespot"},
            "second": {"label": "Desklib"},
        },
        "rules": {"contractions": record},
    }
    resource = tmp_path / "data" / "rule_impacts.json"
    resource.parent.mkdir()

    def write():
        resource.write_text(json.dumps(archive))

    write()
    monkeypatch.setattr(rule_impacts, "files", lambda _: tmp_path)
    return archive, write


def test_badge_uses_ai_accepted_output_and_details_distinguish_other_measurements(
    measured_archive,
):
    archive, _ = measured_archive
    measured = rule_impacts.recorded_rule_impacts()["contractions"]
    assert measured["mean_delta"] == pytest.approx(-0.02)
    assert rule_impacts.recorded_impact_label(measured) == "Moderate reduction"
    assert measured["raw_detector_deltas"] == {"first": -0.4, "second": -0.2}
    detail = measured["detail"]
    assert "60 AI passages from HC3 and DACTYL development" in detail
    assert "10 matched, 6 accepted and 4 refused" in detail
    assert "Fakespot -3.0000 pp; Desklib -1.0000 pp" in detail
    assert "Raw candidates before preservation checks" in detail
    assert (
        "Human controls: 60 passages, 12 matched, 10 accepted and 2 refused" in detail
    )
    assert "Fakespot +0.4000 pp; Desklib -0.3000 pp" in detail
    assert "Fakespot 1 upward / 0 downward; Desklib 0 upward / 2 downward" in detail
    assert "Matching example (separate diagnostic): refused" in detail
    assert "Fakespot -80.0000 pp; Desklib -60.0000 pp" in detail
    assert "matching example does not determine the badge" in detail
    assert "not a fresh holdout" in detail
    assert archive["source"]["sha256"] not in detail
    assert measured["source_sha256"] == archive["source"]["sha256"]


@pytest.mark.parametrize(
    ("attempted", "refused", "label", "explanation"),
    [
        (0, 0, "No AI matches", "No AI passage matched this rule"),
        (10, 10, "All edits refused", "Every matching AI edit was refused"),
    ],
)
def test_zero_accepted_edits_explain_coverage_or_refusal(
    measured_archive, attempted, refused, label, explanation
):
    archive, write = measured_archive
    archive["rules"]["contractions"].update(
        attempted=attempted,
        accepted=0,
        refused=refused,
        detector_deltas={"first": 0, "second": 0},
    )
    write()
    recorded = rule_impacts.recorded_rule_impacts()["contractions"]
    assert recorded["mean_delta"] == 0
    assert rule_impacts.recorded_impact_label(recorded) == label
    assert explanation in recorded["detail"]


def test_recorded_impacts_cover_the_complete_current_catalog(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    impacts = rule_impacts.recorded_rule_impacts()
    assert set(impacts) == set(RULES)
    assert len(impacts) == 135
    for rule, record in impacts.items():
        assert record["trial"] == rule
        assert record["mean_delta"] == pytest.approx(
            fmean(record["detector_deltas"].values())
        )


def test_recorded_impacts_preserve_sample_and_source_provenance():
    archive = json.loads(
        files("fakehuman").joinpath("data", "rule_impacts.json").read_text()
    )
    assert archive["schema_version"] == 2
    assert source_hashes() in (
        archive["implementation_sha256"],
        archive.get("compatible_implementation_sha256"),
    )
    assert archive["source"]["origin"] == "ai"
    assert archive["source"]["split"] == "development"
    assert archive["source"]["seed"] == 17
    assert archive["source"]["corpus_label"] == "HC3 and DACTYL development"
    for field in (
        "sha256",
        "corpus_sha256",
        "results_sha256",
        "metadata_sha256",
        "probe_results_sha256",
    ):
        assert re.fullmatch(r"[a-f0-9]{64}", archive["source"][field])
    for detector in archive["detectors"].values():
        assert detector["model_id"]
        assert re.fullmatch(r"[a-f0-9]{40}", detector["revision"])
    for row in rule_impacts.recorded_rule_impacts().values():
        assert row["n"] == 60
        assert 0 <= row["accepted"] <= row["attempted"] <= row["n"]
        assert row["accepted"] + row["refused"] == row["attempted"]
        assert row["errors"] == 0
        assert row["detector_deltas"].keys() == archive["detectors"].keys()
        assert row["detector_intervals"].keys() == archive["detectors"].keys()
        for low, high in row["detector_intervals"].values():
            assert -1 <= low <= high <= 1
        human = row["human"]
        assert human["n"] == 60
        assert 0 <= human["accepted"] <= human["attempted"] <= human["n"]
        assert human["accepted"] + human["refused"] == human["attempted"]
        assert human["errors"] == 0
        for detector in archive["detectors"]:
            upward = human["crossed_up_0_5"][detector]
            downward = human["crossed_down_0_5"][detector]
            assert 0 <= upward + downward <= human["accepted"]
        probe = row["matching_example"]
        assert probe["outcome"] in {"preserved", "refused"}
        for deltas in (
            row["detector_deltas"],
            row["raw_detector_deltas"],
            human["detector_deltas"],
            probe["detector_deltas"],
            probe["raw_detector_deltas"],
        ):
            assert deltas.keys() == archive["detectors"].keys()
            assert all(
                math.isfinite(delta) and -1 <= delta <= 1 for delta in deltas.values()
            )
        if probe["outcome"] == "refused":
            assert all(delta == 0 for delta in probe["detector_deltas"].values())
        else:
            assert probe["detector_deltas"] == probe["raw_detector_deltas"]
        assert row["source_label"] == archive["source"]["label"]
        assert row["source_path"] == archive["source"]["path"]
        assert row["source_sha256"] == archive["source"]["sha256"]
        assert "Unchanged and refused edits count as zero" in row["detail"]
        assert "development corpus is exploratory evidence" in row["detail"]


def test_source_changes_invalidate_historical_attribution(
    monkeypatch, measured_archive
):
    monkeypatch.setattr(rule_impacts, "source_hashes", lambda: {})
    assert rule_impacts.recorded_rule_impacts() == {}


def test_unsupported_archive_schema_does_not_supply_current_evidence(measured_archive):
    archive, write = measured_archive
    archive["schema_version"] = 1
    write()
    assert rule_impacts.recorded_rule_impacts() == {}
