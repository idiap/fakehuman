# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: William Droz <william.droz@idiap.ch>
# SPDX-License-Identifier: GPL-3.0-only

"""Exercise browser-facing transformations and shared model inference."""

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from fakehuman.cli import load_manifest
from fakehuman.document_chunks import MAX_TECHNICAL_CHARACTERS
from fakehuman.research_rules import RULES
from fakehuman.rule_catalog import catalog_rule_names
from fakehuman.service import TextService


def test_guard_and_detector_calls_do_not_overlap_across_workers():
    guard_entered = threading.Event()
    release_guard = threading.Event()
    score_started = threading.Event()
    detector_entered = threading.Event()

    class BlockingGuard:
        def check_pairs(self, pairs):
            guard_entered.set()
            if not release_guard.wait(timeout=5):
                raise TimeoutError("The test did not release the guard")
            return [{"passed": True, "reason": "fixture"} for _ in pairs]

    class Detector:
        def score(self, texts):
            detector_entered.set()
            return [0.25 for _ in texts]

    service = TextService(
        load_manifest(), guard=BlockingGuard(), detectors={"fixture": Detector()}
    )

    def score():
        score_started.set()
        return service.score_text("A passage to score.")

    with ThreadPoolExecutor(max_workers=2) as workers:
        transformation = workers.submit(
            service.inspect_transformation,
            "Therefore, the team prepared a brief report.",
        )
        try:
            assert guard_entered.wait(timeout=5)
            scores = workers.submit(score)
            assert score_started.wait(timeout=5)
            assert not detector_entered.wait(timeout=0.1)
        finally:
            release_guard.set()
        assert transformation.result(timeout=5)["text"] == (
            "Thus, the team prepared a short report."
        )
        assert scores.result(timeout=5) == {"fixture": 0.25}
        assert detector_entered.is_set()


class AllowGuard:
    def check_pairs(self, pairs):
        return [{"passed": True, "reason": "fixture"} for _ in pairs]


@pytest.mark.parametrize("technical_status", ["changed", "provider_error"])
def test_yolo_runs_edits_rewrite_and_frozen_rules_in_order(
    monkeypatch, technical_status
):
    service = TextService(load_manifest(), guard=AllowGuard(), detectors={})
    source = "The team did not send the report."
    edited = "Therefore, the team did not send the report."
    rewritten = "Therefore, the team did not deliver the report."
    calls = []

    def rewrite(text, *, strategy, rounds=3, input_format):
        calls.append((text, strategy, rounds, input_format))
        if strategy == "adaptive":
            return {"text": edited, "status": "changed"}
        return {
            "text": rewritten if technical_status == "changed" else text,
            "attempted_text": rewritten,
            "status": technical_status,
        }

    monkeypatch.setattr(service, "rewrite_text", rewrite)
    result = service.run_yolo(source, rounds=2, input_format="plain")
    assert calls == [
        (source, "adaptive", 2, "plain"),
        (edited, "technical-update-experimental", 3, "plain"),
    ]
    assert list(result["stages"]) == [
        "adaptive",
        "technical-update-experimental",
        "frozen",
    ]
    expected = rewritten if technical_status == "changed" else edited
    assert result["text"] == expected.replace("Therefore", "Thus")
    assert result["stages"]["frozen"]["text"] == result["text"]
    assert result["stages"]["adaptive"]["source_text"] == source
    assert result["stages"]["technical-update-experimental"]["source_text"] == edited
    assert result["stages"]["frozen"]["source_text"] == expected


def test_yolo_retains_a_technical_sized_input_when_sentence_editing_grows_it(
    monkeypatch,
):
    service = TextService(load_manifest(), guard=AllowGuard(), detectors={})
    source = "x" * MAX_TECHNICAL_CHARACTERS
    calls = []

    def rewrite(text, *, strategy, **options):
        calls.append((text, strategy))
        return {"text": text + "x" if strategy == "adaptive" else text}

    monkeypatch.setattr(service, "rewrite_text", rewrite)
    with pytest.raises(ValueError, match="24,000 characters"):
        service.run_yolo(source + "x")
    assert not calls

    result = service.run_yolo(source)
    assert calls == [
        (source, "adaptive"),
        (source, "technical-update-experimental"),
    ]
    assert result["adaptive_output_over_limit"]
    assert result["stages"]["technical-update-experimental"]["source_text"] == source
    assert result["text"] == source


def test_yolo_contextual_policy_applies_to_sentence_editing(monkeypatch):
    service = TextService(load_manifest(), guard=AllowGuard(), detectors={})
    calls = []

    def rewrite(text, **options):
        calls.append(options)
        return {"text": text}

    monkeypatch.setattr(service, "rewrite_text", rewrite)
    service.run_yolo("A passage.", review_policy="contextual")
    assert calls[0]["review_policy"] == "contextual"
    assert "review_policy" not in calls[1]


def test_repair_service_keeps_original_and_uses_unique_runs(monkeypatch, tmp_path):
    from fakehuman import repair

    calls = []

    def propose(original, candidate, findings, **options):
        calls.append((original, candidate, findings, options))
        return {"text": original, "attempted_text": "Repaired.", "changed": False}

    monkeypatch.setattr(repair, "repair_candidate", propose)
    monkeypatch.setenv("FAKEHUMAN_RUNS_DIR", str(tmp_path))
    service = TextService(load_manifest(), guard=AllowGuard(), detectors={})
    for _ in range(2):
        result = service.repair_candidate(
            "Original.", "Draft.", [], input_format="plain"
        )
        assert result["text"] == "Original."
        assert not result["changed"]
    assert calls[0][:3] == ("Original.", "Draft.", [])
    assert calls[0][3]["input_format"] == "plain"
    assert calls[0][3]["output_dir"].parent == tmp_path
    assert calls[0][3]["output_dir"] != calls[1][3]["output_dir"]


def test_selection_uses_manifest_order_and_leaves_default_pipeline_intact():
    manifest = load_manifest()
    manifest["rules"] = ["contractions", "expand_contractions"]
    service = TextService(manifest, guard=AllowGuard(), detectors={})
    original = "The team did not send the report."

    report = service.inspect_transformation(
        original, enabled_rules=["expand_contractions", "contractions"]
    )
    assert report["text"] == original
    assert [change["rule"] for change in report["changes"]] == manifest["rules"]

    assert service.inspect_transformation(original, enabled_rules=[])["changes"] == []
    selected = service.inspect_transformation(original, enabled_rules=["contractions"])
    assert selected["text"] == "The team didn't send the report."
    assert service.humanify(original) == original
    with pytest.raises(ValueError, match="outside the configured pipeline"):
        service.inspect_transformation(original, enabled_rules=["normalize_quotes"])


def test_catalog_rules_require_explicit_selection_and_opt_in():
    service = TextService(load_manifest(), guard=AllowGuard(), detectors={})
    original = "The team did not send the report."

    with pytest.raises(ValueError, match="outside the configured pipeline"):
        service.inspect_transformation(original, enabled_rules=["contractions"])

    report = service.inspect_transformation(
        original, enabled_rules=["contractions"], include_catalog=True
    )
    assert report["text"] == "The team didn't send the report."
    assert [change["rule"] for change in report["changes"]] == ["contractions"]
    assert not report["refused"]
    assert service.humanify(original) == original
    assert service.inspect_transformation(original)["text"] == original
    assert (
        service.inspect_transformation(original, include_catalog=True)["text"]
        == original
    )
    assert (
        service.inspect_transformation(
            original, enabled_rules=[], include_catalog=True
        )["changes"]
        == []
    )


@pytest.mark.parametrize("include_catalog", [False, True])
def test_unknown_rule_selections_are_rejected(include_catalog):
    service = TextService(load_manifest(), guard=AllowGuard(), detectors={})
    with pytest.raises(ValueError, match="unknown_rule"):
        service.inspect_transformation(
            "The team did not send the report.",
            enabled_rules=["unknown_rule"],
            include_catalog=include_catalog,
        )


def test_catalog_execution_keeps_configured_rules_first_and_deduplicates():
    manifest = load_manifest()
    manifest["rules"] = ["expand_contractions", "expand_contractions"]
    service = TextService(manifest, guard=AllowGuard(), detectors={})
    original = "The team didn't send the report."

    report = service.inspect_transformation(
        original,
        enabled_rules=["contractions", "expand_contractions", "contractions"],
        include_catalog=True,
    )
    assert [change["rule"] for change in report["changes"]] == [
        "expand_contractions",
        "contractions",
    ]
    assert report["text"] == original
    assert service.humanify(original) == "The team did not send the report."


def test_catalog_ranking_evaluates_every_rule_once_in_catalog_order(monkeypatch):
    manifest = load_manifest()
    manifest["rules"] = ["lex_due_to", "contractions", "lex_due_to"]
    service = TextService(manifest, guard=AllowGuard(), detectors={})
    calls = []

    def unchanged(text, *, rules):
        calls.append(rules)
        return {"text": text, "refused": False}

    monkeypatch.setattr(service, "_checked", unchanged)
    default = service.rank_rules("The report is ready.")
    assert calls == [["lex_due_to"], ["contractions"]]
    assert [row["rule"] for row in default] == ["lex_due_to", "contractions"]

    calls.clear()
    catalog = catalog_rule_names(manifest["rules"])
    assert catalog[:2] == ("lex_due_to", "contractions")
    assert catalog[2:] == tuple(
        name for name in RULES if name not in {"lex_due_to", "contractions"}
    )
    assert len(catalog) == len(set(catalog)) == len(RULES)
    ranking = service.rank_rules("The report is ready.", include_catalog=True)
    assert calls == [[name] for name in catalog]
    assert [row["rule"] for row in ranking] == list(catalog)
    assert all(row["status"] == "unchanged" for row in ranking)


def test_rank_scores_preserved_isolated_edits_with_one_baseline(monkeypatch):
    class SelectiveGuard:
        def check_pairs(self, pairs):
            return [
                {
                    "passed": "Nevertheless" not in candidate,
                    "reason": "contradiction"
                    if "Nevertheless" in candidate
                    else "fixture",
                }
                for _, candidate in pairs
            ]

    manifest = load_manifest()
    manifest["rules"].append("normalize_quotes")
    service = TextService(manifest, guard=SelectiveGuard(), detectors={})
    original = (
        "Therefore, we took a brief break due to the delay. "
        "However, the meeting resumed."
    )
    score_by_text = {
        original: (0.75, 0.625),
        original.replace("Therefore", "Thus"): (0.5, 0.625),
        original.replace("due to", "because of"): (0.375, 0.5),
        original.replace("a brief", "a short"): (0.5, 0.625),
    }
    calls = []

    def score(text, *, detailed):
        assert detailed
        calls.append(text)
        a, b = score_by_text[text]
        return {"scores": {"a": a, "b": b}, "detectors": {}}

    monkeypatch.setattr(service, "score_text", score)
    ranked = service.rank_rules(original)
    assert calls == list(score_by_text)
    assert [(row["rule"], row["status"], row["rank"]) for row in ranked] == [
        ("lex_due_to", "ranked", 1),
        ("lex_therefore_thus", "ranked", 2),
        ("lex_a_brief_short", "ranked", 2),
        ("lex_initial_however_nevertheless", "refused", None),
        ("normalize_quotes", "unchanged", None),
    ]
    assert ranked[0]["mean_delta"] == -0.25
    assert ranked[0]["detectors"]["a"] == {
        "input": 0.75,
        "output": 0.375,
        "delta": -0.375,
    }
    assert "possible contradiction" in ranked[3]["refusal_reasons"][0]
    calls.clear()
    assert all(
        row["status"] == "unchanged"
        for row in service.rank_rules("The meeting resumed.")
    )
    assert calls == []


def test_rank_excludes_candidates_with_mixed_detector_methods(monkeypatch):
    service = TextService(load_manifest(), guard=AllowGuard(), detectors={})

    def score(text, *, detailed):
        assert detailed
        method = (
            {} if text.startswith("Therefore") else {"window_scoring": {"mean": 0.4}}
        )
        return {
            "scores": {"a": 0.4, "b": 0.4},
            "detectors": {"a": {}, "b": method},
        }

    monkeypatch.setattr(service, "score_text", score)
    result = service.rank_rules("Therefore, the meeting resumed.")
    assert result[0]["status"] == "no_comparable_scores"
    assert result[0]["rank"] is None
    assert result[0]["mean_delta"] is None
