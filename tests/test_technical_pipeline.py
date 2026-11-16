# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: William Droz <william.droz@idiap.ch>
# SPDX-License-Identifier: GPL-3.0-only

"""Offline end-to-end fidelity, scoring and resume checks for technical rewrites."""

import json

import pytest

from fakehuman import technical_pipeline as pipeline

SOURCE = "The export keeps all sessions. Authors can review the history.\n"
GENERATOR = "test/generator"
REVIEWERS = ["test/reviewer-a", "test/reviewer-b", "test/adjudicator"]
ACCEPT = {"verdict": "accept", "issues": []}


def response(value):
    return {"raw_response": json.dumps(value), "finish_reason": "stop", "usage": {}}


def rewritten_sections(sections, verb="retains"):
    return [
        {"id": section["id"], "text": section["text"].replace("keeps", verb).strip()}
        for section in sections
    ]


class Counter:
    def __init__(self, *, overflow_word=None):
        self.overflow_word = overflow_word

    def __call__(self, text):
        if self.overflow_word is not None and self.overflow_word in text:
            return 600
        return len(text.split()) + 2


class Detector:
    def __init__(self, name, value):
        self.name, self.value = name, value
        self.calls = []

    def metadata(self):
        return {"model_id": "offline-" + self.name, "revision": "frozen-test"}

    def score(self, texts):
        self.calls.append(list(texts))
        return [self.value(self.name, text) for text in texts]


class Calls:
    def __init__(self, *, generators=None, reviews=None, before_call=None):
        self.generators = generators or {GENERATOR: rewritten_sections}
        self.reviews = reviews or {}
        self.before_call = before_call
        self.bodies = []

    def __call__(self, body):
        if self.before_call is not None:
            self.before_call()
        self.bodies.append(body)
        payload = json.loads(body["messages"][1]["content"])
        model = body["model"]
        if "sections" in payload:
            generator = self.generators[model]
            if isinstance(generator, dict):
                return generator
            return response({"sections": generator(payload["sections"])})
        assert set(payload) <= {"original", "candidate", "prior_claims"}
        assert payload["original"] and payload["candidate"]
        decision = self.reviews.get(model, ACCEPT)
        return decision if "raw_response" in decision else response(decision)


@pytest.fixture(autouse=True)
def isolate_source_snapshots(monkeypatch):
    # Unit runs do not archive the working repository or depend on concurrent edits.
    monkeypatch.setattr(pipeline, "source_hashes", lambda: {})


def dependencies(monkeypatch, *, calls=None, section_value=None, window_value=None):
    section_value = section_value or (
        lambda name, text: 0.2 if "retains" in text else 0.8
    )
    window_value = window_value or section_value
    detectors = {name: Detector(name, section_value) for name in pipeline.NAMES}
    window_calls = []

    def windows(text, *, detectors, counters):
        assert set(detectors) == set(counters) == set(pipeline.NAMES)
        window_calls.append(text)
        rows = {}
        for name in pipeline.NAMES:
            value = window_value(name, text)
            rows[name] = {
                "available": value is not None,
                "mean": value,
                "reason": None if value is not None else "offline unavailable window",
            }
        return {"method": "offline-window-diagnostic", "detectors": rows}

    monkeypatch.setattr(pipeline, "score_document_windows", windows)
    deps = {
        "call": calls or Calls(),
        "detectors": detectors,
        "token_counters": {name: Counter() for name in pipeline.NAMES},
    }
    return deps, window_calls


def run(tmp_path, deps, *, text=SOURCE, models=None):
    models = models or [GENERATOR]
    catalog = {
        model: {"supported_parameters": ["response_format", "max_tokens"]}
        for model in [*models, *REVIEWERS]
    }
    return pipeline.process_technical_text(
        text,
        output_dir=tmp_path,
        models=models,
        reviewers=REVIEWERS,
        catalog=catalog,
        workers=1,
        **deps,
    )


@pytest.mark.parametrize("regressing_detector", pipeline.NAMES)
@pytest.mark.parametrize("regressing_measurement", ["sections", "windows"])
def test_any_detector_or_boundary_regression_returns_exact_original(
    tmp_path, monkeypatch, regressing_detector, regressing_measurement
):
    def value(measurement):
        def score(name, text):
            if "retains" not in text:
                return 0.8
            if name == regressing_detector and measurement == regressing_measurement:
                return 0.9
            return 0.2

        return score

    deps, _ = dependencies(
        monkeypatch,
        section_value=value("sections"),
        window_value=value("windows"),
    )
    result = run(tmp_path, deps)
    assert result["changed"] is False
    assert result["selected_model"] is None
    assert result["text"] == SOURCE
    assert result["candidates"][0]["acceptable"] is True
    assert (tmp_path / "output.txt").read_bytes() == SOURCE.encode()
    for scores in result["scores"]["detectors"].values():
        assert scores == {"original": 0.8, "rewritten": 0.8, "delta": 0.0}


def test_selects_best_worst_relative_gain_and_reports_both_full_scores(
    tmp_path, monkeypatch
):
    models = ["test/very-low-sections", "test/consistent"]
    calls = Calls(
        generators={
            models[0]: rewritten_sections,
            models[1]: lambda sections: rewritten_sections(sections, "preserves"),
        }
    )

    def section_value(name, text):
        return 0.1 if "retains" in text else 0.25 if "preserves" in text else 0.8

    def window_value(name, text):
        return 0.5 if "retains" in text else 0.25 if "preserves" in text else 0.8

    deps, _ = dependencies(
        monkeypatch, calls=calls, section_value=section_value, window_value=window_value
    )
    result = run(tmp_path, deps, models=models)
    assert result["selected_model"] == models[1]
    assert result["text"] == SOURCE.replace("keeps", "preserves")
    assert result["changed"] is True
    assert result["final_review"]["accepted"] is True
    assert result["candidate_count"] == 2
    for name in pipeline.NAMES:
        assert result["scores"]["detectors"][name]["original"] == 0.8
        assert result["scores"]["detectors"][name]["rewritten"] == 0.25
        assert result["scores"]["detectors"][name]["delta"] == pytest.approx(-0.55)
        assert (
            result["measurements"]["rewritten"]["windows"]["detectors"][name]["mean"]
            == 0.25
        )
    assert json.loads((tmp_path / "result.json").read_text()) == result
    assert result["pangram_submissions"] == 0
    assert result["human_review_performed"] is False


def test_conflicting_grounded_reviews_use_score_blind_adjudication(
    tmp_path, monkeypatch
):
    objection = {
        "verdict": "reject",
        "issues": [
            {
                "severity": "material",
                "category": "wording",
                "source_quote": "keeps all sessions",
                "candidate_quote": "retains all sessions",
                "explanation": "The reviewer alleges a difference between keeps and retains.",
            }
        ],
    }
    calls = Calls(reviews={REVIEWERS[1]: objection})
    deps, window_calls = dependencies(monkeypatch, calls=calls)

    def before_call():
        assert not window_calls
        assert all(not detector.calls for detector in deps["detectors"].values())

    calls.before_call = before_call
    result = run(tmp_path, deps)
    candidate = result["candidates"][0]
    assert candidate["acceptable"] is True
    assert [review["status"] for review in candidate["reviews"]] == ["accept", "reject"]
    assert candidate["adjudication"]["model"] == REVIEWERS[2]
    assert candidate["adjudication"]["status"] == "accept"
    assert [body["model"] for body in calls.bodies] == [GENERATOR, *REVIEWERS]
    adjudication_payload = json.loads(calls.bodies[-1]["messages"][1]["content"])
    assert set(adjudication_payload) == {"original", "candidate", "prior_claims"}
    assert adjudication_payload["original"] == SOURCE
    assert adjudication_payload["candidate"] == result["text"]
    assert len(adjudication_payload["prior_claims"]) == 1
    assert "model" not in adjudication_payload["prior_claims"][0]


@pytest.mark.parametrize("finish_reason", ["length", "transport_error"])
def test_generation_operational_failure_is_not_material_refusal(
    tmp_path, monkeypatch, finish_reason
):
    calls = Calls(
        generators={
            GENERATOR: {
                "raw_response": "",
                "finish_reason": finish_reason,
                "usage": None,
            }
        }
    )
    deps, _ = dependencies(monkeypatch, calls=calls)
    result = run(tmp_path, deps)
    candidate = result["candidates"][0]
    assert candidate["disposition"] == "generation_unavailable"
    assert candidate["generation"]["finish_reason"] == finish_reason
    assert candidate["reviews"] == []
    assert len(calls.bodies) == 1
    assert result["text"] == SOURCE
    assert result["status"] == "provider_error"
    assert result["provider_status"] == "failed"
    assert result["provider_errors"][0]["stage"] == "generation"


def test_unavailable_review_and_adjudication_are_not_semantic_rejections(
    tmp_path, monkeypatch
):
    unavailable = {
        "raw_response": "",
        "finish_reason": "transport_error",
        "usage": None,
    }
    calls = Calls(reviews={REVIEWERS[0]: unavailable, REVIEWERS[2]: unavailable})
    deps, _ = dependencies(monkeypatch, calls=calls)
    result = run(tmp_path, deps)
    candidate = result["candidates"][0]
    assert candidate["disposition"] == "review_unavailable"
    assert candidate["resolution"]["status"] == "unavailable"
    assert candidate["adjudication"]["status"] == "unavailable"
    assert candidate["acceptable"] is False
    assert result["changed"] is False
    assert result["status"] == "provider_error"
    assert result["provider_status"] == "failed"
    assert {error["stage"] for error in result["provider_errors"]} == {"review"}


def test_accepted_candidate_survives_other_provider_errors(tmp_path, monkeypatch):
    failed_model = "test/failed-generator"
    calls = Calls(
        generators={
            GENERATOR: rewritten_sections,
            failed_model: {
                "raw_response": "PRIVATE PROVIDER DETAILS",
                "finish_reason": "transport_error",
                "transport_errors": [{"type": "http_error", "http_status": 401}],
            },
        }
    )
    deps, _ = dependencies(monkeypatch, calls=calls)
    result = run(tmp_path, deps, models=[GENERATOR, failed_model])
    assert result["text"] == SOURCE.replace("keeps", "retains")
    assert result["status"] == "changed_with_provider_errors"
    assert result["provider_status"] == "partial_failure"
    error = result["provider_errors"][0]
    assert error["candidate_key"] == failed_model
    assert error["http_status"] == 401
    assert "PRIVATE" not in json.dumps(result["provider_errors"])


def test_missing_generated_section_rejects_format_without_review(tmp_path, monkeypatch):
    calls = Calls(generators={GENERATOR: response({"sections": []})})
    deps, _ = dependencies(monkeypatch, calls=calls)
    result = run(tmp_path, deps)
    candidate = result["candidates"][0]
    assert candidate["disposition"] == "generation_invalid_format"
    assert candidate["reviews"] == []
    assert result["text"] == SOURCE


def test_resume_reuses_generation_reviews_and_all_scores_without_calls(
    tmp_path, monkeypatch
):
    deps, window_calls = dependencies(monkeypatch)
    result = run(tmp_path, deps)
    before = (
        len(deps["call"].bodies),
        len(window_calls),
        [len(detector.calls) for detector in deps["detectors"].values()],
    )
    assert before == (3, 2, [2, 2])
    assert run(tmp_path, deps) == result
    assert before == (
        len(deps["call"].bodies),
        len(window_calls),
        [len(detector.calls) for detector in deps["detectors"].values()],
    )


def test_changed_input_cannot_reuse_or_replace_immutable_run(tmp_path, monkeypatch):
    deps, _ = dependencies(monkeypatch)
    run(tmp_path, deps)
    before_calls = len(deps["call"].bodies)
    with pytest.raises(ValueError, match="Checkpoint integrity/binding mismatch"):
        run(tmp_path, deps, text=SOURCE + "There is another export.\n")
    assert len(deps["call"].bodies) == before_calls
    assert (tmp_path / "input.txt").read_bytes() == SOURCE.encode()


def test_long_input_preserves_all_text_and_original_boundary_whitespace(
    tmp_path, monkeypatch
):
    paragraph = "The export keeps all sessions and authors can review history. " * 20
    text = "  " + "\r\n\r\n".join([paragraph.rstrip()] * 3) + "\r\n"
    deps, _ = dependencies(monkeypatch)
    result = run(tmp_path, deps, text=text)
    protocol = json.loads((tmp_path / "protocol.json").read_text())
    assert len(protocol["sections"]) > 1
    assert pipeline.assemble(protocol["sections"]) == text
    assert sum(protocol["original_word_weights"]) == len(text.split())
    assert result["text"] == text.replace("keeps", "retains")
    assert (tmp_path / "output.txt").read_bytes() == result["text"].encode()
    payload = json.loads(deps["call"].bodies[0]["messages"][1]["content"])
    assert pipeline.assemble(payload["sections"]) == text


@pytest.mark.parametrize("missing_measurement", ["sections", "windows"])
def test_missing_candidate_scores_are_explicit_and_prevent_selection(
    tmp_path, monkeypatch, missing_measurement
):
    missing_name = "desklib_deberta"

    def window_value(name, text):
        if "retains" in text:
            return (
                None
                if name == missing_name and missing_measurement == "windows"
                else 0.2
            )
        return 0.8

    deps, _ = dependencies(monkeypatch, window_value=window_value)
    if missing_measurement == "sections":
        deps["token_counters"][missing_name] = Counter(overflow_word="retains")
    result = run(tmp_path, deps)
    candidate = result["candidates"][0]
    assert candidate["acceptable"] is True
    scores = candidate["scores"][missing_measurement]
    missing = (
        scores["detectors"][missing_name]
        if missing_measurement == "windows"
        else scores[missing_name]
    )
    assert missing["available"] is False
    assert missing["mean"] is None
    assert missing["reason"]
    assert result["selected_model"] is None
    assert result["text"] == SOURCE
    if missing_measurement == "sections":
        assert len(deps["detectors"][missing_name].calls) == 1


def test_blank_preprocessed_section_is_not_silently_dropped(monkeypatch):
    deps, _ = dependencies(monkeypatch)
    monkeypatch.setattr(
        pipeline, "fakespot_clean_text", lambda text: "" if "code" in text else text
    )
    result = pipeline.score_sections(
        [{"id": "one", "text": "Useful prose."}, {"id": "two", "text": "code"}],
        [2, 1],
        deps["detectors"],
        deps["token_counters"],
    )
    missing = result["fakespot_roberta"]
    assert missing["available"] is False
    assert missing["sections"] is missing["mean"] is None
    assert not deps["detectors"]["fakespot_roberta"].calls
    assert result["desklib_deberta"]["available"] is True


@pytest.mark.parametrize("entry_point", ["process", "generate"])
def test_duplicate_reviewer_ids_cannot_count_as_independent_agreement(
    tmp_path, monkeypatch, entry_point
):
    deps, _ = dependencies(monkeypatch)
    reviewers = [REVIEWERS[0], REVIEWERS[0], *REVIEWERS[1:]]
    catalog = {model: {} for model in [GENERATOR, *REVIEWERS]}
    with pytest.raises(ValueError):
        if entry_point == "process":
            pipeline.process_technical_text(
                SOURCE,
                output_dir=tmp_path,
                models=[GENERATOR],
                reviewers=reviewers,
                catalog=catalog,
                workers=1,
                **deps,
            )
        else:
            pipeline.generate_review_candidate(
                [{"id": "section-0", "text": SOURCE}],
                GENERATOR,
                catalog=catalog,
                reviewers=reviewers,
                directory=tmp_path,
                binding={},
                call=deps["call"],
            )
    assert not deps["call"].bodies


def test_whole_document_acceptance_cannot_hide_facts_moved_between_sections(tmp_path):
    sections = [
        {"id": "first", "text": "Authors may export.\n\n"},
        {"id": "second", "text": "Administrators approve.\n"},
    ]
    calls = Calls(
        generators={
            GENERATOR: lambda _: [
                {"id": "first", "text": "Administrators approve."},
                {"id": "second", "text": "Authors may export."},
            ]
        }
    )

    def review_sections(body):
        reply = calls(body)
        payload = json.loads(body["messages"][1]["content"])
        if "sections" in payload or payload["original"] == pipeline.assemble(sections):
            return reply
        return response(
            {
                "verdict": "reject",
                "issues": [
                    {
                        "severity": "material",
                        "category": "scope",
                        "source_quote": "Authors may export.",
                        "candidate_quote": "Administrators approve.",
                        "explanation": "The original section's export capability moved elsewhere.",
                    }
                ],
            }
        )

    row = pipeline.generate_review_candidate(
        sections,
        GENERATOR,
        catalog={model: {} for model in [GENERATOR, *REVIEWERS]},
        reviewers=REVIEWERS,
        directory=tmp_path,
        binding={},
        call=review_sections,
    )
    assert [review["status"] for review in row["reviews"]] == ["accept", "accept"]
    assert row["section_reviews"][0]["resolution"]["status"] == "reject"
    assert row["disposition"] == "section_fidelity_refusal"
    assert row["acceptable"] is False


@pytest.mark.parametrize(
    "source",
    [
        "- **Summary:** The export keeps\n  all sessions. Authors can review the history.\n",
        '<p class="note">The export keeps all sessions.</p>'
        '<script>const value = "keeps";</script>',
    ],
)
def test_formatted_technical_rewrite_preserves_source_structure(
    tmp_path, monkeypatch, source
):
    deps, _ = dependencies(monkeypatch)
    result = run(tmp_path, deps, text=source)
    assert result["changed"] is True
    assert "retains" in result["text"]
    assert (tmp_path / "input.txt").read_bytes() == source.encode()
    if source.startswith("<"):
        assert '<p class="note">' in result["text"]
        assert '<script>const value = "keeps";</script>' in result["text"]
        assert "<hf-" not in result["text"]
    else:
        assert result["text"].startswith("- **Summary:**")
    assert run(tmp_path, deps, text=source) == result


def test_formatted_technical_candidate_cannot_discard_emphasis(tmp_path, monkeypatch):
    text = "- **Summary:** The export keeps all sessions.\n"
    calls = Calls(
        generators={
            GENERATOR: lambda sections: [
                {"id": row["id"], "text": row["text"].replace("**", "")}
                for row in rewritten_sections(sections)
            ]
        }
    )
    deps, _ = dependencies(monkeypatch, calls=calls)
    result = run(tmp_path, deps, text=text)
    assert result["text"] == text
    candidate = result["candidates"][0]
    assert candidate["disposition"] == "format_refusal"
    assert candidate["format_issues"]
    assert candidate["reviews"] == []
    assert "scores" not in candidate


def test_formatted_identity_fallback_keeps_original_line_wraps(tmp_path, monkeypatch):
    text = "- **Summary:** The export keeps\n  all sessions.\n"
    deps, _ = dependencies(monkeypatch, section_value=lambda name, text: 0.8)
    result = run(tmp_path, deps, text=text)
    assert result["changed"] is False
    assert result["text"] == text


def test_costs_include_section_reviews_and_adjudications(tmp_path, monkeypatch):
    objection = {
        "verdict": "reject",
        "issues": [
            {
                "severity": "material",
                "category": "wording",
                "source_quote": "keeps all sessions",
                "candidate_quote": "retains all sessions",
                "explanation": "The reviewer alleges a difference in wording.",
            }
        ],
    }
    calls = Calls(reviews={REVIEWERS[1]: objection})

    def costed_call(body):
        result = calls(body)
        result["usage"] = {"cost": 0.002}
        return result

    deps, _ = dependencies(monkeypatch, calls=costed_call)
    result = run(tmp_path, deps, text=SOURCE * 60)
    candidate = result["candidates"][0]
    assert len(candidate["section_reviews"]) > 1
    expected = 1 + 3 * (1 + len(candidate["section_reviews"]))
    assert len(calls.bodies) == expected
    summary = result["cost_summary"]
    assert summary["request_count"] == expected
    assert summary["gateway_cost_usd"]["total"] == pytest.approx(expected * 0.002)
    assert summary["upstream_cost_usd"]["total"] is None
    assert summary["upstream_cost_usd"]["missing_requests"] == expected
    assert json.loads((tmp_path / "result.json").read_text())["cost_summary"] == summary
    assert run(tmp_path, deps, text=SOURCE * 60)["cost_summary"] == summary
    assert len(calls.bodies) == expected
