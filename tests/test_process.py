# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: William Droz <william.droz@idiap.ch>
# SPDX-License-Identifier: GPL-3.0-only

"""Offline integration coverage for complete-document processing and checkpoints."""

import json

import pytest

from fakehuman.adaptive import EDITOR, REVIEWER
from fakehuman.detectors import fakespot_clean_text
from fakehuman.process import NAMES, process_text

SOURCE = (
    "The gentle river flows beside the village. The stone bridge crosses the river."
)
PASS = {"passed": True, "reason": "The facts are retained.", "category": "faithful"}


def response(value):
    return {"raw_response": json.dumps(value), "finish_reason": "stop", "usage": {}}


class Guard:
    def __init__(self):
        self.pairs = []

    def metadata(self):
        return {"model_id": "offline-guard"}

    def check_pairs(self, pairs):
        self.pairs.extend(pairs)
        return [{"passed": True} for _ in pairs]


class Detector:
    def __init__(self, name):
        self.name = name
        self.inputs = []

    def metadata(self):
        return {"name": self.name, "model_id": "offline-" + self.name}

    def value(self, text):
        # Variable section scores expose accidental unweighted averaging.
        # Desklib deliberately worsens: it must never influence selection.
        baseline = 0.6 + len(text) / 10000
        change = 0.1 if self.name == "desklib_deberta" else -0.2
        return baseline + change * ("next to" in text)

    def score(self, texts):
        assert texts and all(text.strip() for text in texts)
        if self.name == "fakespot_roberta":
            assert all(fakespot_clean_text(text).strip() for text in texts)
        assert all(len(text) <= 512 for text in texts)
        self.inputs.extend(texts)
        return [self.value(text) for text in texts]


class Calls:
    def __init__(self, *, reject_final=False, fail_final=False, protected_index=False):
        self.bodies = []
        self.reject_final = reject_final
        self.fail_final = fail_final
        self.protected_index = protected_index

    def __call__(self, body):
        self.bodies.append(body)
        content = json.loads(body["messages"][1]["content"])
        if body["model"] == EDITOR:
            if self.protected_index:
                assert 0 not in {row["sentence_index"] for row in content["sentences"]}
                return response({"sentence_index": 0, "replacement": "# Changed."})
            sentence = next(
                (row for row in content["sentences"] if "beside" in row["text"]),
                content["sentences"][0],
            )
            return response(
                {
                    "sentence_index": sentence["sentence_index"],
                    "replacement": sentence["text"].replace("beside", "next to"),
                }
            )
        assert body["model"] == REVIEWER
        assert "current_fakespot_score" not in content
        # Local reviews carry document_context; the final independent review
        # receives the complete pair directly.
        if "document_context" not in content:
            if self.fail_final:
                raise TimeoutError("offline simulated final-review failure")
            if self.reject_final:
                return response(
                    {
                        "passed": False,
                        "reason": "The combined changes alter the emphasis.",
                        "category": "semantic_drift",
                    }
                )
        return response(PASS)


def dependencies(calls=None):
    return {
        "detectors": {name: Detector(name) for name in NAMES},
        "guard": Guard(),
        "call": calls if calls is not None else Calls(),
        "token_counters": dict.fromkeys(NAMES, len),
    }


def run(text, tmp_path, *, deps=None, **kwargs):
    return process_text(
        text,
        output_dir=tmp_path,
        rounds=1,
        workers=1,
        **(dependencies() if deps is None else deps),
        **kwargs,
    )


def test_long_document_reassembles_all_sections_and_uses_original_weights(tmp_path):
    text = (SOURCE + "\n\n") * 12 + "The river reaches the sea.\n"
    deps = dependencies()
    result = run(text, tmp_path, deps=deps, section_tokens=200)

    assert result["status"] == "changed"
    assert result["section_count"] > 1
    assert result["text"].replace("next to", "beside") == text
    assert (tmp_path / "input.txt").read_bytes() == text.encode()
    assert (tmp_path / "output.txt").read_bytes() == result["text"].encode()
    protocol = json.loads((tmp_path / "protocol.json").read_text())
    assert "".join(section["text"] for section in protocol["sections"]) == text
    assert all(max(row["token_counts"].values()) <= 200 for row in protocol["sections"])

    summary = result["scores"]
    assert summary["method"] == "original-word-weighted-section-mean"
    sections = summary["sections"]
    weights = [row["original_weight"] for row in sections]
    assert len(set(weights)) > 1
    assert summary["total_original_weight"] == len(text.split())
    assert len(result["text"].split()) > summary["total_original_weight"]
    for name in NAMES:
        for key in ("original", "rewritten"):
            expected = sum(
                row["original_weight"] * row["scores"][name][key] for row in sections
            ) / sum(weights)
            assert summary["detectors"][name][key] == pytest.approx(expected)
    assert summary["detectors"]["fakespot_roberta"]["delta"] < 0
    assert summary["detectors"]["desklib_deberta"]["delta"] > 0
    assert (text, result["text"]) in deps["guard"].pairs


def test_small_document_scores_are_native_and_resume_performs_no_calls(tmp_path):
    deps = dependencies()
    result = run(SOURCE, tmp_path, deps=deps)
    assert result["scores"]["method"] == "single-section-native"
    assert result["final_review"]["passed"] is True
    for name, detector in deps["detectors"].items():
        assert result["scores"]["detectors"][name]["original"] == detector.value(SOURCE)
        assert result["scores"]["detectors"][name]["rewritten"] == detector.value(
            result["text"]
        )
    prior = (
        len(deps["call"].bodies),
        len(deps["guard"].pairs),
        [len(detector.inputs) for detector in deps["detectors"].values()],
    )
    assert run(SOURCE, tmp_path, deps=deps) == result
    assert prior == (
        len(deps["call"].bodies),
        len(deps["guard"].pairs),
        [len(detector.inputs) for detector in deps["detectors"].values()],
    )


@pytest.mark.parametrize("failure", ["reject_final", "fail_final"])
def test_final_global_refusal_or_transport_error_rolls_back_and_scores_original(
    tmp_path, failure
):
    result = run(SOURCE, tmp_path, deps=dependencies(Calls(**{failure: True})))
    assert result["status"] == (
        "provider_error" if failure == "fail_final" else "reverted_final_review"
    )
    assert result["changed"] is False
    assert result["text"] == SOURCE
    if failure == "fail_final":
        assert "attempted_text" not in result
    else:
        assert result["attempted_text"] == SOURCE.replace("beside", "next to")
    assert "next to" in (tmp_path / "attempted.txt").read_text()
    assert (tmp_path / "changes.diff").read_text() == ""
    assert result["final_review"]["passed"] is False
    assert result["selected_search_steps"] > 0
    assert result["kept_candidate_count"] == 0
    assert all(row["delta"] == 0 for row in result["scores"]["detectors"].values())


def test_protected_sentence_index_is_rejected_before_rewriting(tmp_path):
    text = "# Report.\n\n" + SOURCE
    deps = dependencies(Calls(protected_index=True))
    result = run(text, tmp_path, deps=deps)
    assert result["text"] == text
    assert result["candidate_count"] == 2
    assert result["kept_candidate_count"] == 0
    generations = list((tmp_path / "search").glob("*/round_*/generation.json"))
    assert len(generations) == 2
    assert all(
        json.loads(path.read_text())["payload"]["finish_reason"]
        == "protected_span_refusal"
        for path in generations
    )
    assert all(body["model"] == EDITOR for body in deps["call"].bodies)


def test_entirely_protected_document_skips_remote_calls(tmp_path):
    text = "# Report.\n\n```text\nThe river flows.\n```\n"
    deps = dependencies()
    result = run(text, tmp_path, deps=deps)
    assert result["status"] == "unchanged"
    assert result["text"] == text
    assert result["candidate_count"] == 0
    assert deps["call"].bodies == []


@pytest.mark.parametrize(
    "text",
    ["```text\nThe river flows.\n```\n", "> The river flows.\n"],
)
def test_preprocessing_erases_content_without_scoring_empty_or_rewriting(
    tmp_path, text
):
    deps = dependencies()
    result = run(text, tmp_path, deps=deps, input_format="plain")
    assert result["text"] == text
    assert (tmp_path / "output.txt").read_bytes() == text.encode()
    assert result["status"] == "unchanged"
    assert result["candidate_count"] == 0
    assert deps["call"].bodies == []
    assert deps["detectors"]["fakespot_roberta"].inputs == []
    assert deps["detectors"]["desklib_deberta"].inputs == [text, text]
    for scope in (
        result["scores"]["detectors"],
        result["scores"]["sections"][0]["scores"],
    ):
        fakespot = scope["fakespot_roberta"]
        assert all(fakespot[key] is None for key in ("original", "rewritten", "delta"))
        assert set(fakespot["unavailable_reasons"]) == {
            "original",
            "rewritten",
            "delta",
        }
        assert scope["desklib_deberta"]["original"] is not None
        assert scope["desklib_deberta"]["delta"] == 0


def test_zero_scores_remain_valid_values_for_nonempty_prose(tmp_path):
    deps = dependencies()
    for detector in deps["detectors"].values():
        detector.value = lambda _text: 0.0
    result = run(SOURCE, tmp_path, deps=deps)
    assert result["text"] == SOURCE
    assert result["status"] == "unchanged"
    assert result["candidate_count"] == 2
    assert result["selected_search_steps"] == 0
    for scope in (
        result["scores"]["detectors"],
        result["scores"]["sections"][0]["scores"],
    ):
        assert all(
            scores == {"original": 0.0, "rewritten": 0.0, "delta": 0.0}
            for scores in scope.values()
        )


def test_resume_rejects_changed_configuration_before_api_calls(tmp_path):
    deps = dependencies()
    run(SOURCE, tmp_path, deps=deps)
    previous = len(deps["call"].bodies)
    with pytest.raises(ValueError, match="configuration or code changed"):
        run(SOURCE, tmp_path, deps=deps, section_tokens=200)
    assert len(deps["call"].bodies) == previous


def test_resume_rejects_changed_evaluation_detector_even_without_search(tmp_path):
    text = "# Report.\n"
    deps = dependencies()
    run(text, tmp_path, deps=deps)
    deps["detectors"]["desklib_deberta"].metadata = lambda: {
        "name": "desklib_deberta",
        "model_id": "different-evaluation-model",
    }
    with pytest.raises(ValueError, match="configuration or code changed"):
        run(text, tmp_path, deps=deps)
    assert deps["call"].bodies == []


def test_crlf_is_preserved_in_files_and_checkpoint_resume(tmp_path):
    text = SOURCE + "\r\n\r\nBest,\\\r\nWilliam\r\n"
    deps = dependencies()
    result = run(text, tmp_path, deps=deps)
    assert result["text"].replace("next to", "beside") == text
    assert (tmp_path / "input.txt").read_bytes() == text.encode()
    assert (tmp_path / "output.txt").read_bytes() == result["text"].encode()
    assert run(text, tmp_path, deps=deps) == result


def test_wrapped_email_list_is_editable_without_losing_formatting(tmp_path):
    text = (
        "Dear colleagues,\n\n"
        "  * **Summary:** The gentle river flows\n"
        "    beside the village. The stone bridge crosses the river.\n\n"
        "Best,\nAlex\n"
    )
    deps = dependencies()
    result = run(text, tmp_path, deps=deps, input_format="markdown")
    assert result["changed"] is True
    assert "next to" in result["text"]
    assert "  * **Summary:**" in result["text"]
    assert result["text"].endswith("Best,\nAlex\n")
    assert (tmp_path / "input.txt").read_bytes() == text.encode()
    assert result["formatting"]["format"] == "markdown"
    assert all("**" not in value for value in deps["detectors"][NAMES[0]].inputs)
    assert run(text, tmp_path, deps=deps, input_format="markdown") == result


def test_failed_formatted_rewrite_restores_exact_wrapped_source(tmp_path):
    text = "- **Summary:** The gentle river flows\n  beside the village.\n"
    result = run(text, tmp_path, deps=dependencies(Calls(reject_final=True)))
    assert result["text"] == text
    assert result["attempted_text"] == (
        "- **Summary:** The gentle river flows next to the village.\n"
    )
    assert (tmp_path / "input.txt").read_bytes() == text.encode()
    assert (tmp_path / "output.txt").read_bytes() == text.encode()


@pytest.mark.parametrize("reject_final", [False, True])
def test_html_processing_edits_prose_without_changing_attributes_or_scripts(
    tmp_path, reject_final
):
    text = (
        '<div class="mail"><p>The gentle river flows beside the village.</p>'
        "<p>The stone bridge crosses the river.</p>"
        '<script>const river = "beside";</script></div>'
    )
    result = run(
        text,
        tmp_path,
        deps=dependencies(Calls(reject_final=reject_final)),
        input_format="html",
    )
    assert result["changed"] is not reject_final
    if reject_final:
        assert result["text"] == text
    candidate = result["attempted_text"] if reject_final else result["text"]
    assert candidate == text.replace("flows beside", "flows next to")


def test_model_overrides_reach_provider_and_bind_checkpoint_resume(tmp_path):
    calls = Calls()
    submitted = []

    def configured_call(body):
        submitted.append(body)
        model = {"custom/editor": EDITOR, "custom/reviewer": REVIEWER}[body["model"]]
        return calls({**body, "model": model})

    deps = dependencies(configured_call)
    result = run(
        SOURCE,
        tmp_path,
        deps=deps,
        editor_model="custom/editor",
        reviewer_model="custom/reviewer",
    )
    assert result["changed"]
    assert {body["model"] for body in submitted} == {"custom/editor", "custom/reviewer"}
    protocol = json.loads((tmp_path / "protocol.json").read_text())
    assert protocol["editor_model"] == "custom/editor"
    assert protocol["reviewer_model"] == "custom/reviewer"
    previous = len(submitted)
    assert (
        run(
            SOURCE,
            tmp_path,
            deps=deps,
            editor_model="custom/editor",
            reviewer_model="custom/reviewer",
        )
        == result
    )
    assert len(submitted) == previous
    with pytest.raises(ValueError, match="configuration or code changed"):
        run(
            SOURCE,
            tmp_path,
            deps=deps,
            editor_model="custom/editor",
            reviewer_model="other/reviewer",
        )
    assert len(submitted) == previous


@pytest.mark.parametrize("failure", ["http", "malformed", "exception"])
def test_unavailable_generation_is_reported_and_preserves_original(tmp_path, failure):
    attempts = []

    def unavailable(body):
        attempts.append(body)
        if failure == "exception":
            raise TimeoutError("PRIVATE provider diagnostic")
        if failure == "malformed":
            return {"finish_reason": "stop", "raw_response": "PRIVATE malformed output"}
        return {
            "finish_reason": "transport_error",
            "raw_response": "PRIVATE error body",
            "transport_errors": [{"type": "http_error", "http_status": 401}],
        }

    deps = dependencies(unavailable)
    result = run(SOURCE, tmp_path, deps=deps)
    assert result["status"] == "provider_error"
    assert result["provider_status"] == "failed"
    assert result["changed"] is False
    assert result["text"] == SOURCE
    assert (tmp_path / "output.txt").read_text() == SOURCE
    assert len(result["provider_errors"]) == 2
    assert {error["stage"] for error in result["provider_errors"]} == {"generation"}
    if failure == "http":
        assert all(error["http_status"] == 401 for error in result["provider_errors"])
    assert "PRIVATE" not in json.dumps(result["provider_errors"])
    assert len(attempts) == 2
    assert run(SOURCE, tmp_path, deps=deps) == result
    assert len(attempts) == 2


def test_unavailable_candidate_review_is_reported_and_preserves_original(tmp_path):
    successful = Calls()

    def unavailable_review(body):
        if body["model"] == REVIEWER:
            return {
                "finish_reason": "transport_error",
                "transport_errors": [{"type": "connection_error", "http_status": None}],
            }
        return successful(body)

    result = run(SOURCE, tmp_path, deps=dependencies(unavailable_review))
    assert result["text"] == SOURCE
    assert result["status"] == "provider_error"
    assert result["provider_status"] == "failed"
    assert len(result["provider_errors"]) == 2
    assert {error["stage"] for error in result["provider_errors"]} == {"review"}
    assert result["kept_candidate_count"] == 0


def test_one_failed_proposal_preserves_independently_reviewed_success(tmp_path):
    successful = Calls()
    generations = 0

    def partially_available(body):
        nonlocal generations
        if body["model"] == EDITOR:
            generations += 1
            if generations == 1:
                return {
                    "finish_reason": "transport_error",
                    "transport_errors": [{"type": "http_error", "http_status": 503}],
                }
        return successful(body)

    result = run(SOURCE, tmp_path, deps=dependencies(partially_available))
    assert result["text"] == SOURCE.replace("beside", "next to")
    assert result["changed"] is True
    assert result["status"] == "changed_with_provider_errors"
    assert result["provider_status"] == "partial_failure"
    assert result["final_review"]["passed"] is True
    assert result["kept_candidate_count"] == 1
    assert len(result["provider_errors"]) == 1
    assert result["provider_errors"][0]["http_status"] == 503


def test_unavailable_final_review_is_an_operational_failure_with_rollback(tmp_path):
    result = run(SOURCE, tmp_path, deps=dependencies(Calls(fail_final=True)))
    assert result["status"] == "provider_error"
    assert result["text"] == SOURCE
    assert result["provider_status"] == "failed"
    assert len(result["provider_errors"]) == 1
    assert result["provider_errors"][0]["stage"] == "final_review"
    assert result["kept_candidate_count"] == 0


@pytest.mark.parametrize("final_verdict", ["accept", "reject", "unavailable"])
def test_contextual_policy_applies_to_final_review_and_retains_grounded_findings(
    tmp_path, final_verdict
):
    source = "The server does not stop when the client disconnects beside the village."
    edited = "The server keeps running when the client disconnects next to the village."
    calls = []

    class AdvisoryGuard(Guard):
        def check_pairs(self, pairs):
            self.pairs.extend(pairs)
            return [
                {"passed": False, "reason": "overlength", "max_contradiction": 1.0}
                for _ in pairs
            ]

    def call(body):
        content = json.loads(body["messages"][1]["content"])
        calls.append(content)
        if body["model"] == EDITOR:
            return response({"sentence_index": 0, "replacement": edited})
        if "document_context" not in content and final_verdict != "accept":
            issue = {
                "severity": "material",
                "category": "scope",
                "source_quote": "does not stop"
                if final_verdict == "reject"
                else "fabricated evidence",
                "candidate_quote": "keeps running",
                "explanation": "The reviewer flags a change in the persistence condition.",
            }
            return response({"verdict": "reject", "issues": [issue]})
        return response({"verdict": "accept", "issues": []})

    deps = dependencies(call)
    deps["guard"] = AdvisoryGuard()
    result = run(source, tmp_path, deps=deps, review_policy="contextual")
    assert result["text"] == (edited if final_verdict == "accept" else source)
    assert result["review_policy"] == "contextual"
    assert (
        result["final_review"]["review_status"]
        == {"accept": "accepted", "reject": "rejected", "unavailable": "unverified"}[
            final_verdict
        ]
    )
    assert "negation changed" in result["final_review"]["advisory_issues"]
    assert not result["final_review"]["nli"]["passed"]
    assert result["candidates"]
    assert all(row["candidate_text"] == edited for row in result["candidates"])
    assert all(row["review_status"] == "accepted" for row in result["candidates"])
    if final_verdict == "unavailable":
        assert result["provider_status"] == "failed"
        assert result["attempted_text"] == edited
    elif final_verdict == "reject":
        assert result["attempted_text"] == edited
        assert (
            result["final_review"]["review"]["evidence"]["issues"][0]["source_quote"]
            == "does not stop"
        )
    before = len(calls)
    assert run(source, tmp_path, deps=deps, review_policy="contextual") == result
    assert len(calls) == before
    with pytest.raises(ValueError, match="configuration or code changed"):
        run(source, tmp_path, deps=deps)


def test_contextual_section_review_receives_full_source_beyond_neighbor_window(
    tmp_path,
):
    source = (SOURCE + "\n\n") * 160
    assert len(source) > 12_000
    reviewed = []
    generate = Calls()

    def call(body):
        if body["model"] == EDITOR:
            return generate(body)
        content = json.loads(body["messages"][1]["content"])
        if "document_context" in content:
            context = content["document_context"]
            assert context["scope"] == "full_document"
            assert context["before"] == source
            assert len(context["after"]) > 12_000
            reviewed.append(context)
            # Valid source evidence can be outside the edited local section.
            assert len(content["before"]) < len(context["before"])
        return response({"verdict": "accept", "issues": []})

    result = run(
        source,
        tmp_path,
        deps=dependencies(call),
        review_policy="contextual",
        section_tokens=200,
    )
    assert result["changed"] is True
    assert reviewed


def test_contextual_policy_does_not_relax_detector_token_limit(tmp_path):
    edited = "The gentle river flows next to " + "the village " * 45 + "."
    calls = []

    def call(body):
        calls.append(body)
        assert body["model"] == EDITOR
        return response({"sentence_index": 0, "replacement": edited})

    result = run(SOURCE, tmp_path, deps=dependencies(call), review_policy="contextual")
    assert result["text"] == SOURCE
    assert all(
        "edited_section_exceeds_detector_limit" in row["mechanical"]["issues"]
        for row in result["candidates"]
    )
    assert len(calls) == 2


def test_contextual_policy_requires_adaptive_strategy_before_initialization(tmp_path):
    with pytest.raises(ValueError, match="sentence editing only"):
        process_text(
            SOURCE,
            output_dir=tmp_path,
            strategy="technical-update-experimental",
            review_policy="contextual",
        )


def test_contextual_unavailable_final_review_retains_combined_formatted_draft(tmp_path):
    source = "# Report\n\n" + (SOURCE + "\n\n") * 10
    generate = Calls()

    def call(body):
        if body["model"] == EDITOR:
            return generate(body)
        content = json.loads(body["messages"][1]["content"])
        if "document_context" not in content:
            return {"raw_response": "", "finish_reason": "transport_error"}
        return response({"verdict": "accept", "issues": []})

    result = run(
        source,
        tmp_path,
        deps=dependencies(call),
        review_policy="contextual",
        section_tokens=200,
    )
    assert result["section_count"] > 1
    assert result["text"] == source
    assert result["review_status"] == "unverified"
    assert result["provider_status"] == "failed"
    assert result["attempted_text"].startswith("# Report\n\n")
    assert result["attempted_text"].replace("next to", "beside") == source
    assert result["attempted_text"].count("next to") > max(
        row["candidate_text"].count("next to") for row in result["candidates"]
    )


def test_contextual_requires_distinct_resolved_models_before_loading_dependencies(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("FAKEHUMAN_EDITOR_MODEL", "same/model")
    monkeypatch.setenv("FAKEHUMAN_REVIEWER_MODEL", "same/model")
    with pytest.raises(ValueError, match="different editor and reviewer models"):
        process_text(SOURCE, output_dir=tmp_path, review_policy="contextual")
    assert not (tmp_path / "protocol.json").exists()


@pytest.mark.parametrize("outcome", ["accepted", "rejected", "failed"])
def test_costs_include_all_search_and_final_requests_and_survive_resume(
    tmp_path, outcome
):
    calls = Calls(reject_final=outcome == "rejected", fail_final=outcome == "failed")

    def costed_call(body):
        result = calls(body)
        result["usage"] = {
            "cost": 0,
            "cost_details": {"upstream_inference_cost": 0.001},
        }
        return result

    deps = dependencies(costed_call)
    result = run(SOURCE, tmp_path, deps=deps)
    summary = result["cost_summary"]
    assert summary["request_count"] == len(calls.bodies) == 5
    known = 4 if outcome == "failed" else 5
    assert summary["gateway_cost_usd"]["total"] == 0
    assert summary["upstream_cost_usd"]["total"] == pytest.approx(known * 0.001)
    assert summary["upstream_cost_usd"]["reported_requests"] == known
    assert summary["upstream_cost_usd"]["missing_requests"] == 5 - known
    assert json.loads((tmp_path / "result.json").read_text())["cost_summary"] == summary
    assert run(SOURCE, tmp_path, deps=deps)["cost_summary"] == summary
    assert len(calls.bodies) == 5
