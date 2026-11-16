# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: William Droz <william.droz@idiap.ch>
# SPDX-License-Identifier: GPL-3.0-only

import json

import pytest

from fakehuman.adaptive import (
    EDITOR,
    REVIEWER,
    _review,
    apply_candidate,
    parse_candidate,
    parse_review,
    run_document,
    select_winner,
    sentence_spans,
)

SOURCE = (
    "The gentle river flows beside the village. The stone bridge crosses the river."
)
SYNTAX = "Beside the village, the gentle river flows."
DICTION = "The gentle river passes beside the village."
LAST = "Across the river stands the stone bridge."
PASS = {
    "passed": True,
    "reason": "The same facts remain fluent.",
    "category": "faithful",
}


def response(value, finish_reason="stop"):
    return {
        "raw_response": json.dumps(value),
        "finish_reason": finish_reason,
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }


class Guard:
    def __init__(self, passed=True):
        self.pairs = []
        self.passed = passed

    def metadata(self):
        return {"model_id": "test-guard", "loaded": bool(self.pairs)}

    def check_pairs(self, pairs):
        self.pairs.extend(pairs)
        return [{"passed": self.passed} for _ in pairs]


class Detector:
    def __init__(self, scores=None, events=None):
        self.scores = scores or {}
        self.inputs = []
        self.events = events if events is not None else []

    def metadata(self):
        return {"name": "fakespot_roberta", "loaded": bool(self.inputs)}

    def score(self, texts):
        self.inputs.extend(texts)
        self.events.extend(("score", text) for text in texts)
        return [self.scores.get(text, 0.8) for text in texts]


class Calls:
    def __init__(self, proposals=None, reviews=None, events=None):
        self.proposals = proposals or {}
        self.reviews = iter(reviews) if reviews is not None else None
        self.bodies = []
        self.events = events if events is not None else []

    def __call__(self, body):
        self.bodies.append(body)
        content = json.loads(body["messages"][1]["content"])
        if body["model"] == EDITOR:
            self.events.append(("generate", content["current"]))
            proposal = self.proposals.get((content["round"], content["style"]))
            if proposal is None:
                proposal = {
                    "sentence_index": 0,
                    "replacement": content["sentences"][0]["text"],
                }
            return proposal if "raw_response" in proposal else response(proposal)
        assert body["model"] == REVIEWER
        assert set(content) == {"before", "after"}
        assert content["before"] == SOURCE
        self.events.append(("review", content["after"]))
        return next(self.reviews) if self.reviews is not None else response(PASS)


def test_sentence_replacement_preserves_all_separators():
    text = "  The river flows.\n\n  The bridge stands.  "
    assert [text[start:end] for start, end in sentence_spans(text)] == [
        "The river flows.",
        "The bridge stands.",
    ]
    assert apply_candidate(
        text, {"sentence_index": 1, "replacement": "The bridge remains."}
    ) == ("  The river flows.\n\n  The bridge remains.  ")
    assert (
        apply_candidate(text, {"sentence_index": 0, "replacement": "The river flows."})
        == text
    )


@pytest.mark.parametrize(
    "payload",
    [
        '{"sentence_index": true, "replacement": "Words."}',
        '{"sentence_index": 0, "replacement": "Words.", "extra": 1}',
        '{"sentence_index": 0, "sentence_index": 1, "replacement": "Words."}',
        '{"sentence_index": NaN, "replacement": "Words."}',
        '[0, "Words."]',
    ],
)
def test_candidate_json_is_strict(payload):
    with pytest.raises(ValueError):
        parse_candidate(payload)


@pytest.mark.parametrize(
    "proposal",
    [
        {"sentence_index": -1, "replacement": "The river flows."},
        {"sentence_index": 9, "replacement": "The river flows."},
        {"sentence_index": 0, "replacement": "Two sentences. Another sentence."},
        {"sentence_index": 0, "replacement": "Two sentences.Another sentence."},
        {"sentence_index": 0, "replacement": "The river flows"},
        {"sentence_index": 0, "replacement": " The river flows."},
        {"sentence_index": 0, "replacement": "The river\u200b flows."},
        {"sentence_index": 0, "replacement": "The river\u00a0flows."},
    ],
)
def test_invalid_replacement_is_refused(proposal):
    with pytest.raises(ValueError):
        apply_candidate(SOURCE, proposal)


@pytest.mark.parametrize(
    "value",
    [
        {**PASS, "passed": 1},
        {**PASS, "passed": False},
        {**PASS, "category": "uncertain"},
        {**PASS, "reason": " "},
        {**PASS, "category": ["faithful"]},
        {**PASS, "extra": "ignored?"},
    ],
)
def test_review_requires_unambiguous_boolean_and_consistent_category(value):
    with pytest.raises(ValueError):
        parse_review(json.dumps(value))


def test_selection_requires_strict_improvement_and_fidelity():
    near = {"score": 0.8 - 0.5e-6, "fidelity_passed": True}
    rejected = {"score": 0.1, "fidelity_passed": False}
    assert select_winner(0.8, [near, rejected]) is None
    first = {"score": 0.2, "fidelity_passed": True, "candidate_key": "a"}
    tie = {"score": 0.2, "fidelity_passed": True, "candidate_key": "b"}
    assert select_winner(0.8, [first, tie]) is first
    with pytest.raises(ValueError):
        select_winner(0.8, [{"score": float("nan"), "fidelity_passed": True}])


def test_search_arms_score_order_and_cumulative_reference(tmp_path):
    first = apply_candidate(SOURCE, {"sentence_index": 0, "replacement": SYNTAX})
    second = apply_candidate(SOURCE, {"sentence_index": 0, "replacement": DICTION})
    final = apply_candidate(second, {"sentence_index": 1, "replacement": LAST})
    events = []
    calls = Calls(
        {
            (1, "syntax"): {"sentence_index": 0, "replacement": SYNTAX},
            (1, "diction"): {"sentence_index": 0, "replacement": DICTION},
            (2, "syntax"): {"sentence_index": 1, "replacement": LAST},
        },
        events=events,
    )
    detector = Detector({SOURCE: 0.8, first: 0.9, second: 0.3, final: 0.2}, events)
    guard = Guard()
    result = run_document(
        tmp_path,
        {"id": "one", "text": SOURCE},
        call=calls,
        guard=guard,
        detector=detector,
    )
    assert result["arms"]["one_shot"]["text"] == first
    assert result["arms"]["best_of_2"]["text"] == second
    assert result["arms"]["adaptive"]["text"] == final
    assert len(result["candidates"]) == 6
    assert [row["score"] for row in result["rounds"]] == [0.3, 0.2, 0.2]
    assert (SOURCE, final) in guard.pairs
    # No drift check is limited to the most recent selected passage.
    assert (second, final) not in guard.pairs
    assert events.index(("review", first)) < events.index(("score", first))
    assert events.index(("review", final)) < events.index(("score", final))
    gen_inputs = [
        json.loads(body["messages"][1]["content"])
        for body in calls.bodies
        if body["model"] == EDITOR
    ]
    assert gen_inputs[0]["current"] == gen_inputs[1]["current"] == SOURCE
    assert gen_inputs[2]["current"] == gen_inputs[3]["current"] == second
    assert gen_inputs[4]["current"] == gen_inputs[5]["current"] == final
    assert len(gen_inputs) == 6
    before = (len(calls.bodies), len(guard.pairs), len(detector.inputs))
    assert (
        run_document(
            tmp_path,
            {"id": "one", "text": SOURCE},
            call=calls,
            guard=guard,
            detector=detector,
        )
        == result
    )
    assert before == (len(calls.bodies), len(guard.pairs), len(detector.inputs))


def test_refusals_are_retained_without_detector_scoring(tmp_path):
    calls = Calls(
        {
            (1, "syntax"): {
                "sentence_index": 0,
                "replacement": "The river flows beside 37 villages.",
            },
            (1, "diction"): {"sentence_index": 0, "replacement": DICTION},
        },
        reviews=[
            response(
                {
                    "passed": False,
                    "reason": "The referent changed.",
                    "category": "semantic_drift",
                }
            )
        ],
    )
    detector = Detector()
    result = run_document(
        tmp_path,
        {"id": "refused", "text": SOURCE},
        call=calls,
        guard=Guard(),
        detector=detector,
        rounds=1,
    )
    mechanical, semantic = result["candidates"]
    assert mechanical["disposition"] == "mechanical_refusal"
    assert mechanical["proposal"]["replacement"].endswith("37 villages.")
    assert mechanical["review"] is None
    assert semantic["disposition"] == "context_refusal"
    assert semantic["attempted"] != SOURCE and semantic["output"] == SOURCE
    assert detector.inputs == [SOURCE]
    assert all(arm["text"] == SOURCE for arm in result["arms"].values())


def test_nli_refusal_still_receives_independent_review_but_no_score(tmp_path):
    calls = Calls(
        {
            (1, style): {"sentence_index": 0, "replacement": DICTION}
            for style in ("syntax", "diction")
        }
    )
    detector = Detector()
    result = run_document(
        tmp_path,
        {"id": "nli", "text": SOURCE},
        call=calls,
        guard=Guard(False),
        detector=detector,
        rounds=1,
    )
    assert all(row["disposition"] == "nli_refusal" for row in result["candidates"])
    assert all(row["review"]["decision"]["passed"] for row in result["candidates"])
    assert detector.inputs == [SOURCE]


def test_contextual_policy_can_keep_a_faithful_edit_despite_local_warnings(tmp_path):
    source = "The server does not stop when the client disconnects."
    edited = "The server keeps running when the client disconnects."

    def call(body):
        if body["model"] == EDITOR:
            return response({"sentence_index": 0, "replacement": edited})
        assert (
            body["response_format"]["json_schema"]["name"]
            == "technical_fidelity_review"
        )
        content = json.loads(body["messages"][1]["content"])
        assert content == {"before": source, "after": edited}
        return response({"verdict": "accept", "issues": []})

    detector, guard = Detector({edited: 0.2}), Guard(False)
    result = run_document(
        tmp_path,
        {"id": "contextual", "text": source},
        call=call,
        guard=guard,
        detector=detector,
        rounds=1,
        review_policy="contextual",
    )
    assert result["arms"]["adaptive"]["text"] == edited
    assert all(row["review_status"] == "accepted" for row in result["candidates"])
    assert all(
        "negation changed" in row["advisory_issues"] for row in result["candidates"]
    )
    assert all(not row["nli"]["passed"] for row in result["candidates"])
    assert all(
        row["review"]["evidence"]["status"] == "accept" for row in result["candidates"]
    )
    with pytest.raises(ValueError, match="integrity/binding mismatch"):
        run_document(
            tmp_path,
            {"id": "contextual", "text": source},
            call=call,
            guard=guard,
            detector=detector,
            rounds=1,
        )


@pytest.mark.parametrize("fabricated_quote", [False, True])
def test_contextual_material_findings_and_unavailable_evidence_block_selection(
    tmp_path, fabricated_quote
):
    edited = "The gentle river flows beside 37 villages."
    issue = {
        "severity": "material",
        "category": "quantity",
        "source_quote": "Fabricated quote" if fabricated_quote else "the village",
        "candidate_quote": "37 villages",
        "explanation": "One village becomes 37 villages.",
    }
    calls = Calls(
        {
            (1, style): {"sentence_index": 0, "replacement": edited}
            for style in ("syntax", "diction")
        },
        reviews=[response({"verdict": "reject", "issues": [issue]})] * 2,
    )
    detector = Detector()
    result = run_document(
        tmp_path,
        {"id": "material", "text": SOURCE},
        call=calls,
        guard=Guard(False),
        detector=detector,
        rounds=1,
        review_policy="contextual",
    )
    assert result["arms"]["adaptive"]["text"] == SOURCE
    assert detector.inputs == [SOURCE]
    assert all(
        row["review_status"] == ("unverified" if fabricated_quote else "rejected")
        for row in result["candidates"]
    )
    assert all(
        row["review"]["evidence"]["issues"][0]["source_quote"] == issue["source_quote"]
        for row in result["candidates"]
    )


@pytest.mark.parametrize("constraint", ["token_limit", "markup", "hidden_character"])
def test_contextual_policy_keeps_hard_constraints_before_model_review(
    tmp_path, constraint
):
    class LimitedGuard(Guard):
        def hard_issues(self, before, after):
            return (
                ["edited_section_exceeds_detector_limit"]
                if constraint == "token_limit"
                else []
            )

    edited = {
        "token_limit": DICTION,
        "markup": "The gentle **river** passes beside the village.",
        "hidden_character": "The gentle river\ufe0f passes beside the village.",
    }[constraint]
    calls = Calls(
        {
            (1, style): {"sentence_index": 0, "replacement": edited}
            for style in ("syntax", "diction")
        }
    )
    guard, detector = LimitedGuard(), Detector()
    result = run_document(
        tmp_path,
        {"id": constraint, "text": SOURCE},
        call=calls,
        guard=guard,
        detector=detector,
        rounds=1,
        review_policy="contextual",
    )
    assert all(
        row["disposition"] == "mechanical_refusal" for row in result["candidates"]
    )
    assert all(row["review"] is None for row in result["candidates"])
    assert guard.pairs == []
    assert detector.inputs == [SOURCE]


@pytest.mark.parametrize(
    "submitted_request",
    [
        {},
        {"messages": []},
        {"messages": [{}, {"content": "[]"}]},
        {"messages": [{}, {"content": "invalid"}]},
    ],
)
def test_contextual_malformed_submitted_context_is_unverified(submitted_request):
    review = _review(
        {
            **response({"verdict": "accept", "issues": []}),
            "submitted_request": submitted_request,
        },
        original=SOURCE,
        attempted=SOURCE,
        review_policy="contextual",
    )
    assert review["decision"]["passed"] is False
    assert review["evidence"]["status"] == "unavailable"


def test_contextual_requires_an_independent_model_before_any_calls(tmp_path):
    with pytest.raises(ValueError, match="different editor and reviewer models"):
        run_document(
            tmp_path,
            {"id": "same", "text": SOURCE},
            call=lambda body: pytest.fail("Remote call before model validation"),
            guard=None,
            detector=None,
            editor_model="same/model",
            reviewer_model="same/model",
            review_policy="contextual",
        )


def test_failed_generation_and_incomplete_review_fail_closed(tmp_path):
    calls = Calls(
        {(1, "syntax"): {"raw_response": "invalid", "finish_reason": "length"}},
        reviews=[response(PASS, finish_reason="length")],
    )
    result = run_document(
        tmp_path,
        {"id": "bad-response", "text": SOURCE},
        call=calls,
        guard=Guard(),
        detector=Detector(),
        rounds=1,
    )
    assert result["candidates"][0]["mechanical"]["issues"] == [
        "Generation did not finish normally"
    ]
    assert result["candidates"][1]["disposition"] == "context_operational_failure"
    assert not any(row["fidelity_passed"] for row in result["candidates"])


def test_interruption_reuses_durable_generation_then_resumes_review(tmp_path):
    calls = Calls()

    def interrupted(body):
        if body["model"] == REVIEWER:
            raise KeyboardInterrupt
        return calls(body)

    with pytest.raises(KeyboardInterrupt):
        run_document(
            tmp_path,
            {"id": "resume", "text": SOURCE},
            call=interrupted,
            guard=Guard(),
            detector=Detector(),
            rounds=1,
        )
    assert len(calls.bodies) == 1
    completed = Calls()
    result = run_document(
        tmp_path,
        {"id": "resume", "text": SOURCE},
        call=completed,
        guard=Guard(),
        detector=Detector(),
        rounds=1,
    )
    assert completed.bodies[0]["model"] == REVIEWER
    assert sum(body["model"] == EDITOR for body in completed.bodies) == 1
    assert len(result["candidates"]) == 2


def test_changed_source_and_corrupted_checkpoint_cannot_resume(tmp_path):
    document = {"id": "integrity", "text": SOURCE}
    calls = Calls()
    run_document(
        tmp_path, document, call=calls, guard=Guard(), detector=Detector(), rounds=1
    )
    with pytest.raises(ValueError, match="integrity/binding"):
        run_document(
            tmp_path,
            {**document, "text": SOURCE + " Another sentence."},
            call=calls,
            guard=Guard(),
            detector=Detector(),
            rounds=1,
        )
    path = next(tmp_path.glob("search/*/round_1_candidate_1/generation.json"))
    checkpoint = json.loads(path.read_text())
    checkpoint["payload"]["raw_response"] = "tampered"
    path.write_text(json.dumps(checkpoint))
    before = len(calls.bodies)
    with pytest.raises(ValueError, match="integrity/binding"):
        run_document(
            tmp_path, document, call=calls, guard=Guard(), detector=Detector(), rounds=1
        )
    assert len(calls.bodies) == before


def test_desklib_cannot_enter_search(tmp_path, monkeypatch):
    detector = Detector()
    monkeypatch.setattr(detector, "metadata", lambda: {"name": "desklib_deberta"})
    with pytest.raises(ValueError, match="Only fakespot"):
        run_document(
            tmp_path,
            {"id": "held-out", "text": SOURCE},
            call=Calls(),
            guard=Guard(),
            detector=detector,
        )
    assert detector.inputs == []
