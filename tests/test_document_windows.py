# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: William Droz <william.droz@idiap.ch>
# SPDX-License-Identifier: GPL-3.0-only

"""Offline coverage, native-score, and boundary retokenization checks."""

import re
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast

import pytest

from fakehuman.detectors import (
    DETECTOR_SPECS,
    TransformerDetector,
    fakespot_clean_text,
)
from fakehuman.document_chunks import TokenCounter
from fakehuman.document_windows import (
    score_document_windows,
    window_spans,
)


class WordTokenizer:
    is_fast = True
    token_pattern = r"\S+"

    def __call__(self, text, **kwargs):
        if isinstance(text, list):
            return {"input_ids": [self(item, **kwargs)["input_ids"] for item in text]}
        spans = [match.span() for match in re.finditer(self.token_pattern, text)]
        ids = list(range(len(spans)))
        if kwargs.get("add_special_tokens"):
            ids = [0, *ids, 0]
        return {"input_ids": ids, "offset_mapping": spans}

    def num_special_tokens_to_add(self, pair):
        assert pair is False
        return 2


class Counter:
    def __init__(self, *, limit=512, clean=lambda value: value, surcharge=0):
        self.spec = SimpleNamespace(max_tokens=limit)
        self._tokenizer = WordTokenizer()
        self.clean = clean
        self.surcharge = surcharge

    def prepare(self, text):
        return self.clean(text)

    def __call__(self, text):
        return len(self.prepare(text).split()) + 2 + self.surcharge

    def metadata(self):
        return {"preprocessing": "test", "max_tokens": self.spec.max_tokens}


class Detector:
    def __init__(self, scores=None):
        self.calls = []
        self.scores = scores

    def score(self, texts):
        self.calls.append(texts)
        return self.scores if self.scores is not None else [0.25] * len(texts)


def diagnose(text, *, counter=None, detector=None, stride=256):
    counter = counter or Counter()
    detector = detector or Detector()
    result = score_document_windows(
        text, detectors={"test": detector}, counters={"test": counter}, stride=stride
    )
    return result["detectors"]["test"], detector


@pytest.mark.parametrize(
    "count", [0, 1, 255, 256, 509, 510, 511, 512, 766, 767, 1020, 1021, 5000]
)
def test_offsets_cover_every_token_without_short_tail(count):
    offsets = [(index * 3, index * 3 + 2) for index in range(count)]
    spans = window_spans(offsets, budget=510, stride=256)
    covered = set()
    for span in spans:
        covered.update(range(span.token_start, span.token_end))
        assert span.start == offsets[span.token_start][0]
        assert span.end == offsets[span.token_end - 1][1]
        assert span.token_end - span.token_start == min(count, 510)
    assert covered == set(range(count))
    if count:
        assert spans[0].token_start == 0
        assert spans[-1].token_end == count
        assert len({span.token_start for span in spans}) == len(spans)


def test_final_window_aligns_to_end_and_keeps_regular_starts():
    spans = window_spans([(i, i + 1) for i in range(1021)], budget=510, stride=256)
    assert [span.token_start for span in spans] == [0, 256, 511]


@pytest.mark.parametrize("budget,stride", [(0, 1), (10, 0), (10, 11), (True, 1)])
def test_invalid_budget_or_stride_cannot_produce_coverage_gaps(budget, stride):
    with pytest.raises(ValueError):
        window_spans([(0, 1)], budget=budget, stride=stride)


@pytest.mark.parametrize("offsets", [[(-1, 1)], [(2, 1)], [(3, 4), (1, 2)], [(0, "1")]])
def test_invalid_offsets_fail(offsets):
    with pytest.raises(ValueError):
        window_spans(offsets, budget=10, stride=5)


def test_overlapping_unicode_character_offsets_are_allowed():
    spans = window_spans([(0, 1), (0, 1), (1, 2)], budget=2, stride=1)
    assert [(span.start, span.end) for span in spans] == [(0, 1), (0, 2)]


def test_short_document_uses_exact_original_and_native_score():
    text = " **Some prose** \n"
    detector = Detector([0.7])
    result, _ = diagnose(
        text,
        counter=Counter(clean=lambda text: text.replace("**", "")),
        detector=detector,
    )
    assert detector.calls == [[text]]
    assert result["method"] == "single_window_native"
    assert result["mean"] == result["min"] == result["max"] == 0.7
    assert result["windows"][0]["end"] == len(" Some prose \n")


def test_short_document_stays_native_when_prepared_apis_are_available():
    class PreparedDetector(Detector):
        def score_prepared(self, _texts):
            pytest.fail("Short document must use the native scoring path")

    text = "**Some prose** &lt;literal&gt;"
    counter = TokenCounter(DETECTOR_SPECS["fakespot_roberta"])
    counter._tokenizer = WordTokenizer()
    detector = PreparedDetector([0.7])
    result, _ = diagnose(text, counter=counter, detector=detector)
    assert detector.calls == [[text]]
    assert result["method"] == "single_window_native"
    assert result["mean"] == 0.7


def test_exact_512_tokens_stays_native():
    result, detector = diagnose(" ".join(["word"] * 510))
    assert result["method"] == "single_window_native"
    assert result["full_token_count"] == 512
    assert len(detector.calls[0]) == 1


def test_first_overflow_uses_complete_overlapping_windows():
    text = " ".join(f"w{i}" for i in range(511))
    result, detector = diagnose(text, detector=Detector([0.1, 0.9]))
    assert result["method"] == "descriptive_overlapping_window_mean"
    assert result["mean"] == 0.5
    assert result["min"] == 0.1
    assert result["max"] == 0.9
    assert result["covered_token_count"] == 511
    assert result["token_coverage_complete"] is True
    for window, actual in zip(result["windows"], detector.calls[0]):
        assert actual == text[window["start"] : window["end"]]
        assert window["token_count"] == 512


def test_offsets_reference_complete_prepared_text():
    text = "**" + " \n".join(f"é{i}" for i in range(800)) + "**"
    counter = Counter(clean=lambda text: text.replace("**", ""))
    result, detector = diagnose(text, counter=counter)
    prepared = counter.prepare(text)
    for window, actual in zip(result["windows"], detector.calls[0]):
        assert actual == prepared[window["start"] : window["end"]]
    assert result["prepared_characters"] == len(prepared)
    assert result["covered_token_count"] == 800


def test_retokenization_overflow_reduces_budget_by_eight():
    result, detector = diagnose(" ".join(["word"] * 800), counter=Counter(surcharge=3))
    assert result["content_budget"] == 502
    assert result["budget_reductions"] == 1
    assert result["covered_token_count"] == 800
    assert all(window["token_count"] <= 512 for window in result["windows"])
    assert len(detector.calls) == 1


def test_windows_include_tokenized_unicode_whitespace():
    class CharacterTokenizer(WordTokenizer):
        token_pattern = r"[\s\S]"

    text = "Start" + "\u2003" * 24 + "end"
    spec = replace(DETECTOR_SPECS["fakespot_roberta"], max_tokens=10)
    counter = TokenCounter(spec)
    counter._tokenizer = CharacterTokenizer()
    detector = TransformerDetector(spec)
    detector._tokenizer = counter._tokenizer
    detector._model = object()
    detector._torch = cast(Any, object())
    spans = window_spans(counter._tokenizer(text)["offset_mapping"], budget=8, stride=4)
    parts = [text[span.start : span.end] for span in spans]
    assert any(not part.strip() for part in parts)
    detector._cache = dict.fromkeys(parts, 0.25)
    with pytest.raises(ValueError, match="nonempty string"):
        detector.score_prepared(["\u2003" * 8])

    result, _ = diagnose(text, counter=counter, detector=detector, stride=4)

    assert result["available"] is True
    assert result["mean"] == 0.25
    assert result["token_coverage_complete"] is True
    assert result["covered_token_count"] == len(text)
    assert len(result["windows"]) == len(parts)
    assert all(window["token_count"] == spec.max_tokens for window in result["windows"])


def test_unsafe_budget_is_unavailable_without_scoring():
    result, detector = diagnose(
        " ".join(["word"] * 600), counter=Counter(surcharge=300)
    )
    assert result["available"] is False
    assert result["mean"] is None
    assert "coverage would have gaps" in result["reason"]
    assert detector.calls == []


def test_blank_preparation_is_null_without_scoring():
    result, detector = diagnose("```only code```", counter=Counter(clean=lambda _: ""))
    assert result["available"] is False
    assert result["mean"] is result["min"] is result["max"] is None
    assert detector.calls == []


def test_non_idempotent_boundary_cleanup_is_reported():
    text = "prefix " + " ".join(["word"] * 800)
    # Full cleaning removes prefix; a subsequent cleaning would remove word.
    counter = Counter(clean=lambda text: text.split(" ", 1)[-1])
    result, detector = diagnose(text, counter=counter)
    assert result["available"] is False
    assert "repeated detector preprocessing" in result["reason"]
    assert detector.calls == []


@pytest.mark.parametrize(
    "text",
    [
        "prefix words before > retain all these words after the boundary",
        "prefix words before 1. retain this item number and all its words",
        "prefix &lt;literal&gt; retained &lt;/literal&gt; and more words fill the document",
    ],
)
def test_prepared_apis_clean_full_document_once_and_preserve_all_window_content(
    monkeypatch, text
):
    from fakehuman import detectors, document_chunks

    prepared = fakespot_clean_text(text)
    spec = replace(DETECTOR_SPECS["fakespot_roberta"], max_tokens=8)
    counter = TokenCounter(spec)
    counter._tokenizer = WordTokenizer()
    detector = TransformerDetector(spec)
    detector._tokenizer = counter._tokenizer
    detector._model = object()
    detector._torch = cast(Any, object())
    expected_spans = window_spans(
        counter._tokenizer(prepared)["offset_mapping"], budget=6, stride=3
    )
    parts = [prepared[span.start : span.end] for span in expected_spans]
    assert any(fakespot_clean_text(part) != part for part in parts)
    # Cached scores still pass through the real adapter's length validation;
    # no model download or synthetic inference implementation is needed here.
    detector._cache = dict.fromkeys(parts, 0.25)
    cleaned = []

    def clean_once(value):
        cleaned.append(value)
        return fakespot_clean_text(value)

    monkeypatch.setattr(document_chunks, "fakespot_clean_text", clean_once)
    monkeypatch.setattr(
        detectors,
        "fakespot_clean_text",
        lambda _: pytest.fail("Prepared window was cleaned again before scoring"),
    )
    result, _ = diagnose(text, counter=counter, detector=detector, stride=3)
    assert result["available"] is True
    assert result["prepared_scoring"] is True
    assert result["token_coverage_complete"] is True
    assert result["covered_token_count"] == len(prepared.split())
    assert cleaned == [text]
    for window, part in zip(result["windows"], parts, strict=True):
        assert prepared[window["start"] : window["end"]] == part
        assert window["token_count"] == len(part.split()) + 2
    assert all(item["input_is_prepared"] for item in detector.last_input_report)
    assert all(not item["truncated"] for item in detector.last_input_report)


@pytest.mark.parametrize("missing_api", ["count_prepared", "score_prepared"])
def test_partial_prepared_support_retains_guarded_legacy_fallback(missing_api):
    class PreparedCounter(Counter):
        def count_prepared(self, text):
            return len(text.split()) + 2

    class PreparedDetector(Detector):
        score_prepared = Detector.score

    counter = PreparedCounter(clean=lambda value: value.split(" ", 1)[-1])
    detector = PreparedDetector()
    adapter = counter if missing_api == "count_prepared" else detector
    setattr(adapter, missing_api, None)
    result, _ = diagnose(
        "prefix " + " ".join(["word"] * 800), counter=counter, detector=detector
    )
    assert result["available"] is False
    assert "repeated detector preprocessing" in result["reason"]
    assert detector.calls == []


def test_missing_fast_tokenizer_is_explicitly_unavailable():
    counter = Counter()
    counter._tokenizer.is_fast = False
    result, detector = diagnose("words here", counter=counter)
    assert result["available"] is False
    assert "fast tokenizer" in result["reason"]
    assert detector.calls == []


@pytest.mark.parametrize("scores", [[float("nan")], [1.1], [-0.1], []])
def test_invalid_detector_scores_fail(scores):
    with pytest.raises(ValueError):
        diagnose("Some prose.", detector=Detector(scores))


def test_detector_names_must_match():
    with pytest.raises(ValueError, match="matching"):
        score_document_windows("prose", detectors={"a": Detector()}, counters={})


def test_service_scores_large_text_and_includes_the_final_window(monkeypatch):
    from fakehuman.cli import load_manifest
    from fakehuman.service import TextService

    spec = DETECTOR_SPECS["fakespot_roberta"]
    tokenizer = WordTokenizer()
    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained", lambda *args, **kwargs: tokenizer
    )
    detector = TransformerDetector(spec)
    detector._tokenizer = tokenizer
    detector._model = object()
    detector._torch = cast(Any, object())
    text = " ".join(f"document-word-{index}" for index in range(6000))
    assert len(text) > 100_000
    spans = window_spans(tokenizer(text)["offset_mapping"], budget=510)
    parts = [text[span.start : span.end] for span in spans]
    detector._cache = dict.fromkeys(parts, 0.25)
    detector._cache[parts[-1]] = 0.75
    service = TextService(load_manifest(), detectors={spec.name: detector})

    report = service.score_text(text, detailed=True)
    details = report["detectors"][spec.name]
    assert details["input"]["tokens"] == 6002
    assert details["input"]["truncated"] is False
    assert details["input"]["prepared_text"] == text
    windows = details["window_scoring"]
    assert windows["covered_token_count"] == 6000
    assert windows["token_coverage_complete"] is True
    assert windows["windows"][-1]["end"] == len(text)
    assert windows["windows"][-1]["score"] == 0.75
    assert report["scores"][spec.name] == pytest.approx(0.25 + 0.5 / len(spans))
    assert service.score_text(text) == report["scores"]
