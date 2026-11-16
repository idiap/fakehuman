# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: William Droz <william.droz@idiap.ch>
# SPDX-License-Identifier: GPL-3.0-only

"""Verify lossless long-document boundaries without model downloads/inference."""

import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest

from fakehuman.detectors import DETECTOR_SPECS, fakespot_clean_text
from fakehuman.document_chunks import (
    TokenCounter,
    aggregate_chunk_scores,
    build_token_counters,
    split_document,
)


def words(text):
    return len(text.split()) + 2


def characters(text):
    return len(text) + 2


def assert_lossless(text, chunks, counters, budget):
    assert "".join(chunk.text for chunk in chunks) == text
    assert chunks[0].start == 0
    assert chunks[-1].end == len(text)
    for index, chunk in enumerate(chunks):
        assert chunk.index == index
        assert text[chunk.start : chunk.end] == chunk.text
        assert chunk.end > chunk.start
        assert chunk.weight >= 1
        if index:
            assert chunks[index - 1].end == chunk.start
        for name, counter in counters.items():
            assert counter(chunk.text) <= budget
            assert chunk.token_counts[name] == counter(chunk.text)


@pytest.mark.parametrize(
    "text",
    [
        "# Heading\n\n**First** paragraph has several words.\n\n"
        "- **Item:** words with [a link](https://example.com).\n"
        "- Another item has several more words.\n\nBest,\\\n&#x20; William\n",
        "  Leading space.\n\n\nThree newlines.\tTabs remain.  \n",
        "Heading\r\n\r\nFirst section.\r\n- One item.\r\n- Another item.\r\n\r\n",
        "École 中文 works. 🧪 Science matters! Καλημέρα. Café preserves e\u0301.\n",
        "z" * 205,
    ],
)
def test_source_slices_preserve_every_character_and_both_budgets(text):
    counters = {"character_detector": characters, "word_detector": words}
    chunks = split_document(text, counters, target_tokens=42)
    assert_lossless(text, chunks, counters, 42)
    assert chunks == split_document(text, counters, target_tokens=42)


def test_exact_512_tokens_remains_one_native_section():
    text = "x" * 510
    chunks = split_document(
        text, {"first": characters, "second": words}, target_tokens=512
    )
    assert len(chunks) == 1
    assert chunks[0].text == text
    assert chunks[0].token_counts == {"first": 512, "second": 3}
    assert not chunks[0].warnings


def test_both_detectors_control_the_limit_not_only_the_first():
    text = "word " * 100
    counters = {"short": words, "long": characters}
    chunks = split_document(text, counters, target_tokens=50)
    assert len(chunks) > 2
    assert_lossless(text, chunks, counters, 50)


def test_balances_sections_instead_of_leaving_a_tiny_tail():
    text = " ".join(f"word{index}" for index in range(99))
    chunks = split_document(text, {"word_detector": words}, target_tokens=50)
    assert len(chunks) == 3
    assert min(chunk.weight for chunk in chunks) >= 25
    assert max(chunk.weight for chunk in chunks) <= 40


def test_prefers_paragraph_and_list_boundaries_when_substantive():
    paragraphs = ["First sentence with words.\n\n", "- Second item with words.\n"]
    text = "".join(paragraphs) + "- Third item with words.\n"
    chunks = split_document(text, {"word_detector": words}, target_tokens=9)
    assert len(chunks) == 3
    assert chunks[0].text == paragraphs[0]
    assert chunks[1].text == paragraphs[1]
    assert all(not chunk.warnings for chunk in chunks)


def test_sentence_boundaries_avoid_forced_split_warning():
    text = "First sentence has several words. Next sentence also has some words."
    chunks = split_document(text, {"word_detector": words}, target_tokens=9)
    assert len(chunks) == 2
    assert chunks[0].text == "First sentence has several words. "
    assert all(not chunk.warnings for chunk in chunks)


def test_no_whitespace_input_falls_back_to_unicode_character_boundaries():
    text = "🧪" * 105
    counters = {"unicode": characters}
    chunks = split_document(text, counters, target_tokens=20)
    assert_lossless(text, chunks, counters, 20)
    assert all("word was split" in chunk.warnings[-1] for chunk in chunks[:-1])
    assert "begins inside a sentence" in chunks[-1].warnings[0]


def test_one_character_that_cannot_fit_raises_instead_of_looping():
    with pytest.raises(ValueError, match="No source section"):
        split_document("abc", {"cannot_fit": lambda _: 10}, target_tokens=3)


def test_nonmonotonic_counting_still_measures_every_returned_section():
    def nonmonotonic(text):
        return len(text) + (15 if len(text) % 3 == 0 else 2)

    text = "abcdefghijklmn" * 10
    counters = {"nonmonotonic": nonmonotonic}
    chunks = split_document(text, counters, target_tokens=25)
    assert_lossless(text, chunks, counters, 25)


@pytest.mark.parametrize("text", ["", " \r\n\t"])
def test_empty_documents_are_rejected(text):
    with pytest.raises(ValueError, match="nonempty"):
        split_document(text, {"counter": words})


@pytest.mark.parametrize("budget", [0, -1, True, 2.5])
def test_invalid_budgets_are_rejected(budget):
    with pytest.raises(ValueError, match="positive integer"):
        split_document("text", {"counter": words}, target_tokens=budget)


def test_counter_configuration_and_values_are_validated():
    with pytest.raises(ValueError, match="At least one"):
        split_document("text", {})
    with pytest.raises(ValueError, match="nonnegative integers"):
        split_document("text", {"bad": lambda _: -1})
    assert list(build_token_counters()) == ["fakespot_roberta", "desklib_deberta"]
    with pytest.raises(ValueError, match="Unknown detector"):
        build_token_counters(["unknown"])
    with pytest.raises(ValueError, match="distinct"):
        build_token_counters(["fakespot_roberta"] * 2)


def test_token_counter_uses_pinned_tokenizer_preprocessing_and_special_tokens(
    monkeypatch,
):
    loads = []
    calls = []

    def tokenizer(text, **kwargs):
        calls.append((text, kwargs))
        return {"input_ids": list(range(words(text)))}

    def load(model_id, **kwargs):
        loads.append((model_id, kwargs))
        return tokenizer

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=load)),
    )
    spec = DETECTOR_SPECS["fakespot_roberta"]
    counter = TokenCounter(spec)
    assert loads == []
    text = "# **Headline**\nSee [paper](https://example.com) &amp; data."
    assert counter(text) == words(fakespot_clean_text(text))
    assert counter("More words") == 4
    assert loads == [
        (spec.model_id, {"revision": spec.revision, "trust_remote_code": False})
    ]
    assert calls[0] == (
        fakespot_clean_text(text),
        {"add_special_tokens": True, "truncation": False, "verbose": False},
    )
    assert counter.metadata()["revision"] == spec.revision
    assert TokenCounter(DETECTOR_SPECS["desklib_deberta"]).prepare(text) == text


def test_original_weights_are_fixed_even_when_rewrite_lengths_change():
    counters = {"first": words, "second": characters}
    chunks = split_document(
        "One two three four.\n\nFive six seven eight.",
        counters,
        target_tokens=26,
    )
    assert len(chunks) == 2
    chunks[0] = replace(chunks[0], weight=2)
    chunks[1] = replace(chunks[1], weight=6)
    rewritten = ["A very long rewrite of the first section.", "Short."]
    result = aggregate_chunk_scores(
        chunks,
        {"first": [0.8, 0.4], "second": [0.2, 0.6]},
        {"first": [0.4, 0.2], "second": [0.1, 0.3]},
        rewritten_texts=rewritten,
        token_counters=counters,
    )
    assert result["method"] == "original-word-weighted-section-mean"
    assert result["total_original_weight"] == 8
    assert result["detectors"]["first"] == pytest.approx(
        {"original": 0.5, "rewritten": 0.25, "delta": -0.25}
    )
    assert result["detectors"]["second"] == pytest.approx(
        {"original": 0.5, "rewritten": 0.25, "delta": -0.25}
    )
    assert result["sections"][0]["rewritten_token_counts"]["first"] == words(
        rewritten[0]
    )
    assert result["sections"][0]["original_token_counts"] == chunks[0].token_counts
    assert "not full-document probabilities" in result["score_interpretation"]


def test_one_section_reports_the_native_score():
    counters = {"counter": words}
    chunks = split_document("One sentence.", counters)
    result = aggregate_chunk_scores(
        chunks,
        {"counter": [0.73]},
        {"counter": [0.31]},
        rewritten_texts=["A sentence."],
        token_counters=counters,
    )
    assert result["method"] == "single-section-native"
    assert result["detectors"]["counter"]["original"] == 0.73


def test_missing_section_scores_are_not_dropped_or_reweighted():
    counters = {"missing": words, "complete": characters}
    chunks = split_document(
        "One two three four.\n\nFive six seven eight.",
        counters,
        target_tokens=26,
    )
    chunks[0] = replace(chunks[0], weight=2)
    chunks[1] = replace(chunks[1], weight=6)
    result = aggregate_chunk_scores(
        chunks,
        {"missing": [0.8, None], "complete": [0.8, 0.4]},
        {"missing": [0.4, None], "complete": [0.4, 0.2]},
        rewritten_texts=[chunk.text for chunk in chunks],
        token_counters=counters,
    )
    assert result["total_original_weight"] == 8
    assert [s["original_weight"] for s in result["sections"]] == [2, 6]
    incomplete = result["detectors"]["missing"]
    assert incomplete["original"] is None
    assert incomplete["rewritten"] is None
    assert incomplete["delta"] is None
    assert set(incomplete["unavailable_reasons"]) == {"original", "rewritten", "delta"}
    assert "Sections [1]" in incomplete["unavailable_reasons"]["original"]
    assert "No sections were excluded" in incomplete["unavailable_reasons"]["original"]
    assert result["detectors"]["complete"] == pytest.approx(
        {"original": 0.5, "rewritten": 0.25, "delta": -0.25}
    )
    complete_section = result["sections"][0]["scores"]["missing"]
    assert complete_section == pytest.approx(
        {"original": 0.8, "rewritten": 0.4, "delta": -0.4}
    )
    missing_section = result["sections"][1]["scores"]["missing"]
    assert missing_section["original"] is None
    assert missing_section["rewritten"] is None
    assert missing_section["delta"] is None
    assert "No scorable content" in missing_section["unavailable_reasons"]["original"]


@pytest.mark.parametrize("missing_side", ["original", "rewritten"])
def test_missing_score_on_one_side_keeps_other_side_available(missing_side):
    counters = {"counter": words}
    chunks = split_document("One sentence.", counters)
    originals = [None if missing_side == "original" else 0.73]
    rewrites = [None if missing_side == "rewritten" else 0.31]
    result = aggregate_chunk_scores(
        chunks,
        {"counter": originals},
        {"counter": rewrites},
        rewritten_texts=["A sentence."],
        token_counters=counters,
    )
    scores = result["detectors"]["counter"]
    assert scores["original"] == originals[0]
    assert scores["rewritten"] == rewrites[0]
    assert scores["delta"] is None
    assert set(scores["unavailable_reasons"]) == {missing_side, "delta"}
    assert set(result["sections"][0]["scores"]["counter"]["unavailable_reasons"]) == {
        missing_side,
        "delta",
    }


def test_fully_unscorable_native_section_reports_null_not_zero():
    counters = {"counter": words}
    chunks = split_document("````\n````", counters)
    result = aggregate_chunk_scores(
        chunks,
        {"counter": [None]},
        {"counter": [None]},
        rewritten_texts=[chunks[0].text],
        token_counters=counters,
    )
    assert result["method"] == "single-section-native"
    assert result["total_original_weight"] == 2
    assert result["detectors"]["counter"]["original"] is None
    assert result["detectors"]["counter"]["rewritten"] is None
    assert result["detectors"]["counter"]["delta"] is None


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -0.1, 1.1])
def test_aggregation_rejects_invalid_detector_scores(score):
    counters = {"counter": words}
    chunks = split_document("One sentence.", counters)
    with pytest.raises(ValueError, match="finite score"):
        aggregate_chunk_scores(
            chunks,
            {"counter": [score]},
            {"counter": [0.2]},
            rewritten_texts=["A sentence."],
            token_counters=counters,
        )


def test_aggregation_rejects_missing_sections_and_mismatched_detector_names():
    counters = {"counter": words}
    chunks = split_document("One sentence.", counters)
    with pytest.raises(ValueError, match="one rewritten text"):
        aggregate_chunk_scores(
            chunks,
            {"counter": [0.2]},
            {"counter": [0.2]},
            rewritten_texts=[],
            token_counters=counters,
        )
    with pytest.raises(ValueError, match="must match"):
        aggregate_chunk_scores(
            chunks,
            {"other": [0.2]},
            {"counter": [0.2]},
            rewritten_texts=["A sentence."],
            token_counters=counters,
        )
