# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: William Droz <william.droz@idiap.ch>
# SPDX-License-Identifier: GPL-3.0-only

"""Guard detector polarity and length handling without downloading checkpoints."""

from types import SimpleNamespace
from typing import Any, cast

import pytest

from fakehuman.detectors import (
    DETECTOR_SPECS,
    TransformerDetector,
    _desklib_model_class,
    build_detectors,
    fakespot_clean_text,
)


@pytest.mark.parametrize("method", ["score", "score_prepared"])
def test_invalid_input_and_empty_batch_do_not_load_weights(monkeypatch, method):
    detector = build_detectors(names=["openai_roberta"])["openai_roberta"]
    monkeypatch.setattr(detector, "_load", lambda: pytest.fail("Loaded unnecessarily"))
    score = getattr(detector, method)
    assert score([]) == []
    with pytest.raises(TypeError, match="list of strings"):
        score(cast(Any, "A single text"))
    with pytest.raises(ValueError, match="nonempty"):
        score([" "])


def test_preprocessing_matches_documented_prose_normalization():
    assert fakespot_clean_text(
        "# **Result**\nSee [paper](https://example.com) &amp; data."
    ) == ("Result See paper & data.")
    detector = build_detectors()["fakespot_roberta"]
    with pytest.raises(ValueError, match="erased"):
        detector.score(["```removed code```"])


def test_overlength_input_is_rejected_before_inference():
    detector = build_detectors(names=["openai_roberta"])["openai_roberta"]
    detector._model = object()
    detector._torch = cast(Any, object())
    detector._tokenizer = lambda *_args, **_kwargs: {"input_ids": [list(range(513))]}
    with pytest.raises(ValueError, match="exceed 512 tokens"):
        detector.score(["A long text"])
    assert detector.last_input_report[0]["tokens"] == 513
    assert detector.last_input_report[0]["truncated"] is False


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("openai_roberta", 0.9820138),
        ("fakespot_roberta", 0.0179862),
        ("desklib_deberta", 0.8807971),
    ],
)
def test_ai_score_uses_opposite_correct_columns_and_preserves_order(name, expected):
    torch = pytest.importorskip("torch")
    calls = []

    class Encoding(dict):
        def to(self, _device):
            return self

    class Tokenizer:
        def __call__(self, texts, **kwargs):
            if kwargs.get("return_tensors") == "pt":
                return Encoding(input_ids=torch.ones((len(texts), 4), dtype=torch.long))
            return {"input_ids": [[0, 1, 2, 3] for _ in texts]}

    class Model:
        def __call__(self, input_ids):
            calls.append(len(input_ids))
            logits = [2.0] if name == "desklib_deberta" else [2.0, -2.0]
            return SimpleNamespace(logits=torch.tensor([logits] * len(input_ids)))

    detector = TransformerDetector(DETECTOR_SPECS[name], batch_size=2)
    detector._model = Model()
    detector._torch = torch
    detector._tokenizer = Tokenizer()
    scores = detector.score(["First prose.", "Second prose.", "First prose."])
    assert scores == pytest.approx([expected] * 3, abs=1e-6)
    assert calls == [2]
    assert detector.score(["Second prose."]) == pytest.approx([expected], abs=1e-6)
    assert calls == [2]


def test_default_pair_is_modern_and_legacy_is_explicitly_available():
    assert list(build_detectors()) == ["fakespot_roberta", "desklib_deberta"]
    assert list(build_detectors(names=["openai_roberta", "fakespot_roberta"])) == [
        "openai_roberta",
        "fakespot_roberta",
    ]
    selected = build_detectors(names=["fakespot_roberta", "desklib_deberta"])
    assert list(selected) == ["fakespot_roberta", "desklib_deberta"]
    assert (
        selected["desklib_deberta"].metadata()["score_definition"].startswith("sigmoid")
    )
    assert selected["desklib_deberta"].metadata()["loaded"] is False
    with pytest.raises(ValueError, match="Unknown detector"):
        build_detectors(names=["unregistered"])
    with pytest.raises(ValueError, match="distinct"):
        build_detectors(names=["fakespot_roberta", "fakespot_roberta"])


def test_desklib_pooling_excludes_padding_without_remote_model_code(monkeypatch):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    class Encoder(torch.nn.Module):
        def forward(self, **kwargs):
            return SimpleNamespace(
                last_hidden_state=torch.tensor(
                    [
                        [[2.0, 4.0], [4.0, 6.0], [900.0, 900.0]],
                        [[1.0, 3.0], [800.0, 800.0], [900.0, 900.0]],
                    ]
                )
            )

    def local_encoder(config, **kwargs):
        assert kwargs["trust_remote_code"] is False
        return Encoder()

    monkeypatch.setattr(transformers.AutoModel, "from_config", local_encoder)
    config = transformers.DebertaV2Config(
        hidden_size=2, num_hidden_layers=1, num_attention_heads=1, intermediate_size=4
    )
    model = _desklib_model_class()(config)
    with torch.no_grad():
        model.classifier.weight.fill_(1.0)
        model.classifier.bias.zero_()
    result = model(
        input_ids=torch.ones((2, 3), dtype=torch.long),
        attention_mask=torch.tensor([[1, 1, 0], [1, 0, 0]]),
    )
    assert result.logits.tolist() == [[8.0], [4.0]]


def test_desklib_refuses_unloaded_or_mismatched_classifier_weights(monkeypatch):
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from fakehuman import detectors

    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: object()
    )
    monkeypatch.setattr(
        transformers.AutoConfig,
        "from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(
            model_type="deberta-v2", architectures=["DesklibAIDetectionModel"]
        ),
    )

    class IncompleteModel:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            assert kwargs["use_safetensors"] is True
            assert kwargs["revision"] == DETECTOR_SPECS["desklib_deberta"].revision
            return object(), {"missing_keys": ["classifier.weight"]}

    monkeypatch.setattr(detectors, "_desklib_model_class", lambda: IncompleteModel)
    detector = build_detectors(names=["desklib_deberta"])["desklib_deberta"]
    with pytest.raises(RuntimeError, match="did not fully match"):
        detector._load()


@pytest.mark.parametrize("overflow", ["error", "truncate"])
def test_prepared_overlength_never_truncates_even_with_native_opt_in(overflow):
    detector = TransformerDetector(
        DETECTOR_SPECS["fakespot_roberta"], overflow=overflow
    )
    detector._model = object()
    detector._torch = cast(Any, object())
    detector._tokenizer = lambda *_args, **_kwargs: {"input_ids": [list(range(513))]}
    with pytest.raises(ValueError, match="score_prepared never truncates"):
        detector.score_prepared(["> Retained text"])
    assert detector.last_input_report[0]["truncated"] is False
    assert detector.last_input_report[0]["input_is_prepared"] is True


def test_prepared_scoring_preserves_markdown_boundary_and_decoded_entities(monkeypatch):
    torch = pytest.importorskip("torch")
    from fakehuman import detectors

    texts = ["> Retain the complete quotation", "<retain>this literal text</retain>"]
    assert any(fakespot_clean_text(text) != text for text in texts)
    calls = []

    class Encoding(dict):
        def to(self, _device):
            return self

    class Tokenizer:
        def __call__(self, batch, **kwargs):
            calls.append((list(batch), kwargs))
            assert kwargs["truncation"] is False
            if kwargs.get("return_tensors") == "pt":
                return Encoding(input_ids=torch.ones((len(batch), 4), dtype=torch.long))
            return {"input_ids": [[0, 1, 2, 3] for _ in batch]}

    class Model:
        def __call__(self, input_ids):
            return SimpleNamespace(logits=torch.tensor([[2.0, -2.0]] * len(input_ids)))

    detector = TransformerDetector(
        DETECTOR_SPECS["fakespot_roberta"], overflow="truncate"
    )
    detector._model = Model()
    detector._torch = torch
    detector._tokenizer = Tokenizer()
    monkeypatch.setattr(
        detectors,
        "fakespot_clean_text",
        lambda _: pytest.fail("Already prepared inputs were cleaned again"),
    )
    assert detector.score_prepared(texts) == pytest.approx([0.0179862] * 2, abs=1e-6)
    assert [batch for batch, _kwargs in calls] == [texts, texts]
    assert all(item["input_is_prepared"] for item in detector.last_input_report)
    assert all(not item["preprocessing_changed"] for item in detector.last_input_report)
