# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: William Droz <william.droz@idiap.ch>
# SPDX-License-Identifier: GPL-3.0-only

"""Real, locally executed AI-text classifiers for controlled experiments.

Dependencies and model weights load on the first nonempty ``score`` call. Scores
are classifier outputs, not calibrated authorship probabilities. A lower
score is an experimental outcome, not evidence that a text was written by a human.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from html import unescape
from importlib.metadata import version
from typing import Literal


@dataclass(frozen=True)
class DetectorSpec:
    name: str
    model_id: str
    revision: str
    license: str
    ai_label: str
    ai_label_id: int
    max_tokens: int = 512
    preprocessing: str = "none"
    limitations: str = ""
    architecture: Literal["sequence_classifier", "desklib_mean_pool"] = (
        "sequence_classifier"
    )
    activation: Literal["softmax", "sigmoid"] = "softmax"


DETECTOR_SPECS = {
    "openai_roberta": DetectorSpec(
        name="openai_roberta",
        model_id="openai-community/roberta-base-openai-detector",
        revision="6cba99c003b711c7fe94f8a3aa2be35a792cb6fa",
        license="MIT",
        ai_label="Fake",
        ai_label_id=0,
        limitations="English; trained on GPT-2 outputs, not modern chat models.",
    ),
    "fakespot_roberta": DetectorSpec(
        name="fakespot_roberta",
        model_id="fakespot-ai/roberta-base-ai-text-detection-v1",
        revision="f9cdb14d1f8b105f597d80fa7b56f20c6ea0e9db",
        license="Apache-2.0",
        ai_label="AI",
        ai_label_id=1,
        preprocessing="fakespot_clean_text",
        limitations=(
            "English; supervised RoBERTa component only, not the complete Apollo "
            "ensemble. Shares the RoBERTa base architecture with the other detector."
        ),
    ),
    "desklib_deberta": DetectorSpec(
        name="desklib_deberta",
        model_id="desklib/ai-text-detector-v1.01",
        revision="5fdea974cd4287c61674951ec78803aa274e2fb7",
        license="MIT",
        ai_label="AI",
        ai_label_id=0,
        architecture="desklib_mean_pool",
        activation="sigmoid",
        limitations=(
            "English; 435M-parameter DeBERTa-v3-large, approximately 1.74 GB weights. "
            "A single output logit means AI according to the published model card. "
            "This adapter conservatively limits inputs to 512 tokens."
        ),
    ),
}

DEFAULT_DETECTOR_NAMES = ("fakespot_roberta", "desklib_deberta")


# SPDX-SnippetBegin
# SPDX-SnippetCopyrightText: Desklib <https://huggingface.co/desklib>
# SPDX-License-Identifier: MIT
# Adapted from Desklib's published model-card architecture. The upstream model
# card identifies Desklib as its developer and has no dated copyright notice.
def _desklib_model_class():
    """Implement the published MIT architecture locally, without remote code.

    Architecture and positive-logit meaning are documented at the pinned model:
    https://huggingface.co/desklib/ai-text-detector-v1.01/blob/
    5fdea974cd4287c61674951ec78803aa274e2fb7/README.md
    """
    import torch
    from transformers import AutoConfig, AutoModel, PreTrainedModel
    from transformers.modeling_outputs import SequenceClassifierOutput

    class DesklibAIDetectionModel(PreTrainedModel):
        config_class = AutoConfig

        def __init__(self, config):
            super().__init__(config)
            self.model = AutoModel.from_config(config, trust_remote_code=False)
            self.classifier = torch.nn.Linear(config.hidden_size, 1)
            self.post_init()

        def forward(self, input_ids, attention_mask=None, token_type_ids=None):
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            hidden = outputs.last_hidden_state
            mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            return SequenceClassifierOutput(logits=self.classifier(pooled))

    return DesklibAIDetectionModel


# SPDX-SnippetEnd


# SPDX-SnippetBegin
# SPDX-SnippetCopyrightText: Fakespot <https://huggingface.co/fakespot-ai>
# SPDX-License-Identifier: Apache-2.0
# Adapted from Fakespot's utils.py. The upstream file has no dated copyright
# notice; retain the original publisher attribution without inventing a year.
def fakespot_clean_text(text: str) -> str:
    """Apply the model card's Markdown/HTML and whitespace normalization.

    Local adaptation of Fakespot's Apache-2.0 ``utils.py`` at the pinned revision:
    https://huggingface.co/fakespot-ai/roberta-base-ai-text-detection-v1/
    blob/f9cdb14d1f8b105f597d80fa7b56f20c6ea0e9db/utils.py
    This intentionally removes code, blockquotes, images, tables and HTML tags.
    Use prose inputs and inspect the reported preprocessing.
    """
    substitutions = [
        (r"```.*?```", "", re.DOTALL),
        (r"`[^`]*`", "", 0),
        (r"!\[.*?\]\(.*?\)", "", 0),
        (r"\[([^\]]+)\]\(.*?\)", r"\1", 0),
        (r"(\*\*|__)(.*?)\1", r"\2", 0),
        (r"(\*|_)(.*?)\1", r"\2", 0),
        (r"#+ ", "", 0),
        (r"^>.*$", "", re.MULTILINE),
        (r"^(\s*[-*+]|\d+\.)\s+", "", re.MULTILINE),
        (r"^\s*[-*_]{3,}\s*$", "", re.MULTILINE),
        (r"\|.*?\|", "", 0),
        (r"<.*?>", "", 0),
    ]
    for pattern, replacement, flags in substitutions:
        text = re.sub(pattern, replacement, text, flags=flags)
    text = unescape(text)
    for whitespace in ("\n", "\t", "^M", "\r"):
        text = text.replace(whitespace, " ")
    return re.sub(" +", " ", text.replace(" ,", ","))


# SPDX-SnippetEnd


class DetectorInputTooLong(ValueError):
    """A classifier input exceeds its native token budget."""


class TransformerDetector:
    """Pinned binary classifier with explicit length handling and cached scores.

    ``score`` preserves input order. Empty/whitespace-only strings and inputs
    erased by preprocessing are rejected. The default overflow policy rejects
    texts exceeding 512 tokens (including special tokens). Opting into truncation
    records it per input in ``last_input_report``; do not compare truncated prose
    with complete prose without reporting that difference.
    """

    def __init__(
        self,
        spec: DetectorSpec,
        *,
        device: str = "cpu",
        batch_size: int = 8,
        overflow: Literal["error", "truncate"] = "error",
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if overflow not in {"error", "truncate"}:
            raise ValueError("overflow must be 'error' or 'truncate'")
        self.spec = spec
        self.name = spec.name
        self.device = device
        self.batch_size = batch_size
        self.overflow = overflow
        self.last_input_report: list[dict] = []
        self._tokenizer = None
        self._model = None
        self._torch = None
        self._cache: dict[str, float] = {}

    def metadata(self) -> dict:
        """Return JSON-serializable provenance without loading the model."""
        result = asdict(self.spec)
        result.update(
            {
                "source": f"https://huggingface.co/{self.spec.model_id}",
                "config_source": (
                    f"https://huggingface.co/{self.spec.model_id}/blob/"
                    f"{self.spec.revision}/config.json"
                ),
                "device": self.device,
                "batch_size": self.batch_size,
                "overflow": self.overflow,
                "loaded": self._model is not None,
                "score_definition": (
                    "sigmoid(single_AI_logit); lower is better"
                    if self.spec.activation == "sigmoid"
                    else "softmax(logits)[ai_label_id]; lower is better"
                ),
                "weights_format": "safetensors",
                "trust_remote_code": False,
            }
        )
        if self._model is not None:
            result["torch_version"] = version("torch")
            result["transformers_version"] = version("transformers")
        if self.spec.architecture == "desklib_mean_pool":
            result["label_source"] = (
                f"https://huggingface.co/{self.spec.model_id}/blob/"
                f"{self.spec.revision}/README.md"
            )
        return result

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import (
                AutoConfig,
                AutoModelForSequenceClassification,
                AutoTokenizer,
            )
        except ImportError as exc:
            raise RuntimeError(
                "Install fakehuman dependencies before running detectors: "
                "uv sync --locked"
            ) from exc
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but is unavailable; use device='cpu'."
            )
        tokenizer = AutoTokenizer.from_pretrained(
            self.spec.model_id,
            revision=self.spec.revision,
            trust_remote_code=False,
        )
        if self.spec.architecture == "desklib_mean_pool":
            config = AutoConfig.from_pretrained(
                self.spec.model_id, revision=self.spec.revision, trust_remote_code=False
            )
            if config.model_type != "deberta-v2" or config.architectures != [
                "DesklibAIDetectionModel"
            ]:
                raise RuntimeError(
                    "Desklib checkpoint architecture does not match the reviewed implementation"
                )
            model, loading = _desklib_model_class().from_pretrained(
                self.spec.model_id,
                revision=self.spec.revision,
                config=config,
                use_safetensors=True,
                output_loading_info=True,
            )
            if any(
                loading.get(key)
                for key in (
                    "missing_keys",
                    "unexpected_keys",
                    "mismatched_keys",
                    "error_msgs",
                )
            ):
                raise RuntimeError(
                    f"Desklib weights did not fully match the local architecture: {loading}"
                )
        else:
            model = AutoModelForSequenceClassification.from_pretrained(
                self.spec.model_id,
                revision=self.spec.revision,
                use_safetensors=True,
                trust_remote_code=False,
            )
            actual_label = model.config.id2label.get(self.spec.ai_label_id)
            if model.config.num_labels != 2 or actual_label != self.spec.ai_label:
                raise RuntimeError(
                    f"Unexpected label mapping for {self.spec.model_id}: "
                    f"{model.config.id2label!r}; expected "
                    f"{self.spec.ai_label_id}={self.spec.ai_label!r}."
                )
        model.to(self.device)
        model.eval()
        self._torch = torch
        self._tokenizer = tokenizer
        self._model = model

    def score(self, texts: list[str]) -> list[float]:
        """Return actual model scores in [0, 1]; never substitute mock results."""
        self._validate_inputs(texts)
        prepared = [
            fakespot_clean_text(text)
            if self.spec.preprocessing == "fakespot_clean_text"
            else text
            for text in texts
        ]
        if any(not text.strip() for text in prepared):
            raise ValueError("Detector preprocessing erased an input; provide prose")
        return self._score(
            texts, prepared, overflow=self.overflow, input_is_prepared=False
        )

    def score_prepared(self, texts: list[str]) -> list[float]:
        """Score exact, already prepared text without cleaning or truncation.

        Use only after applying this detector's preprocessing to the complete
        document. In particular, prepared window boundaries must not trigger a
        second Markdown cleanup. Overflow always raises, even when native
        ``score`` was configured with ``overflow='truncate'``.
        """
        self._validate_inputs(texts)
        return self._score(texts, texts, overflow="error", input_is_prepared=True)

    def _score_prepared_windows(self, texts: list[str]) -> list[float]:
        """Allow whitespace spans from an otherwise scorable prepared document."""
        self._validate_inputs(texts, allow_whitespace=True)
        return self._score(texts, texts, overflow="error", input_is_prepared=True)

    def _validate_inputs(
        self, texts: list[str], *, allow_whitespace: bool = False
    ) -> None:
        if isinstance(texts, str):
            raise TypeError("score expects a list of strings, not one string")
        self.last_input_report = []
        if any(
            not isinstance(text, str)
            or not (text if allow_whitespace else text.strip())
            for text in texts
        ):
            raise ValueError("Every input must be a nonempty string")

    def _score(
        self,
        texts: list[str],
        prepared: list[str],
        *,
        overflow: Literal["error", "truncate"],
        input_is_prepared: bool,
    ) -> list[float]:
        if not texts:
            return []
        self._load()
        assert self._tokenizer is not None
        assert self._model is not None
        assert self._torch is not None
        lengths = [
            len(ids)
            for ids in self._tokenizer(
                prepared, add_special_tokens=True, truncation=False, verbose=False
            )["input_ids"]
        ]
        self.last_input_report = [
            {
                "input_index": i,
                "input_characters": len(original),
                "prepared_characters": len(text),
                "preprocessing_changed": original != text,
                "input_is_prepared": input_is_prepared,
                "tokens": length,
                "max_tokens": self.spec.max_tokens,
                "truncated": length > self.spec.max_tokens and overflow == "truncate",
            }
            for i, (original, text, length) in enumerate(
                zip(texts, prepared, lengths, strict=True)
            )
        ]
        too_long = [
            i for i, length in enumerate(lengths) if length > self.spec.max_tokens
        ]
        if too_long and overflow == "error":
            guidance = (
                "Shorten the prepared inputs; score_prepared never truncates."
                if input_is_prepared
                else "Shorten the inputs or explicitly use overflow='truncate'."
            )
            raise DetectorInputTooLong(
                f"{self.name}: input indices {too_long} exceed {self.spec.max_tokens} "
                f"tokens. {guidance}"
            )
        pending = list(
            dict.fromkeys(text for text in prepared if text not in self._cache)
        )
        for start in range(0, len(pending), self.batch_size):
            batch = pending[start : start + self.batch_size]
            encoded = self._tokenizer(
                batch,
                padding=True,
                truncation=overflow == "truncate",
                max_length=self.spec.max_tokens if overflow == "truncate" else None,
                return_tensors="pt",
            ).to(self.device)
            with self._torch.inference_mode():
                logits = self._model(**encoded).logits
                if self.spec.activation == "sigmoid":
                    if logits.ndim != 2 or logits.shape[1] != 1:
                        raise RuntimeError(
                            "Desklib must return exactly one AI logit per input"
                        )
                    probabilities = self._torch.sigmoid(logits[:, 0])
                else:
                    probabilities = self._torch.softmax(logits, dim=-1)[
                        :, self.spec.ai_label_id
                    ]
                scores = probabilities.cpu().tolist()
            for text, score in zip(batch, scores, strict=True):
                if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                    raise RuntimeError(
                        f"{self.name} produced an invalid score: {score}"
                    )
                self._cache[text] = float(score)
        return [self._cache[text] for text in prepared]


def build_detectors(
    *,
    names: list[str] | tuple[str, ...] | None = None,
    device: str = "cpu",
    batch_size: int = 8,
    overflow: Literal["error", "truncate"] = "error",
) -> dict[str, TransformerDetector]:
    """Construct selected adapters; default uses Fakespot and Desklib.

    For the legacy comparison, choose ``names=("openai_roberta", "fakespot_roberta")``.
    Model choice must be fixed before running selection or inspecting test scores.
    Downloads and inference happen lazily in ``score``.
    """
    names = list(DEFAULT_DETECTOR_NAMES if names is None else names)
    if not names or len(names) != len(set(names)):
        raise ValueError("Choose distinct detector names")
    unknown = set(names).difference(DETECTOR_SPECS)
    if unknown:
        raise ValueError("Unknown detector(s): " + ", ".join(sorted(unknown)))
    return {
        name: TransformerDetector(
            DETECTOR_SPECS[name],
            device=device,
            batch_size=batch_size,
            overflow=overflow,
        )
        for name in names
    }
