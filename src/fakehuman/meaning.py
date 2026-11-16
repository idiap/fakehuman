# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: William Droz <william.droz@idiap.ch>
# SPDX-License-Identifier: GPL-3.0-only

"""Local bidirectional NLI screen for meaning preservation experiments.

This is a conservative screening signal, not a semantic proof. It performs no
text generation and never truncates input. The checkpoint and thresholds are
fixed so rule selection cannot tune them after seeing detector results.
"""

from __future__ import annotations

import difflib
import re
from collections.abc import Callable, Sequence
from typing import Any

MODEL_ID = "cross-encoder/nli-deberta-v3-small"
REVISION = "fa2804872c3b4bd748f38c0185cc85775361e735"
LICENSE = "Apache-2.0"
MAX_TOKENS = 512
ENTAILMENT_THRESHOLD = 0.8
CONTRADICTION_THRESHOLD = 0.1
_LABELS = {"contradiction": 0, "entailment": 1, "neutral": 2}


class MeaningGuard:
    """Check premise/hypothesis pairs with a pinned NLI model.

    Inputs are checked in both directions. For long pairs, only sentence-aligned
    changed spans are scored; unchanged aligned sentences need no rechecking.
    If an insertion/deletion or unbounded changed sentence prevents safe
    alignment, the result fails closed instead of dropping or truncating text.
    """

    def __init__(
        self,
        *,
        device: str = "cpu",
        batch_size: int = 8,
        tokenizer: Any | None = None,
        model: Any | None = None,
        torch_module: Any | None = None,
        loader: Callable[[], tuple[Any, Any, Any]] | None = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.device = device
        self.batch_size = batch_size
        self._tokenizer = tokenizer
        self._model = model
        self._torch = torch_module
        self._loader = loader
        self._cache: dict[tuple[str, str], dict] = {}

    def metadata(self) -> dict:
        """Return fixed model, threshold, and scope provenance."""
        return {
            "model_id": MODEL_ID,
            "revision": REVISION,
            "license": LICENSE,
            "source": f"https://huggingface.co/{MODEL_ID}/tree/{REVISION}",
            "config_source": f"https://huggingface.co/{MODEL_ID}/blob/{REVISION}/config.json",
            "labels": dict(_LABELS),
            "entailment_threshold": ENTAILMENT_THRESHOLD,
            "contradiction_threshold": CONTRADICTION_THRESHOLD,
            "max_pair_tokens": MAX_TOKENS,
            "device": self.device,
            "batch_size": self.batch_size,
            "loaded": self._model is not None,
            "trust_remote_code": False,
            "use_safetensors": True,
            "limitations": (
                "English NLI probabilities are screening scores, not proof of meaning preservation. "
                "Long inputs are checked on sentence-aligned changed spans; unsafe alignment, "
                "insertions/deletions, or spans above 512 pair tokens fail closed."
            ),
        }

    def _load(self) -> None:
        if self._model is not None:
            return
        if self._loader is not None:
            self._tokenizer, self._model, self._torch = self._loader()
            return
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Install fakehuman dependencies: uv sync --locked"
            ) from exc
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but is unavailable; use device='cpu'."
            )
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID, revision=REVISION, trust_remote_code=False
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_ID,
            revision=REVISION,
            trust_remote_code=False,
            use_safetensors=True,
        )
        label2id = {str(k).lower(): int(v) for k, v in model.config.label2id.items()}
        if any(label2id.get(label) != index for label, index in _LABELS.items()):
            raise RuntimeError(
                f"Unexpected NLI label mapping: {model.config.label2id!r}"
            )
        model.to(self.device)
        model.eval()
        self._tokenizer, self._model, self._torch = tokenizer, model, torch

    def _token_count(self, premise: str, hypothesis: str) -> int:
        self._load()
        assert self._tokenizer is not None
        # This preflight counts exact input before any model inference.
        encoded = self._tokenizer.encode(
            premise,
            hypothesis,
            add_special_tokens=True,
            truncation=False,
            verbose=False,
        )
        return len(encoded)

    @staticmethod
    def _sentence_chunks(text: str) -> list[str]:
        return [
            part.strip()
            for part in re.split(r"(?<=[.!?])\s+", text.strip())
            if part.strip()
        ]

    def _aligned_changed_chunks(
        self, before: str, after: str
    ) -> list[tuple[str, str]] | None:
        old_sentences = self._sentence_chunks(before)
        new_sentences = self._sentence_chunks(after)
        matcher = difflib.SequenceMatcher(
            a=old_sentences, b=new_sentences, autojunk=False
        )
        chunks: list[tuple[str, str]] = []
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                continue
            # Added/deleted sentences have no bidirectional counterpart.
            if tag in {"insert", "delete"}:
                return None
            left = " ".join(old_sentences[i1:i2])
            right = " ".join(new_sentences[j1:j2])
            if not left or not right:
                return None
            chunks.append((left, right))
        return chunks or None

    def _score_pairs(
        self, pairs: Sequence[tuple[str, str]]
    ) -> list[tuple[float, float]]:
        self._load()
        assert (
            self._tokenizer is not None
            and self._model is not None
            and self._torch is not None
        )
        scores: list[tuple[float, float]] = []
        for start in range(0, len(pairs), self.batch_size):
            batch = pairs[start : start + self.batch_size]
            premises = [pair[0] for pair in batch]
            hypotheses = [pair[1] for pair in batch]
            encoded = self._tokenizer(
                premises,
                hypotheses,
                padding=True,
                truncation=False,
                max_length=None,
                return_tensors="pt",
            ).to(self.device)
            with self._torch.inference_mode():
                logits = self._model(**encoded).logits
                probabilities = self._torch.softmax(logits, dim=-1).cpu().tolist()
            scores.extend(
                (
                    float(row[_LABELS["entailment"]]),
                    float(row[_LABELS["contradiction"]]),
                )
                for row in probabilities
            )
        return scores

    def _check_one(self, before: str, after: str) -> dict:
        if not before.strip() or not after.strip():
            return self._failed("empty_input", "unscorable")
        if before == after:
            return {
                "passed": True,
                "reason": "identical",
                "min_entailment": 1.0,
                "max_contradiction": 0.0,
                "scope": "identity",
            }

        pair_tokens = self._token_count(before, after)
        if pair_tokens <= MAX_TOKENS:
            chunks = [(before, after)]
            scope = "full_pair"
        else:
            aligned = self._aligned_changed_chunks(before, after)
            if aligned is None:
                return self._failed("alignment_failed", "unscorable")
            chunks = aligned
            if any(
                self._token_count(left, right) > MAX_TOKENS for left, right in chunks
            ):
                return self._failed("overlength", "changed_sentences")
            scope = "changed_sentences"

        # Each original/rewritten chunk is scored in both directions.
        directed = [
            direction
            for left, right in chunks
            for direction in ((left, right), (right, left))
        ]
        all_scores: list[tuple[float, float]] = []
        for start in range(0, len(directed), self.batch_size):
            batch = directed[start : start + self.batch_size]
            all_scores.extend(self._score_pairs(batch))
        min_entailment = min(entailment for entailment, _ in all_scores)
        max_contradiction = max(contradiction for _, contradiction in all_scores)
        passed = (
            min_entailment >= ENTAILMENT_THRESHOLD
            and max_contradiction <= CONTRADICTION_THRESHOLD
        )
        if passed:
            reason = "bidirectional_entailment"
        elif max_contradiction > CONTRADICTION_THRESHOLD:
            reason = "contradiction"
        else:
            reason = "low_entailment"
        return {
            "passed": passed,
            "reason": reason,
            "min_entailment": min_entailment,
            "max_contradiction": max_contradiction,
            "scope": scope,
        }

    @staticmethod
    def _failed(reason: str, scope: str) -> dict:
        return {
            "passed": False,
            "reason": reason,
            "min_entailment": 0.0,
            "max_contradiction": 1.0,
            "scope": scope,
        }

    def check_pairs(self, pairs: list[tuple[str, str]]) -> list[dict]:
        """Return one fail-closed bidirectional NLI decision per text pair."""
        if not isinstance(pairs, list):
            raise TypeError("pairs must be a list of (before, after) string tuples")
        for pair in pairs:
            if (
                not isinstance(pair, tuple)
                or len(pair) != 2
                or any(not isinstance(text, str) for text in pair)
            ):
                raise TypeError("each pair must be a (before, after) string tuple")
        results: list[dict] = []
        for before, after in pairs:
            key = (before, after) if before <= after else (after, before)
            if key not in self._cache:
                self._cache[key] = self._check_one(before, after)
            results.append(dict(self._cache[key]))
        return results
