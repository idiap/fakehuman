# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: William Droz <william.droz@idiap.ch>
# SPDX-License-Identifier: GPL-3.0-only

"""Shared text processing for the MCP server and optional local interface."""

from __future__ import annotations

import threading
from collections.abc import Collection
from math import inf
from statistics import fmean

from .bounded_evaluate import check_document_pair
from .detectors import DetectorInputTooLong, build_detectors, fakespot_clean_text
from .document_chunks import (
    MAX_DOCUMENT_CHARACTERS,
    MAX_TECHNICAL_CHARACTERS,
    TokenCounter,
)
from .document_windows import score_document_windows
from .inspection import refusal_reasons, trace_transformation
from .meaning import MeaningGuard
from .research_rules import RULE_LABELS, apply_rules, validate_manifest
from .rule_catalog import catalog_rule_names
from .rules import quality_issues


class TextService:
    """Reuse the pinned models while serializing calls from worker threads."""

    def __init__(self, manifest: dict, *, guard=None, detectors=None) -> None:
        validate_manifest(manifest)
        self.manifest = manifest
        self.guard = MeaningGuard() if guard is None else guard
        self.detectors = build_detectors() if detectors is None else detectors
        self._token_counters: dict[str, TokenCounter] = {}
        self._inference_lock = threading.Lock()

    @staticmethod
    def _validate_text(text: str, limit: int = MAX_DOCUMENT_CHARACTERS) -> None:
        if not text.strip() or len(text) > limit:
            raise ValueError(f"Provide nonempty text up to {limit:,} characters")

    def _checked(
        self, text: str, *, rules: list[str] | None = None, preview: bool = False
    ) -> dict:
        self._validate_text(text)
        if rules is None:
            rules = self.manifest["rules"]
        if preview:
            attempted, changes = trace_transformation(
                text, rules, self.manifest["seed"]
            )
        else:
            attempted = apply_rules(text, rules, self.manifest["seed"])
        issues = quality_issues(text, attempted)
        with self._inference_lock:
            meaning = check_document_pair(self.guard, text, attempted)
        refused = bool(issues or not meaning["passed"])
        result = {
            "text": text if refused else attempted,
            "refused": refused,
            "quality_issues": issues,
            "meaning": meaning,
        }
        if preview:
            result.update(
                attempted_text=attempted,
                changes=changes,
                refusal_reasons=refusal_reasons(issues, meaning),
            )
        return result

    def humanify(self, text: str) -> str:
        """Return accepted edits, or the original text when a check refuses them."""
        return self._checked(text)["text"]

    def inspect_transformation(
        self,
        text: str,
        *,
        enabled_rules: Collection[str] | None = None,
        include_catalog: bool = False,
    ) -> dict:
        """Inspect selected edits, optionally allowing rules outside the pipeline.

        Omitted selections always use the configured pipeline. Explicit catalog
        selections run configured rules first, then remaining registry rules.
        """
        rules = self.manifest["rules"]
        if enabled_rules is not None:
            if include_catalog:
                rules = catalog_rule_names(rules)
            selected = set(enabled_rules)
            unknown = selected.difference(rules)
            if unknown:
                raise ValueError(
                    (
                        "Rules outside the available catalog: "
                        if include_catalog
                        else "Rules outside the configured pipeline: "
                    )
                    + ", ".join(sorted(unknown))
                )
            rules = [name for name in rules if name in selected]
        return self._checked(text, rules=rules, preview=True)

    def rank_rules(self, text: str, *, include_catalog: bool = False) -> list[dict]:
        """Rank isolated edits from the pipeline or the explicitly enabled catalog."""
        self._validate_text(text)
        baseline = None
        rows = []
        rules = (
            catalog_rule_names(self.manifest["rules"])
            if include_catalog
            else tuple(dict.fromkeys(self.manifest["rules"]))
        )
        for name in rules:
            row = {
                "rule": name,
                "label": RULE_LABELS.get(name, name),
                "status": "unchanged",
                "rank": None,
                "mean_delta": None,
                "detectors": {},
                "refusal_reasons": [],
            }
            rows.append(row)
            checked = self._checked(text, rules=[name])
            if checked["refused"]:
                row["status"] = "refused"
                row["refusal_reasons"] = refusal_reasons(
                    checked["quality_issues"], checked["meaning"]
                )
                continue
            candidate = checked["text"]
            if candidate == text:
                continue
            if baseline is None:
                baseline = self.score_text(text, detailed=True)
            scored = self.score_text(candidate, detailed=True)
            for detector, before in baseline["scores"].items():
                before_window = (
                    baseline["detectors"].get(detector, {}).get("window_scoring")
                )
                after_window = (
                    scored["detectors"].get(detector, {}).get("window_scoring")
                )
                if detector in scored["scores"] and bool(before_window) == bool(
                    after_window
                ):
                    after = scored["scores"][detector]
                    row["detectors"][detector] = {
                        "input": before,
                        "output": after,
                        "delta": after - before,
                    }
            # Use the same detector set for every ranked rule.
            if (
                not row["detectors"]
                or row["detectors"].keys() != baseline["scores"].keys()
            ):
                row["status"] = "no_comparable_scores"
                continue
            row["status"] = "ranked"
            row["mean_delta"] = fmean(
                item["delta"] for item in row["detectors"].values()
            )

        rows.sort(
            key=lambda row: inf if row["mean_delta"] is None else row["mean_delta"]
        )
        previous_delta = None
        for position, row in enumerate(rows, start=1):
            delta = row["mean_delta"]
            if delta is None:
                break
            if delta != previous_delta:
                rank = position
                previous_delta = delta
            row["rank"] = rank
        return rows

    def rewrite_text(
        self,
        text: str,
        *,
        strategy: str = "adaptive",
        rounds: int = 3,
        input_format: str = "auto",
        review_policy: str = "strict",
    ) -> dict:
        """Rewrite with the configured remote LLM and keep local run checkpoints."""
        from .llm_client import validate_configuration
        from .process import process_text
        from .run_storage import new_run_directory

        if strategy not in {"adaptive", "technical-update-experimental"}:
            raise ValueError("Unknown text processing strategy")
        limit = (
            MAX_TECHNICAL_CHARACTERS
            if strategy == "technical-update-experimental"
            else MAX_DOCUMENT_CHARACTERS
        )
        self._validate_text(text, limit)
        if type(rounds) is not int or not 1 <= rounds <= 3:
            raise ValueError("rounds must be between 1 and 3")
        if input_format not in {"auto", "plain", "markdown", "html"}:
            raise ValueError("input_format must be auto, plain, markdown, or html")
        if review_policy not in {"strict", "contextual"}:
            raise ValueError("review_policy must be strict or contextual")
        if strategy != "adaptive" and review_policy != "strict":
            raise ValueError("Contextual review is available for sentence editing")
        validate_configuration()
        # The imported processors serialize their own workers. This outer lock
        # also excludes inference requested through the frozen tools or UI.
        with self._inference_lock:
            return process_text(
                text,
                output_dir=new_run_directory(),
                strategy=strategy,
                rounds=rounds,
                input_format=input_format,
                guard=self.guard,
                detectors=self.detectors,
                **(
                    {"review_policy": review_policy}
                    if review_policy != "strict"
                    else {}
                ),
            )

    def run_yolo(
        self,
        text: str,
        *,
        rounds: int = 3,
        input_format: str = "auto",
        review_policy: str = "strict",
    ) -> dict:
        """Run sentence editing, experimental rewriting, and the frozen pipeline."""
        self._validate_text(text, MAX_TECHNICAL_CHARACTERS)
        adaptive = self.rewrite_text(
            text,
            strategy="adaptive",
            rounds=rounds,
            input_format=input_format,
            **({"review_policy": review_policy} if review_policy != "strict" else {}),
        )
        adaptive_text = adaptive["text"]
        over_limit = len(adaptive_text) > MAX_TECHNICAL_CHARACTERS
        technical = self.rewrite_text(
            text if over_limit else adaptive_text,
            strategy="technical-update-experimental",
            input_format=input_format,
        )
        frozen = self.inspect_transformation(technical["text"])
        adaptive = {**adaptive, "source_text": text}
        technical = {
            **technical,
            "source_text": text if over_limit else adaptive_text,
        }
        frozen = {**frozen, "source_text": technical["text"]}
        result = {
            "text": frozen["text"],
            "stages": {
                "adaptive": adaptive,
                "technical-update-experimental": technical,
                "frozen": frozen,
            },
        }
        if over_limit:
            result["adaptive_output_over_limit"] = True
        return result

    def repair_candidate(
        self,
        original: str,
        candidate: str,
        findings: list[dict],
        *,
        input_format: str = "auto",
    ) -> dict:
        """Propose and review one repair while retaining the source as output."""
        from .repair import repair_candidate
        from .run_storage import new_run_directory

        self._validate_text(original, MAX_TECHNICAL_CHARACTERS)
        self._validate_text(candidate, MAX_TECHNICAL_CHARACTERS)
        with self._inference_lock:
            return repair_candidate(
                original,
                candidate,
                findings,
                input_format=input_format,
                output_dir=new_run_directory(),
            )

    def score_text(self, text: str, detailed: bool = False) -> dict:
        """Score complete text, using overlapping windows when needed."""
        if not text.strip():
            raise ValueError("Provide nonempty text")
        scores = {}
        details = {}
        # Adapters keep their latest input report on the instance. Read scores
        # and metadata under the same lock, also excluding guard inference.
        with self._inference_lock:
            for name, detector in self.detectors.items():
                windows = None
                try:
                    scores[name] = detector.score([text])[0]
                except DetectorInputTooLong:
                    input_report = dict(detector.last_input_report[0])
                    if name not in self._token_counters:
                        self._token_counters[name] = TokenCounter(detector.spec)
                    windows = score_document_windows(
                        text,
                        detectors={name: detector},
                        counters={name: self._token_counters[name]},
                        stride=max(1, min(256, detector.spec.max_tokens // 2)),
                    )["detectors"][name]
                    if not windows["available"]:
                        raise ValueError(f"{name}: {windows['reason']}") from None
                    scores[name] = windows["mean"]
                if detailed:
                    metadata = detector.metadata()
                    if windows is None:
                        input_report = dict(detector.last_input_report[0])
                    input_report["prepared_text"] = (
                        fakespot_clean_text(text)
                        if metadata["preprocessing"] == "fakespot_clean_text"
                        else text
                    )
                    details[name] = {"model": metadata, "input": input_report}
                    if windows is not None:
                        details[name]["window_scoring"] = windows
        if not detailed:
            return scores
        return {
            "scores": scores,
            "detectors": details,
            "interpretation": (
                "Scores are classifier outputs, not calibrated authorship probabilities. "
                "For long inputs, window means summarize overlapping portions of the text. "
                "Lower scores indicate less AI-like text to that detector; "
                "they do not establish human authorship or better writing."
            ),
        }
