# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: William Droz <william.droz@idiap.ch>
# SPDX-License-Identifier: GPL-3.0-only

"""Bounded technical rewrites with evidence-based review and paired scores."""

from __future__ import annotations

import difflib
import hashlib
import json
import math
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .adaptive import (
    _call,
    _checkpoint,
    _json,
    _metadata,
    _score_value,
    digest,
    provider_error,
)
from .adaptive_transport import DeadlineClient
from .costs import summarize_run_costs
from .detectors import build_detectors, fakespot_clean_text
from .document_chunks import (
    MAX_TECHNICAL_CHARACTERS,
    build_token_counters,
    split_document,
)
from .document_windows import score_document_windows
from .formatted_adapters import project_dependencies
from .formatted_text import prepare_formatted_text
from .llm_client import validate_configuration
from .technical import (
    balanced_review_body,
    generation_body,
    mechanical_check,
    parse_candidate,
    parse_review,
    resolve_reviews,
)

PACKAGE_DIR = Path(__file__).resolve().parent
NAMES = ("fakespot_roberta", "desklib_deberta")
DEFAULT_GENERATORS = ("openai/gpt-6-luna", "deepseek/deepseek-v4.1-flash")
DEFAULT_REVIEWERS = (
    "deepseek/deepseek-v4.1-flash",
    "openai/gpt-6-luna",
    "google/gemini-3.8-flash",
    "claude-haiku-4.5",
)
_DEFAULT_MODEL_ALIASES = {
    "openai/gpt-6-luna": "gpt-6-luna",
    "google/gemini-3.8-flash": "gemini-3.8-flash",
    "claude-haiku-4.5": "anthropic/claude-haiku-4.5",
}


def resolve_default_models(names: tuple[str, ...], catalog: dict) -> list[str]:
    """Keep exact IDs when present, or use the catalog's known equivalent ID."""
    resolved = []
    for name in names:
        alias = _DEFAULT_MODEL_ALIASES.get(name, name)
        resolved.append(alias if name not in catalog and alias in catalog else name)
    return resolved


def load_catalog(output_dir: Path, *, timeout_seconds: float = 120) -> dict:
    """Resume recorded metadata, or discover models from the configured provider."""
    protocol = Path(output_dir) / "protocol.json"
    if protocol.exists():
        saved = json.loads(protocol.read_text(encoding="utf-8"))
        if saved.get("strategy") == "technical_update":
            return saved["catalog"]
        raise ValueError(
            "Output directory belongs to another strategy; choose a fresh directory"
        )
    return DeadlineClient(timeout_seconds=timeout_seconds).models()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_json(value) + "\n", encoding="utf-8")


def source_hashes() -> dict:
    paths = list(PACKAGE_DIR.glob("*.py"))
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(paths)
    }


def snapshot_sources(directory: Path, hashes: dict) -> None:
    for relative, expected in hashes.items():
        if Path(relative).name != relative or not relative.endswith(".py"):
            raise ValueError("Invalid source snapshot path")
        content = (PACKAGE_DIR / relative).read_bytes()
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError("Source changed while snapshotting")
        destination = directory / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and destination.read_bytes() != content:
            raise ValueError("Source snapshot differs; choose a fresh directory")
        destination.write_bytes(content)


def prepare_sections(text: str, counters: dict) -> list[dict]:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Provide nonempty text")
    counts = {name: count(text) for name, count in counters.items()}
    chunks = split_document(
        text, counters, target_tokens=512 if max(counts.values()) <= 512 else 360
    )
    sections = [
        {"id": f"section-{chunk.index}", "text": chunk.text} for chunk in chunks
    ]
    assert assemble(sections) == text
    return sections


def assemble(sections: list[dict]) -> str:
    return "".join(section["text"] for section in sections)


def preserve_separators(sections: list[dict], original: list[dict]) -> list[dict]:
    """Keep original boundary whitespace so separately returned parts join safely."""
    result = []
    for before, after in zip(original, sections, strict=True):
        leading = re.match(r"\s*", before["text"])
        trailing = re.search(r"\s*$", before["text"])
        assert leading is not None and trailing is not None
        result.append(
            {
                "id": after["id"],
                "text": leading.group() + after["text"].strip() + trailing.group(),
            }
        )
    return result


def generate_review_candidate(
    sections: list[dict],
    model: str,
    *,
    catalog: dict,
    reviewers: list[str],
    directory: Path,
    binding: dict,
    call,
    seed: int = 61021,
    document=None,
) -> dict:
    """Finish all review decisions without access to detector values."""
    if len(set(reviewers)) != len(reviewers):
        raise ValueError("Reviewer IDs must be distinct")
    independent = [reviewer for reviewer in reviewers if reviewer != model]
    if len(set(independent)) < 3:
        raise ValueError("Provide three distinct reviewers other than the generator")
    original = assemble(sections)
    body = generation_body(sections, model, catalog, seed)
    if document is not None and document.format != "plain":
        body["messages"][0]["content"] += (
            "\nThis is a formatted document. Preserve its paragraph and list structure, "
            "headings, indentation, emphasis markers, link destinations, code, and "
            "all HTML or placeholder tags exactly. Edit prose only. Do not add markup. "
            "Keep every opaque placeholder in its original order and section."
        )
    generation = _checkpoint(
        directory / "generation.json",
        {**binding, "request": body},
        lambda: _call(call, body),
    )
    assert isinstance(generation, dict)
    row = {
        "model": model,
        "generation": generation,
        "sections": None,
        "text": None,
        "reviews": [],
        "adjudication": None,
        "acceptable": False,
        "disposition": "generation_unavailable",
    }
    try:
        if generation.get("finish_reason") != "stop":
            return row
        parts = preserve_separators(
            parse_candidate(generation["raw_response"], sections), sections
        )
    except (ValueError, KeyError, TypeError) as exc:
        row["disposition"] = "generation_invalid_format"
        row["parse_error"] = str(exc)
        return row
    candidate = assemble(parts)
    row.update(sections=parts, text=candidate)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "attempted.txt").write_text(candidate, encoding="utf-8", newline="")
    formatting_issues = document.issues(candidate) if document is not None else []
    row["format_issues"] = formatting_issues
    if formatting_issues:
        row["disposition"] = "format_refusal"
        return row
    visible = document.visible_text if document is not None else lambda text: text
    if document is not None:
        row["formatted_text"] = document.restore(candidate)
    mechanics = mechanical_check(visible(original), visible(candidate))
    row["mechanical"] = mechanics
    if not mechanics["passed"]:
        row["disposition"] = "mechanical_refusal"
        return row

    def review(
        reviewer: str, prior=None, *, before=original, after=candidate, label=""
    ):
        before, after = visible(before), visible(after)
        request = balanced_review_body(
            before, after, reviewer, catalog, prior_reviews=prior
        )
        response = _checkpoint(
            directory
            / (
                label + "adjudication.json"
                if prior is not None
                else f"{label}review-{digest(reviewer)[:12]}.json"
            ),
            {**binding, "candidate_sha256": digest(candidate), "request": request},
            lambda: _call(call, request),
        )
        assert isinstance(response, dict)
        return {
            "model": reviewer,
            "response": response,
            **parse_review(response, before, after),
        }

    row["reviews"] = [review(reviewer) for reviewer in independent[:2]]
    resolution = resolve_reviews(row["reviews"])
    if resolution["needs_adjudication"]:
        prior = [
            {
                key: value
                for key, value in decision.items()
                if key not in {"response", "model"}
            }
            for decision in row["reviews"]
        ]
        row["adjudication"] = review(independent[2], prior)
        resolution = resolve_reviews(row["reviews"], row["adjudication"])
    row["resolution"] = resolution
    row["acceptable"] = resolution["accepted"]
    row["disposition"] = (
        "accepted"
        if row["acceptable"]
        else "material_refusal"
        if resolution["status"] == "reject"
        else "review_unavailable"
    )
    # Whole-document agreement cannot detect facts moved between scoring units.
    # Check the corresponding sections too; allow all harmless wording/layout edits.
    row["section_reviews"] = []
    if row["acceptable"] and len(sections) > 1:
        for index, (before, after) in enumerate(zip(sections, parts, strict=True)):
            if before["text"] == after["text"]:
                continue
            kwargs = {
                "before": before["text"],
                "after": after["text"],
                "label": f"section-{index}-",
            }
            initial = [review(reviewer, **kwargs) for reviewer in independent[:2]]
            decision = resolve_reviews(initial)
            adjudication = None
            if decision["needs_adjudication"]:
                prior = [
                    {
                        key: value
                        for key, value in item.items()
                        if key not in {"response", "model"}
                    }
                    for item in initial
                ]
                adjudication = review(independent[2], prior, **kwargs)
                decision = resolve_reviews(initial, adjudication)
            row["section_reviews"].append(
                {
                    "id": before["id"],
                    "reviews": initial,
                    "adjudication": adjudication,
                    "resolution": decision,
                }
            )
            if not decision["accepted"]:
                row["acceptable"] = False
                row["disposition"] = (
                    "section_fidelity_refusal"
                    if decision["status"] == "reject"
                    else "section_review_unavailable"
                )
                row["resolution"] = decision
                break
    return row


def score_sections(
    sections: list[dict], weights: list[int], detectors: dict, counters: dict
) -> dict:
    result = {}
    texts = [section["text"] for section in sections]
    for name in NAMES:
        counts = [counters[name](text) for text in texts]
        prepare = getattr(counters[name], "prepare", None)
        prepared = [
            prepare(text)
            if prepare is not None
            else fakespot_clean_text(text)
            if name == "fakespot_roberta"
            else text
            for text in texts
        ]
        if any(count > 512 for count in counts) or any(
            not value.strip() for value in prepared
        ):
            result[name] = {
                "available": False,
                "sections": None,
                "mean": None,
                "reason": "section_overlength_or_empty_after_preprocessing",
            }
            continue
        values = [_score_value(value) for value in detectors[name].score(texts)]
        if len(values) != len(texts):
            raise ValueError("Wrong detector score count")
        result[name] = {
            "available": True,
            "sections": values,
            "mean": sum(w * value for w, value in zip(weights, values, strict=True))
            / sum(weights),
            "token_counts": counts,
        }
    return result


def measure(
    sections: list[dict], weights: list[int], detectors: dict, counters: dict
) -> dict:
    return {
        "sections": score_sections(sections, weights, detectors, counters),
        "windows": score_document_windows(
            assemble(sections), detectors=detectors, counters=counters
        ),
    }


def relative_gain(before: dict, after: dict) -> float | None:
    """Require improvement in both detectors under both measurement schemes."""
    ratios = []
    for name in NAMES:
        pairs = [
            (before["sections"][name], after["sections"][name]),
            (before["windows"]["detectors"][name], after["windows"]["detectors"][name]),
        ]
        for original, candidate in pairs:
            if not original["available"] or not candidate["available"]:
                return None
            old, new = _score_value(original["mean"]), _score_value(candidate["mean"])
            if old <= 0 or old - new <= max(1e-6, old * 0.01):
                return None
            ratios.append(new / old)
    return max(ratios)


def choose_candidate(rows: list[dict], original_scores: dict) -> dict | None:
    eligible = []
    for row in rows:
        if row["acceptable"] and row.get("scores"):
            ratio = relative_gain(original_scores, row["scores"])
            if ratio is not None:
                eligible.append((ratio, row["model"], row))
    return min(eligible, key=lambda item: item[:2])[2] if eligible else None


def candidate_provider_errors(row: dict) -> list[dict]:
    """Expose unavailable generation/review evidence without raw provider errors."""
    errors = []
    generation = row["generation"]
    if (
        generation.get("finish_reason") != "stop"
        or row["disposition"] == "generation_invalid_format"
    ):
        errors.append(
            provider_error(generation, stage="generation", candidate_key=row["model"])
        )

    def review_errors(reviews, adjudication, *, section_index=None):
        for review in [*reviews, *([adjudication] if adjudication else [])]:
            if review.get("status") == "unavailable":
                errors.append(
                    provider_error(
                        review["response"],
                        stage="review",
                        candidate_key=row["model"],
                        section_index=section_index,
                    )
                )

    review_errors(row["reviews"], row["adjudication"])
    for index, section in enumerate(row.get("section_reviews", [])):
        review_errors(section["reviews"], section["adjudication"], section_index=index)
    return errors


def process_technical_text(
    text: str,
    *,
    output_dir: Path,
    models: list[str],
    reviewers: list[str],
    catalog: dict,
    workers: int = 3,
    request_timeout: float = 120,
    detectors=None,
    token_counters=None,
    call=None,
    progress=None,
    input_format: str = "auto",
) -> dict:
    """Generate a bounded portfolio, score complete candidates, or keep the input."""
    if (
        not isinstance(text, str)
        or not text.strip()
        or len(text) > MAX_TECHNICAL_CHARACTERS
    ):
        raise ValueError(
            f"Technical updates require 1–{MAX_TECHNICAL_CHARACTERS:,} nonblank characters"
        )
    document = prepare_formatted_text(text, format=input_format)
    source_text, text = text, document.text
    if type(workers) is not int or not 1 <= workers <= 4:
        raise ValueError("workers must be 1–4")
    if not math.isfinite(request_timeout) or request_timeout <= 0:
        raise ValueError("request_timeout must be positive and finite")
    if not models or len(models) > 8 or len(set(models)) != len(models):
        raise ValueError("Provide 1–8 distinct generator models")
    if len(set(reviewers)) != len(reviewers):
        raise ValueError("Reviewer IDs must be distinct")
    if any(model not in catalog for model in [*models, *reviewers]):
        raise ValueError("Missing model metadata")
    if any(len(set(reviewers) - {model}) < 3 for model in models):
        raise ValueError("Each generator needs three independent reviewers")
    if call is None:
        validate_configuration()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    counters = build_token_counters() if token_counters is None else token_counters
    detectors = build_detectors(batch_size=4) if detectors is None else detectors
    detectors, counters, _ = project_dependencies(document, detectors, counters)
    if set(counters) != set(NAMES) or set(detectors) != set(NAMES):
        raise ValueError("Both pinned detectors and counters are required")
    sections = prepare_sections(text, counters)
    weights = [len(section["text"].split()) for section in sections]
    protocol = {
        "strategy": "technical_update",
        "experimental": True,
        "promotion_gate_passed": False,
        "acceptance": "balanced_evidence_with_adjudication",
        "input_sha256": digest(source_text),
        "editable_input_sha256": digest(text),
        "formatting": document.metadata(),
        "sections": sections,
        "original_word_weights": weights,
        "models": models,
        "reviewers": reviewers,
        "catalog": catalog,
        "source_hashes": source_hashes(),
        "workers": workers,
        "request_timeout_seconds": request_timeout,
        "detectors": {name: _metadata(model) for name, model in detectors.items()},
        "tokenizers": {
            name: counter.metadata()
            if hasattr(counter, "metadata")
            else {"injected": True}
            for name, counter in counters.items()
        },
        "selection": "both fixed-section and whole-text window scores must improve on both detectors by >1%; minimize worst relative score",
        "human_review_performed": False,
        "pangram_submissions": 0,
        "files": {
            "input": str(output_dir / "input.txt"),
            "output": str(output_dir / "output.txt"),
            "result": str(output_dir / "result.json"),
            "diff": str(output_dir / "changes.diff"),
        },
    }
    binding = {"protocol_sha256": digest(_json(protocol))}
    _checkpoint(output_dir / "protocol.checkpoint.json", binding, lambda: protocol)
    write_json(output_dir / "protocol.json", protocol)
    snapshot_sources(output_dir / "source_snapshot", protocol["source_hashes"])
    (output_dir / "input.txt").write_text(source_text, encoding="utf-8", newline="")
    (output_dir / "editable_input.txt").write_text(text, encoding="utf-8", newline="")
    call = (
        DeadlineClient(timeout_seconds=request_timeout).call if call is None else call
    )

    def generate(model):
        row = generate_review_candidate(
            sections,
            model,
            catalog=catalog,
            reviewers=reviewers,
            directory=output_dir / "candidates" / model.replace("/", "--"),
            binding=binding,
            call=call,
            document=document,
        )
        if progress:
            progress(f"{model}: {row['disposition']}")
        return row

    if document.visible_text().strip():
        with ThreadPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(generate, models))
    else:
        rows = []
    original_scores = _checkpoint(
        output_dir / "scores" / "original.json",
        binding,
        lambda: measure(sections, weights, detectors, counters),
    )
    assert isinstance(original_scores, dict)
    for row in rows:
        if row["sections"] is not None and not row.get("format_issues"):
            row["scores"] = _checkpoint(
                output_dir / "scores" / (row["model"].replace("/", "--") + ".json"),
                {**binding, "candidate_sha256": digest(row["text"])},
                lambda row=row: measure(row["sections"], weights, detectors, counters),
            )
    winner = choose_candidate(rows, original_scores)
    provider_errors = [
        error for row in rows for error in candidate_provider_errors(row)
    ]
    failed = bool(provider_errors) and all(
        row["disposition"]
        in {
            "generation_unavailable",
            "generation_invalid_format",
            "review_unavailable",
            "section_review_unavailable",
        }
        for row in rows
    )
    output = document.restore(winner["text"] if winner else text)
    final_scores = winner["scores"] if winner else original_scores
    result = {
        "text": output,
        "cost_summary": summarize_run_costs(output_dir),
        "changed": output != source_text,
        "status": (
            "provider_error"
            if failed
            else "changed_with_provider_errors"
            if provider_errors and output != source_text
            else "unchanged_with_provider_errors"
            if provider_errors
            else "changed"
            if output != source_text
            else "unchanged_no_qualified_improvement"
        ),
        "provider_errors": provider_errors,
        "provider_status": "failed"
        if failed
        else "partial_failure"
        if provider_errors
        else "completed",
        "strategy": "technical_update",
        "experimental": True,
        "formatting": document.metadata(),
        "promotion_gate_passed": False,
        "validation_status": "Exploratory gains on polished synthetic updates; preset reliability gate failed. This mode is opt-in and retains original when review or score checks fail.",
        "selected_model": winner["model"] if winner else None,
        "candidate_count": len(rows),
        "candidates": rows,
        "scores": {
            "method": "fixed-original-word-weighted-sections",
            "detectors": {
                name: {
                    "original": original_scores["sections"][name]["mean"],
                    "rewritten": final_scores["sections"][name]["mean"],
                    "delta": final_scores["sections"][name]["mean"]
                    - original_scores["sections"][name]["mean"]
                    if final_scores["sections"][name]["available"]
                    and original_scores["sections"][name]["available"]
                    else None,
                }
                for name in NAMES
            },
        },
        "measurements": {"original": original_scores, "rewritten": final_scores},
        "final_review": winner["resolution"]
        if winner
        else {"accepted": True, "reason": "original preserved"},
        "files": protocol["files"],
        "human_review_performed": False,
        "pangram_submissions": 0,
    }
    (output_dir / "output.txt").write_text(output, encoding="utf-8", newline="")
    (output_dir / "changes.diff").write_text(
        "".join(
            difflib.unified_diff(
                source_text.splitlines(True),
                output.splitlines(True),
                fromfile="input.txt",
                tofile="output.txt",
            )
        ),
        encoding="utf-8",
    )
    write_json(output_dir / "result.json", result)
    return result
