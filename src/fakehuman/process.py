# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: William Droz <william.droz@idiap.ch>
# SPDX-License-Identifier: GPL-3.0-only

"""Process complete text with LLM edits and paired local detector scores."""

from __future__ import annotations

import difflib
import hashlib
import json
import math
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

from .adaptive import (
    CONTEXTUAL_REVIEW_VERSION,
    REVIEW_POLICIES,
    _call,
    _checkpoint,
    _metadata,
    _review,
    contextual_mechanical,
    digest,
    generation_body,
    nli_advisories,
    parse_candidate,
    provider_error,
    review_body,
    review_status,
    run_document,
)
from .adaptive_transport import DeadlineClient
from .bounded_evaluate import screen_candidate
from .costs import summarize_run_costs
from .detectors import build_detectors, fakespot_clean_text
from .document_chunks import (
    MAX_DOCUMENT_CHARACTERS,
    aggregate_chunk_scores,
    build_token_counters,
    split_document,
)
from .document_guards import enrich_request, format_issues
from .formatted_adapters import project_dependencies
from .formatted_text import prepare_formatted_text
from .llm_client import (
    LLMConfigurationError,
    resolve_model_names,
    validate_configuration,
)
from .meaning import MeaningGuard
from .rules import quality_issues

NAMES = ("fakespot_roberta", "desklib_deberta")


class LockedModel:
    def __init__(self, model, lock):
        self.model, self.lock = model, lock

    def metadata(self):
        return self.model.metadata()

    def score(self, texts):
        with self.lock:
            return self.model.score(texts)

    def check_pairs(self, pairs):
        with self.lock:
            return self.model.check_pairs(pairs)


class SectionGuard:
    """Conserve structure and refuse oversized edits before detector scoring."""

    def __init__(self, guard, counters, source, chunk):
        self.guard, self.counters, self.source, self.chunk = (
            guard,
            counters,
            source,
            chunk,
        )

    def metadata(self):
        return {
            **self.guard.metadata(),
            "document_structure_guard": 1,
            "candidate_token_limit": 512,
        }

    def check_pairs(self, pairs):
        results = []
        for before, after in pairs:
            issues = self.hard_issues(before, after)
            if issues:
                results.append(
                    {
                        "passed": False,
                        "reason": "document_structure_or_length_refusal",
                        "issues": issues,
                    }
                )
            else:
                results.extend(self.guard.check_pairs([(before, after)]))
        return results

    def hard_issues(self, before, after):
        """Structural and detector limits apply under every review policy."""
        issues = format_issues(before, after)
        if before == self.chunk.text:
            whole = (
                self.source[: self.chunk.start] + after + self.source[self.chunk.end :]
            )
            issues += format_issues(self.source, whole)
        if any(count(after) > 512 for count in self.counters.values()):
            issues.append("edited_section_exceeds_detector_limit")
        return list(dict.fromkeys(issues))


def _source_hashes():
    hashes = {
        name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in (
            "process.py",
            "document_chunks.py",
            "document_guards.py",
            "adaptive.py",
            "adaptive_transport.py",
            "llm_client.py",
            "detectors.py",
            "meaning.py",
            "rules.py",
            "bounded_evaluate.py",
            "bounded_edits.py",
            "formatted_text.py",
            "formatted_adapters.py",
            "technical.py",
            "inspection.py",
        )
    }
    return hashes


def _scorable(name, text, counters):
    prepare = getattr(counters[name], "prepare", None)
    prepared = (
        prepare(text)
        if prepare is not None
        else fakespot_clean_text(text)
        if name == "fakespot_roberta"
        else text
    )
    return bool(prepared.strip())


def _json_write(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _candidate_previews(document, source, chunks, section_results):
    """Retain usable drafts and findings independently of automatic selection."""
    previews = []
    for chunk, section in zip(chunks, section_results, strict=True):
        for candidate in section["candidates"]:
            attempted = candidate["attempted"]
            whole = source[: chunk.start] + attempted + source[chunk.end :]
            if whole == source or format_issues(source, whole):
                continue
            try:
                restored = document.restore(whole)
            except ValueError:
                continue
            previews.append(
                {
                    key: candidate[key]
                    for key in (
                        "candidate_key",
                        "review_policy",
                        "review_status",
                        "disposition",
                        "advisory_issues",
                        "mechanical",
                        "nli",
                        "selected",
                    )
                }
                | {
                    "section_index": chunk.index,
                    "candidate_text": restored,
                    "review": {
                        key: value
                        for key, value in (candidate["review"] or {}).items()
                        if key != "response"
                    }
                    or None,
                }
            )
    return previews


def process_text(
    text: str,
    *,
    output_dir: Path,
    rounds: int = 3,
    section_tokens: int = 360,
    workers: int = 2,
    request_timeout: float = 120,
    detectors=None,
    guard=None,
    call=None,
    token_counters=None,
    progress=None,
    strategy: str = "adaptive",
    models: list[str] | None = None,
    reviewers: list[str] | None = None,
    model_catalog: dict | None = None,
    input_format: str = "auto",
    editor_model: str | None = None,
    reviewer_model: str | None = None,
    review_policy: str = "strict",
) -> dict:
    """Return complete text, paired scores and explicit review status.

    Long inputs use fixed source sections and original-word-weighted scores.
    A failed final review rolls back the entire document. Checkpoints bind input,
    settings and source; use a fresh directory when these change. Dependencies
    can be injected for offline tests.
    """
    if strategy not in {"adaptive", "technical-update-experimental"}:
        raise ValueError("Unknown text processing strategy")
    if review_policy not in REVIEW_POLICIES:
        raise ValueError("review_policy must be strict or contextual")
    if strategy == "technical-update-experimental" and review_policy != "strict":
        raise ValueError("Contextual review policy applies to sentence editing only")
    if strategy == "adaptive" and (
        models is not None or reviewers is not None or model_catalog is not None
    ):
        raise ValueError("Model overrides require the experimental technical strategy")
    if strategy == "technical-update-experimental":
        from .technical_pipeline import (
            DEFAULT_GENERATORS,
            DEFAULT_REVIEWERS,
            load_catalog,
            process_technical_text,
            resolve_default_models,
        )

        if call is None:
            validate_configuration()
        catalog = (
            load_catalog(output_dir, timeout_seconds=request_timeout)
            if model_catalog is None
            else model_catalog
        )
        chosen_models = (
            resolve_default_models(DEFAULT_GENERATORS, catalog)
            if models is None
            else models
        )
        chosen_reviewers = (
            resolve_default_models(DEFAULT_REVIEWERS, catalog)
            if reviewers is None
            else reviewers
        )
        needed = set(chosen_models) | set(chosen_reviewers)
        missing = needed - set(catalog)
        if missing:
            raise LLMConfigurationError(
                "Missing provider model metadata: " + ", ".join(sorted(missing))
            )
        return process_technical_text(
            text,
            output_dir=output_dir,
            models=chosen_models,
            reviewers=chosen_reviewers,
            catalog={name: catalog[name] for name in sorted(needed)},
            workers=workers,
            request_timeout=request_timeout,
            detectors=detectors,
            token_counters=token_counters,
            call=call,
            progress=progress,
            input_format=input_format,
        )
    editor_model, reviewer_model = resolve_model_names(editor_model, reviewer_model)
    if review_policy == "contextual" and editor_model == reviewer_model:
        raise ValueError(
            "Contextual review requires different editor and reviewer models"
        )
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Provide nonempty text")
    if len(text) > MAX_DOCUMENT_CHARACTERS:
        raise ValueError(
            f"Provide up to {MAX_DOCUMENT_CHARACTERS:,} characters per request"
        )
    document = prepare_formatted_text(text, format=input_format)
    source_text, text = text, document.text
    if type(rounds) is not int or not 1 <= rounds <= 3:
        raise ValueError("rounds must be between 1 and 3")
    if (
        type(section_tokens) is not int
        or type(workers) is not int
        or not 64 <= section_tokens <= 400
        or not 1 <= workers <= 4
    ):
        raise ValueError("section_tokens must be 64–400 and workers 1–4")
    if not math.isfinite(request_timeout) or request_timeout <= 0:
        raise ValueError("request_timeout must be positive and finite")
    if call is None:
        validate_configuration()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    token_counters = (
        build_token_counters() if token_counters is None else token_counters
    )
    detectors = build_detectors(batch_size=4) if detectors is None else detectors
    guard = MeaningGuard(batch_size=4) if guard is None else guard
    detectors, token_counters, guard = project_dependencies(
        document, detectors, token_counters, guard
    )
    if set(token_counters) != set(NAMES):
        raise ValueError("Both pinned detector token counters are required")
    full_counts = {name: count(text) for name, count in token_counters.items()}
    chunk_target = 512 if max(full_counts.values()) <= 512 else section_tokens
    chunks = split_document(text, token_counters, target_tokens=chunk_target)
    assert "".join(chunk.text for chunk in chunks) == text
    if set(detectors) != set(NAMES):
        raise ValueError("Both pinned detectors are required")
    configuration = {
        "schema_version": 1,
        "input_sha256": digest(source_text),
        "editable_input_sha256": digest(text),
        "formatting": document.metadata(),
        "rounds": rounds,
        "editor_model": editor_model,
        "reviewer_model": reviewer_model,
        "review_policy": review_policy,
        "contextual_review_version": CONTEXTUAL_REVIEW_VERSION,
        "section_tokens": section_tokens,
        "request_timeout_seconds": request_timeout,
        "workers": workers,
        "input_tokens": full_counts,
        "sections": [asdict(chunk) for chunk in chunks],
        "source_hashes": _source_hashes(),
        "detectors": {name: _metadata(model) for name, model in detectors.items()},
        "meaning_guard": _metadata(guard),
        "tokenizers": {
            name: counter.metadata() if hasattr(counter, "metadata") else {}
            for name, counter in token_counters.items()
        },
        "selection": (
            "Fakespot only; two candidates per section per round; strict improvement and source-quoted review required; local NLI and literal checks are advisory"
            if review_policy == "contextual"
            else "Fakespot only; two candidates per section per round; strict improvement and local guards required"
        ),
        "final_review": (
            "whole original versus reassembled text: hard structure and mechanics, independent source-quoted context review; local NLI and literal checks are advisory"
            if review_policy == "contextual"
            else "whole original versus reassembled text: structure, mechanics, NLI and independent context review; oversized aligned NLI spans can be screened sentence by sentence"
        ),
        "scoring": "native score for one section; otherwise fixed-original-word-weighted section mean, not native whole-document probability",
        "data_provenance": "user-supplied text; authorship unknown; excluded from research corpora",
        "human_review_performed": False,
    }
    # Round-trip converts tuple fields from dataclasses to their JSON form.
    configuration = json.loads(json.dumps(configuration))
    protocol_path = output_dir / "protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text(encoding="utf-8")) != configuration:
            raise ValueError(
                "Input, configuration or code changed; choose a new output directory"
            )
    else:
        _json_write(protocol_path, configuration)
    input_path = output_dir / "input.txt"
    if (
        input_path.exists()
        and input_path.read_text(encoding="utf-8", newline="") != source_text
    ):
        raise ValueError("Saved input differs from current input")
    input_path.write_text(source_text, encoding="utf-8", newline="")
    (output_dir / "editable_input.txt").write_text(text, encoding="utf-8", newline="")
    binding = {"protocol_sha256": digest(protocol_path.read_text(encoding="utf-8"))}
    call = (
        DeadlineClient(timeout_seconds=request_timeout).call if call is None else call
    )
    lock = threading.RLock()
    local_guard = LockedModel(guard, lock)
    target = LockedModel(detectors["fakespot_roberta"], lock)

    def process_section(chunk):
        def contextual_call(body):
            submitted = enrich_request(
                body,
                source_document=text,
                chunk_span=(chunk.start, chunk.end),
                logical_lines=True,
            )
            content = json.loads(submitted["messages"][1]["content"])
            if review_policy == "contextual" and "sentences" not in content:
                # The grounded reviewer compares claims anywhere in the source,
                # including introductions outside a section's local NLI window.
                revised = text[: chunk.start] + content["after"] + text[chunk.end :]
                content["before"] = document.visible_text(content["before"])
                content["after"] = document.visible_text(content["after"])
                content["document_context"] = {
                    "scope": "full_document",
                    "before": document.visible_text(text),
                    "after": document.visible_text(revised),
                }
                submitted["messages"][1]["content"] = json.dumps(
                    content, ensure_ascii=False
                )
            if "sentences" in content:
                content["sentences"] = [
                    sentence
                    for sentence in content["sentences"]
                    if document.visible_text(sentence["text"]).strip()
                ]
                submitted["messages"][1]["content"] = json.dumps(
                    content, ensure_ascii=False
                )
            if "sentences" in content and not content["sentences"]:
                return {
                    "raw_response": "",
                    "finish_reason": "no_editable_sentences",
                    "usage": None,
                    "api_call_skipped": True,
                }
            response = dict(call(submitted))
            if "sentences" in content and response.get("finish_reason") == "stop":
                try:
                    proposal = parse_candidate(response["raw_response"])
                    if proposal["sentence_index"] not in {
                        s["sentence_index"] for s in content["sentences"]
                    }:
                        response.update(
                            provider_finish_reason="stop",
                            finish_reason="protected_span_refusal",
                        )
                except (ValueError, KeyError, TypeError):
                    pass  # The frozen parser records malformed proposals.
            return {
                **response,
                "submitted_request": submitted,
                "submitted_request_sha256": digest(
                    json.dumps(submitted, sort_keys=True)
                ),
            }

        if progress:
            progress(f"Processing section {chunk.index + 1}/{len(chunks)}")
        request = enrich_request(
            generation_body(
                chunk.text,
                chunk.text,
                0,
                1,
                1,
                logical_lines=True,
                editor_model=editor_model,
            ),
            source_document=text,
            chunk_span=(chunk.start, chunk.end),
            logical_lines=True,
        )
        editable = [
            sentence
            for sentence in json.loads(request["messages"][1]["content"])["sentences"]
            if document.visible_text(sentence["text"]).strip()
        ]
        if not editable or not _scorable(
            "fakespot_roberta", chunk.text, token_counters
        ):
            result = {
                "id": f"section-{chunk.index}",
                "arms": {"adaptive": {"text": chunk.text}},
                "candidates": [],
                "status": "no_editable_or_scorable_prose",
            }
        else:
            result = run_document(
                output_dir,
                {"id": f"section-{chunk.index}", "text": chunk.text},
                call=contextual_call,
                guard=SectionGuard(local_guard, token_counters, text, chunk),
                detector=target,
                rounds=rounds,
                logical_lines=True,
                editor_model=editor_model,
                reviewer_model=reviewer_model,
                review_policy=review_policy,
            )
        if progress:
            progress(f"Finished section {chunk.index + 1}/{len(chunks)}")
        return result

    with ThreadPoolExecutor(max_workers=workers) as pool:
        section_results = list(pool.map(process_section, chunks))
    attempted_sections = [row["arms"]["adaptive"]["text"] for row in section_results]
    attempted = "".join(attempted_sections)

    def final_screen():
        if attempted == text:
            return {
                "passed": True,
                "reason": "unchanged",
                "mechanical_issues": [],
                "nli": {"passed": True, "reason": "identity"},
                "review": None,
                "review_status": "accepted",
                "advisory_issues": [],
            }
        literal_issues = quality_issues(
            document.visible_text(text), document.visible_text(attempted)
        )
        advisory_issues = literal_issues if review_policy == "contextual" else []
        issues = format_issues(text, attempted) + document.issues(attempted)
        if review_policy == "contextual":
            issues += contextual_mechanical(text, attempted)["issues"]
        else:
            issues += literal_issues
        nli = (
            screen_candidate(local_guard, text, attempted, allow_overlength=True)
            if not issues
            else {"passed": False, "reason": "structure_refusal"}
        )
        if review_policy == "contextual":
            advisory_issues += nli_advisories(nli)
        body = review_body(
            document.visible_text(text),
            document.visible_text(attempted),
            reviewer_model=reviewer_model,
            review_policy=review_policy,
        )
        response = _checkpoint(
            output_dir / "final_context_response.json",
            {**binding, "attempted_sha256": digest(attempted), "request": body},
            lambda: _call(call, body),
        )
        assert isinstance(response, dict)
        review = _review(
            response,
            original=document.visible_text(text),
            attempted=document.visible_text(attempted),
            review_policy=review_policy,
        )
        passed = (
            not issues
            and (review_policy == "contextual" or bool(nli["passed"]))
            and bool(review["decision"]["passed"])
        )
        return {
            "passed": passed,
            "reason": "passed" if passed else "final_review_refused",
            "mechanical_issues": issues,
            "nli": nli,
            "review": review,
            "review_status": review_status(
                {"passed": not issues}, nli, review, review_policy
            ),
            "advisory_issues": advisory_issues,
        }

    final_guard = _checkpoint(
        output_dir / "final_review.json",
        {**binding, "attempted_sha256": digest(attempted)},
        final_screen,
    )
    assert isinstance(final_guard, dict)
    final_sections = (
        attempted_sections
        if final_guard["passed"]
        else [chunk.text for chunk in chunks]
    )
    output = "".join(final_sections)

    def scores():
        original_scores, final_scores, model_metadata = {}, {}, {}
        for name in NAMES:
            paired_texts = [chunk.text for chunk in chunks] + final_sections
            scorable = [
                i
                for i, value in enumerate(paired_texts)
                if _scorable(name, value, token_counters)
            ]
            measured = (
                detectors[name].score([paired_texts[i] for i in scorable])
                if scorable
                else []
            )
            if len(measured) != len(scorable):
                raise ValueError("Detector returned the wrong number of scores")
            values = [None] * len(paired_texts)
            for index, value in zip(scorable, measured, strict=True):
                values[index] = value
            original_scores[name], final_scores[name] = (
                values[: len(chunks)],
                values[len(chunks) :],
            )
            model_metadata[name] = detectors[name].metadata()
        summary = aggregate_chunk_scores(
            chunks,
            original_scores,
            final_scores,
            rewritten_texts=final_sections,
            token_counters=token_counters,
        )
        return {"summary": summary, "models": model_metadata}

    score_result = _checkpoint(
        output_dir / "score_results.json",
        {**binding, "output_sha256": digest(output)},
        scores,
    )
    assert isinstance(score_result, dict)
    output = document.restore(output)
    provider_errors = []
    generation_count, failed_generations = 0, 0
    review_count, failed_reviews = 0, 0
    for section_index, row in enumerate(section_results):
        for candidate in row["candidates"]:
            generation_count += 1
            generation = candidate["generation"]
            finish_reason = generation.get("finish_reason")
            if (
                finish_reason
                not in ("stop", "protected_span_refusal", "no_editable_sentences")
                or finish_reason == "stop"
                and candidate["proposal"] is None
            ):
                failed_generations += 1
                provider_errors.append(
                    provider_error(
                        generation,
                        stage="generation",
                        section_index=section_index,
                        candidate_key=candidate["candidate_key"],
                    )
                )
            review = candidate["review"]
            if review:
                review_count += 1
            if review and review.get("issues"):
                failed_reviews += 1
                provider_errors.append(
                    provider_error(
                        review["response"],
                        stage="review",
                        section_index=section_index,
                        candidate_key=candidate["candidate_key"],
                    )
                )
    final_review = final_guard.get("review")
    final_provider_failure = bool(final_review and final_review.get("issues"))
    if final_provider_failure:
        provider_errors.append(
            provider_error(final_review["response"], stage="final_review")
        )
    changed = output != source_text
    status = (
        "changed"
        if changed
        else ("reverted_final_review" if not final_guard["passed"] else "unchanged")
    )
    provider_status = "completed"
    if provider_errors:
        failed = (
            final_provider_failure
            or (generation_count > 0 and failed_generations == generation_count)
            or (review_count > 0 and failed_reviews == review_count)
        )
        provider_status = "failed" if failed else "partial_failure"
        status = (
            "changed_with_provider_errors"
            if changed
            else "provider_error"
            if failed
            else "unchanged_with_provider_errors"
        )
    result = {
        "text": output,
        "cost_summary": summarize_run_costs(output_dir),
        "changed": changed,
        "status": status,
        "provider_status": provider_status,
        "provider_errors": provider_errors,
        "scores": score_result["summary"],
        "models": score_result["models"],
        "formatting": document.metadata(),
        "input_tokens": full_counts,
        "section_count": len(chunks),
        "candidate_count": sum(len(row["candidates"]) for row in section_results),
        "selected_search_steps": sum(
            bool(c["selected"]) for row in section_results for c in row["candidates"]
        ),
        "kept_candidate_count": sum(
            bool(c["selected"]) for row in section_results for c in row["candidates"]
        )
        if final_guard["passed"]
        else 0,
        "final_review": final_guard,
        "review_policy": review_policy,
        "review_status": final_guard["review_status"],
        "candidates": _candidate_previews(document, text, chunks, section_results),
        "human_review_performed": False,
        "limitations": "Classifier scores are not calibrated authorship probabilities. Automated fidelity checks are fallible. Section means are not directly comparable to native whole-input scores.",
        "files": {
            name: str((output_dir / filename).resolve())
            for name, filename in {
                "input": "input.txt",
                "output": "output.txt",
                "result": "result.json",
                "diff": "changes.diff",
            }.items()
        },
    }
    if (
        not final_guard["passed"]
        and (not final_provider_failure or review_policy == "contextual")
        and document.visible_text(attempted).strip()
        and not format_issues(text, attempted)
    ):
        try:
            restored_attempted = document.restore(attempted)
        except ValueError:
            # The original stays usable when the candidate cannot be restored.
            pass
        else:
            if restored_attempted != source_text:
                result["attempted_text"] = restored_attempted
    (output_dir / "output.txt").write_text(output, encoding="utf-8", newline="")
    (output_dir / "attempted.txt").write_text(attempted, encoding="utf-8", newline="")
    (output_dir / "changes.diff").write_text(
        "".join(
            difflib.unified_diff(
                source_text.splitlines(keepends=True),
                output.splitlines(keepends=True),
                fromfile="input.txt",
                tofile="output.txt",
            )
        ),
        encoding="utf-8",
        newline="",
    )
    _json_write(output_dir / "result.json", result)
    return result
