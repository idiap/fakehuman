# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: William Droz <william.droz@idiap.ch>
# SPDX-License-Identifier: GPL-3.0-only

"""Optional local NiceGUI workbench; importing this module needs no UI packages."""

from __future__ import annotations

import json
import logging
from collections import Counter
from html import escape

from .costs import combine_cost_summaries
from .document_chunks import MAX_DOCUMENT_CHARACTERS, MAX_TECHNICAL_CHARACTERS
from .formatted_text import prepare_formatted_text
from .inspection import text_edits
from .llm_client import LLMConfigurationError, LLMModelCatalogError
from .rule_catalog import catalog_rule_names
from .rule_examples import rule_examples
from .rule_impacts import (
    impact_color,
    impact_label,
    recorded_impact_label,
    recorded_rule_impacts,
)
from .ui_diff import INLINE_DIFF_CSS, inline_diff_html
from .ui_styles import WORKBENCH_CSS

_EXAMPLE = (
    "Therefore, the team prepared a brief report due to the delay. "
    "However, they did not change the 12 mm component."
)
_LOCAL_SUMMARY = (
    "Applies the selected editing rules in a fixed order, then checks meaning. "
    "All editing stays on this computer."
)
_YOLO_SUMMARY = (
    "Runs LLM sentence editing, experimental technical rewriting, and the "
    "configured frozen rules in that order. Each stage uses the last accepted text."
)
_LOCAL_ANALYSIS_CAPTION = (
    "Choose local rules, compare detector scores, or rank rules for your draft."
)
_WORKFLOW_OPTIONS = {
    "frozen": "Local editing rules",
    "adaptive": "LLM sentence editing",
    "technical-update-experimental": "LLM technical rewrite (experimental)",
    "yolo": "YOLO · Run all three",
}
_LLM_GUIDES = {
    "adaptive": {
        "title": "How sentence editing works",
        "summary": (
            "Refines wording one sentence at a time, keeping the surrounding text. "
            "Use this for small, targeted edits."
        ),
        "steps": (
            (
                "Propose sentence edits",
                "Each round compares two alternatives per section. More rounds "
                "allow further edits and make more provider requests.",
            ),
            (
                "Check meaning and scores",
                "The selected preservation policy checks meaning with an independent LLM review. "
                "Only edits that lower the Fakespot detector score are kept; "
                "the other detector is reported for comparison.",
            ),
            (
                "Review the combined result",
                "A final preservation review can restore the original. If no edit "
                "passes the checks and improves the score, the text stays unchanged.",
            ),
        ),
        "before": (
            "The team carried out a review of the results. "
            "The release remains scheduled for Friday."
        ),
        "after": (
            "The team reviewed the results. The release remains scheduled for Friday."
        ),
        "caption": "Tightens one sentence while keeping the release date and surrounding text.",
    },
    "technical-update-experimental": {
        "title": "How technical rewriting works",
        "summary": (
            "Rewrites the document as a plain, factual update for colleagues. "
            "It can combine or reorder sentences and simplify promotional wording."
        ),
        "steps": (
            (
                "Draft complete rewrites",
                "Models propose alternatives for the whole document, allowing "
                "broader changes to wording and sentence structure.",
            ),
            (
                "Review facts and completeness",
                "Independent LLM reviewers check meaning, missing details and "
                "readability. An additional reviewer resolves disagreements.",
            ),
            (
                "Keep a qualifying improvement",
                "A rewrite must pass review and improve both detector scores "
                "by more than 1% relative to the original, using both section and "
                "whole-text window measurements. Otherwise, the original is returned.",
            ),
        ),
        "before": (
            "The new release introduces enhanced reporting capabilities. "
            "Workspace administrators now have the ability to export reports as CSV files. "
            "Only the currently filtered rows are included in each export."
        ),
        "after": (
            "The new release lets workspace administrators export reports as CSV files. "
            "Each export includes only the currently filtered rows."
        ),
        "caption": (
            "Combines related sentences while preserving who can export reports "
            "and which rows are included."
        ),
    },
}
_REDUCTION_BANDS = ("Small reduction", "Moderate reduction", "Large reduction")
_IMPACT_FILTERS = {
    "All impacts": None,
    "At least a small reduction": _REDUCTION_BANDS,
    "At least a moderate reduction": _REDUCTION_BANDS[1:],
    **{
        label: (label,)
        for label in (
            "Score increase",
            "Negligible change",
            *_REDUCTION_BANDS,
            "Not yet measured",
            "No AI matches",
            "All edits refused",
            "No edit on this input",
            "Preservation check refused",
            "Comparable scores unavailable",
        )
    },
}
_DETECTOR_LABELS = {"fakespot_roberta": "Fakespot", "desklib_deberta": "Desklib"}
_LOG = logging.getLogger(__name__)


def _impact_figure(title: str, deltas: dict[str, float]) -> str:
    """Plot signed score changes around zero and show their numeric values."""
    if not deltas:
        return ""
    limit = max(0.001, max(map(abs, deltas.values())))
    rows = []
    for detector, delta in deltas.items():
        name = escape(_DETECTOR_LABELS.get(detector, detector))
        origin = "right" if delta < 0 else "left"
        color = "bg-teal-700" if delta < 0 else "bg-red-700"
        width = 50 * abs(delta) / limit
        rows.append(
            '<div class="flex items-baseline justify-between gap-2 text-sm">'
            f'<span class="min-w-0 break-words">{name}</span>'
            f'<span class="shrink-0 tabular-nums">{delta * 100:+.4f} pp</span></div>'
            '<div class="relative mb-2 h-2 rounded bg-slate-100" aria-hidden="true">'
            f'<span class="absolute inset-y-0 {color}" '
            f'style="{origin}:50%;width:{width:.6f}%"></span>'
            '<span class="absolute inset-y-0 left-1/2 w-px bg-slate-500"></span></div>'
        )
    return (
        '<figure class="m-0 min-w-0 rounded border border-slate-200 px-3 py-2">'
        f'<figcaption class="mb-2 text-sm font-semibold">{escape(title)}</figcaption>'
        + "".join(rows)
        + '<div class="grid grid-cols-3 text-xs tabular-nums text-slate-600" aria-hidden="true">'
        f'<span>−{limit * 100:.4f}</span><span class="text-center">0</span>'
        f'<span class="text-right">+{limit * 100:.4f} pp</span>'
        "</div></figure>"
    )


def _recorded_impact_figures(record: dict) -> str:
    figures = [
        _impact_figure(title, sample["detector_deltas"])
        for title, sample in (
            ("AI benchmark · mean change", record),
            ("Human controls · mean change", record["human"]),
        )
        if sample["accepted"]
    ]
    probe = record["matching_example"]
    if probe["outcome"] == "preserved":
        figures.append(
            _impact_figure("Matching example · accepted", probe["detector_deltas"])
        )
    else:
        figures.append(
            _impact_figure(
                "Matching example · refused candidate", probe["raw_detector_deltas"]
            )
        )
    return "".join(figures)


def _rewrite_inspection(report: dict, original: str) -> dict:
    """Adapt document workflow results for the shared checks display."""
    output = report["text"]
    refused = report["status"] in {
        "reverted_final_review",
        "unchanged_no_qualified_improvement",
        "provider_error",
    }
    reasons = []
    if report["status"] == "reverted_final_review":
        reasons.append("The final preservation review refused the combined edits.")
        review = report.get("final_review", {}).get("review")
        if review and review.get("decision", {}).get("reason"):
            reasons.append(review["decision"]["reason"])
    elif report["status"] == "provider_error":
        reasons.append(
            "The LLM provider request failed. Check the provider configuration and "
            "model access, then retry."
        )
    elif refused:
        reasons.append("No proposed rewrite passed the review and improvement checks.")
    edits = text_edits(original, output) if original != output else []
    return {
        "text": output,
        "attempted_text": report.get("attempted_text", output),
        "refused": refused,
        "refusal_reasons": reasons,
        "provider_errors": report.get("provider_errors", []),
        "changes": [{"label": "LLM rewrite", "edits": edits}] if edits else [],
    }


def _review_details(record: dict, source: str, candidate: str) -> dict:
    """Collect available review evidence without treating model claims as facts."""
    review = record.get("review") or {}
    evidence = review.get("evidence", review)
    reviews = [evidence, *record.get("reviews", [])]
    if record.get("adjudication"):
        reviews.append(record["adjudication"])
    for section in record.get("section_reviews", []):
        reviews.extend(section.get("reviews", []))
        if section.get("adjudication"):
            reviews.append(section["adjudication"])
    findings, seen = [], set()
    source_words, candidate_words = (
        " ".join(prepare_formatted_text(source).visible_text().split()),
        " ".join(prepare_formatted_text(candidate).visible_text().split()),
    )
    for item in reviews:
        # Failed evidence validation may leave partial issues in the parser output.
        if item.get("status") == "unavailable":
            continue
        for finding in item.get("issues", []):
            if not isinstance(finding, dict):
                continue
            fields = (
                "severity",
                "category",
                "source_quote",
                "candidate_quote",
                "explanation",
            )
            if any(not isinstance(finding.get(key), str) for key in fields):
                continue
            quoted_source = " ".join(finding["source_quote"].split())
            quoted_candidate = " ".join(finding["candidate_quote"].split())
            if not quoted_source or quoted_source not in source_words:
                continue
            if (quoted_candidate and quoted_candidate not in candidate_words) or (
                not quoted_candidate and finding["category"].lower() != "omission"
            ):
                continue
            key = tuple(finding[field] for field in fields)
            if key not in seen:
                seen.add(key)
                findings.append(finding)
    status = record.get("review_status")
    if status not in {"accepted", "rejected", "unverified"}:
        resolution = record.get("resolution", {})
        verdict = resolution.get("status", evidence.get("status"))
        if verdict in {"accept", "minor"} or record.get("acceptable"):
            status = "accepted"
        elif verdict == "reject":
            status = "rejected"
        elif review.get("decision", {}).get("passed"):
            status = "accepted"
        else:
            status = "unverified"
    return {
        "review_status": status,
        "findings": findings,
        "advisory_issues": record.get("advisory_issues", []),
        "blocking_issues": [
            *record.get("mechanical_issues", []),
            *record.get("mechanical", {}).get("issues", []),
            *record.get("format_issues", []),
        ],
        "source": source,
    }


def _result_previews(original: str, selected: str, stages: dict) -> tuple[str, list]:
    """Build draft comparisons and review evidence from complete stage results."""
    seen = {original, selected}
    alternatives = []

    def add(title, text, detail, diff_label, source, record):
        if not text or not text.strip() or text in seen:
            return
        seen.add(text)
        alternatives.append(
            {
                "title": title,
                "text": text,
                "detail": detail,
                "diff_label": diff_label,
                "markup": inline_diff_html(original, text),
                **_review_details(record, source, text),
            }
        )

    source = original
    for method, stage in stages.items():
        source = stage.get("source_text", source)
        prefix = f"{_WORKFLOW_OPTIONS[method]} · " if len(stages) > 1 else ""
        final = stage.get("final_review", {})
        review = final.get("review") or {}
        reason = review.get("decision", {}).get("reason", "")
        add(
            f"{prefix}Attempted candidate",
            stage.get("attempted_text"),
            reason,
            "Refused candidate diff",
            source,
            final,
        )
        selected_model = stage.get("selected_model")
        for row in stage.get("candidates", []):
            if row.get("candidate_text"):
                add(
                    f"{prefix}Sentence candidate {row.get('candidate_key', '')}".strip(),
                    row["candidate_text"],
                    (row.get("review") or {}).get("decision", {}).get("reason", ""),
                    "Candidate diff",
                    source,
                    row,
                )
                continue
            if (
                row.get("model") == selected_model
                or row.get("format_issues")
                or not row.get("mechanical", {}).get("passed")
            ):
                continue
            if row["acceptable"]:
                detail = (
                    "Passed preservation review. The automatic result used another draft."
                    if selected_model
                    else "Passed preservation review. It did not qualify on detector scores."
                )
            elif row.get("resolution", {}).get("status") == "unavailable":
                detail = "Preservation review was unavailable for this draft."
            else:
                detail = "The preservation review flagged changes in this draft."
            add(
                f"{prefix}{row['model']}",
                row.get("formatted_text"),
                detail,
                "Candidate diff",
                source,
                row,
            )
        source = stage.get("text", source)
    return inline_diff_html(original, selected), alternatives


def _build_page(service) -> None:
    """Build independent, unsaved state for each browser tab."""
    from nicegui import run, ui

    state = {
        "inspection": None,
        "input_scores": None,
        "output_scores": None,
        "rewrite": None,
        "yolo": None,
        "rule_ranking": None,
        "repairs": [],
        "cost_summary": None,
    }
    configured_rules = tuple(dict.fromkeys(service.manifest["rules"]))
    available_rules = catalog_rule_names(configured_rules)
    recorded_impacts = recorded_rule_impacts()
    grouped_examples = {}
    for example in rule_examples(available_rules, service.manifest["seed"]):
        grouped_examples.setdefault(example["category"], []).append(example)
    rule_headings = {}
    rule_rows = {}
    rule_switches = {}
    rank_badges = {}
    impact_badges = {}
    impact_labels = {}
    impact_figures = {}
    impact_details = {}
    impact_panels = {}
    workflow_guides = {}
    controls = []
    review_controls = []
    busy = False
    draft_replacement: tuple[str, str] | None = None
    inspection_previews = ("", [])

    def report_text() -> str:
        return json.dumps(state, indent=2, ensure_ascii=False)

    async def copy(text: str) -> None:
        if not text:
            ui.notify("There is no text to copy yet.", type="info")
            return
        try:
            copied = await ui.run_javascript(
                "try { await navigator.clipboard.writeText("
                + json.dumps(text)
                + "); return true; } catch { return false; }",
                timeout=10,
            )
        except TimeoutError:
            copied = False
        ui.notify(
            "Copied to clipboard."
            if copied
            else "Clipboard unavailable. Select the text and press Ctrl/Cmd+C.",
            type="positive" if copied else "warning",
        )

    def copy_button(label, get_text):
        async def clicked():
            await copy(get_text())

        button = ui.button(label, icon="content_copy", on_click=clicked).props(
            "flat no-caps dense"
        )
        controls.append(button)
        return button

    def render_inline_diff(label: str, markup: str) -> None:
        ui.html(markup).classes("workbench-inline-diff w-full min-w-0").props(
            f'tabindex="0" role="region" aria-label="{label}"'
        )

    def show_inline_diff(label: str, markup: str, *, expanded: bool = False) -> None:
        with ui.expansion(label, icon="difference", value=expanded).classes(
            "w-full min-w-0"
        ):
            render_inline_diff(label, markup)

    def set_busy(value: bool) -> None:
        nonlocal busy
        busy = value
        original.set_enabled(not value)
        for control in [*controls, *review_controls]:
            control.set_enabled(not value)
        update_actions()

    def render_reports() -> None:
        score_area.clear()
        with score_area:
            reports = [state["input_scores"], state["output_scores"]]
            if any(reports):
                names = dict.fromkeys(
                    name for report in reports if report for name in report["scores"]
                )

                def score_method(report, name):
                    if not report or name not in report["scores"]:
                        return None
                    if report.get("detectors", {}).get(name, {}).get("window_scoring"):
                        return "window mean"
                    if report.get("method") not in {None, "single-section-native"}:
                        return "section mean"
                    return "native"

                def score_label(value, method):
                    if value is None:
                        return "—"
                    return f"{value:.4f}" + (
                        f" ({method})" if method != "native" else ""
                    )

                rows = []
                methods = set()
                mixed_methods = False
                for name in names:
                    before = reports[0]["scores"].get(name) if reports[0] else None
                    after = reports[1]["scores"].get(name) if reports[1] else None
                    before_method, after_method = (
                        score_method(report, name) for report in reports
                    )
                    methods.update((before_method, after_method))
                    comparable = before_method == after_method
                    mixed_methods |= (
                        before is not None and after is not None and not comparable
                    )
                    rows.append(
                        {
                            "detector": _DETECTOR_LABELS.get(name, name),
                            "input": score_label(before, before_method),
                            "output": score_label(after, after_method),
                            "change": f"{after - before:+.4f}"
                            if before is not None and after is not None and comparable
                            else "—",
                        }
                    )
                ui.table(
                    columns=[
                        {"name": key, "label": label, "field": key, "align": "left"}
                        for key, label in (
                            ("detector", "Detector"),
                            ("input", "Input"),
                            ("output", "Output"),
                            ("change", "Change"),
                        )
                    ],
                    rows=rows,
                    row_key="detector",
                ).classes("w-full").props(
                    'flat bordered dense :grid="Quasar.Screen.lt.sm"'
                )
                if "section mean" in methods:
                    ui.label(
                        "These scores combine document sections using their original "
                        "word counts as weights."
                    ).classes("text-sm text-slate-600")
                if "window mean" in methods:
                    ui.label(
                        "Window means average scores over overlapping portions of the text."
                    ).classes("text-sm text-slate-600")
                if mixed_methods:
                    ui.label(
                        "Change is shown when both scores use the same method."
                    ).classes("text-sm text-slate-600")
            else:
                ui.label(
                    "Score the input or output to compare detector results."
                ).classes("text-slate-600")
        raw_area.clear()
        raw_area.set_visibility(any(state.values()))
        with raw_area:
            if any(state.values()):
                with ui.expansion("Full report · JSON", icon="data_object").classes(
                    "w-full"
                ):
                    # Keep this dynamic button out of controls: its parent is
                    # replaced whenever the report changes.
                    ui.button(
                        "Copy report", icon="content_copy", on_click=copy_report
                    ).props("flat no-caps")
                    ui.code(report_text(), language="json").classes("w-full")

    def render_costs() -> None:
        cost_area.clear()
        cost_area.set_visibility(state["inspection"] is not None)
        if state["inspection"] is None:
            return
        with cost_area:
            if state["rewrite"] is None and state["yolo"] is None:
                ui.label("Provider cost: $0.00 USD · local processing").classes(
                    "text-sm text-slate-600 tabular-nums"
                )
                return
            ui.label("Run cost · USD").classes("text-sm font-semibold").props(
                "role=heading aria-level=3"
            )
            summary = state["cost_summary"]
            if summary is None:
                ui.label("Provider cost unavailable").classes("text-sm text-slate-600")
                return
            for key, label in (
                ("gateway_cost_usd", "Reported gateway"),
                ("upstream_cost_usd", "Reported upstream"),
            ):
                cost = summary[key]
                amount = cost["total"]
                if amount is None:
                    value = "Not reported"
                elif amount == 0:
                    value = "$0.00"
                elif amount < 0.00000001:
                    value = "<$0.00000001"
                else:
                    value = f"${amount:,.8f}".rstrip("0").rstrip(".")
                if amount is not None and cost["missing_requests"]:
                    value += (
                        f" (partial · {cost['reported_requests']} of "
                        f"{summary['request_count']} requests)"
                    )
                ui.label(f"{label}: {value}").classes(
                    "text-sm text-slate-700 tabular-nums [overflow-wrap:anywhere]"
                )
            count = summary["request_count"]
            ui.label(
                f"{count:,} provider {'request' if count == 1 else 'requests'} · "
                "includes all candidates, reviews and repairs for this run."
            ).classes("text-xs text-slate-600")
            ui.label(
                "Provider-reported amounts; unreported charges are unknown."
            ).classes("text-xs text-slate-600")
            if summary["completion_unknown_requests"]:
                unknown = summary["completion_unknown_requests"]
                ui.label(
                    f"{unknown:,} {'request has' if unknown == 1 else 'requests have'} "
                    "an unknown completion; charges may be missing."
                ).classes("text-sm text-amber-900")

    def render_review_evidence(row: dict) -> None:
        status_label = {
            "accepted": "Preservation review passed",
            "rejected": (
                "Material changes flagged"
                if any(item["severity"] == "material" for item in row["findings"])
                else "Preservation review refused"
            ),
            "unverified": "Preservation unverified",
        }[row["review_status"]]
        ui.label(status_label).classes("text-sm font-semibold")
        if row.get("detail"):
            ui.label(row["detail"]).classes("text-sm text-slate-600 break-words")
        for finding in row["findings"]:
            ui.label(
                f"{finding['severity'].capitalize()} · {finding['category']}"
            ).classes("text-sm font-medium")
            ui.label(finding["explanation"]).classes("text-sm break-words")
            ui.label("Source passage").classes("text-sm font-medium")
            ui.label(finding["source_quote"]).classes(
                "whitespace-pre-wrap [overflow-wrap:anywhere]"
            )
            ui.label("Draft passage").classes("text-sm font-medium")
            ui.label(finding["candidate_quote"] or "Omitted from draft").classes(
                "whitespace-pre-wrap [overflow-wrap:anywhere]"
            )
        for issue in row["blocking_issues"]:
            ui.label(str(issue).replace("_", " ")).classes(
                "text-sm text-amber-900 [overflow-wrap:anywhere]"
            )
        if row["advisory_issues"]:
            with ui.expansion("Local checks · advisory").classes("w-full"):
                for issue in row["advisory_issues"]:
                    ui.label(str(issue).replace("_", " ")).classes(
                        "text-sm [overflow-wrap:anywhere]"
                    )
        if row.get("provider_errors"):
            ui.label(
                "Some repair provider requests failed. Review is incomplete."
            ).classes("text-sm text-amber-900")

    def render_inspection(diff: str, alternatives=()) -> None:
        edits_area.clear()
        review_controls.clear()
        render_costs()
        alternatives = [*alternatives]
        for index, repair in enumerate(state["repairs"], start=1):
            text = repair["attempted_text"]
            alternatives.append(
                {
                    "title": f"Repair draft {index}",
                    "expanded": index == len(state["repairs"]),
                    "text": text,
                    "detail": "One repair attempt completed. Review the draft before using it.",
                    "diff_label": "Repair candidate diff",
                    "markup": inline_diff_html(original.value or "", text),
                    **_review_details(repair, repair["source_text"], text),
                    "provider_errors": repair.get("provider_errors", []),
                }
            )
        report = state["inspection"]
        rewrite = state["rewrite"]
        yolo = state["yolo"]
        result_verdict.classes(remove="text-teal-800 text-amber-900")
        result_hint.set_text("")
        if report is None:
            result_verdict.set_text("")
            return
        if rewrite is not None and rewrite["status"] == "provider_error":
            verdict = "Provider request failed · original retained"
        elif (
            rewrite is not None
            and rewrite["status"] == "unchanged_no_qualified_improvement"
        ):
            verdict = "Original retained · no rewrite qualified"
        elif report["refused"]:
            verdict = "Original retained · change refused"
        elif report["text"] != original.value:
            verdict = "Changes accepted"
        elif report.get("provider_errors"):
            verdict = "Original retained · some provider requests failed"
        else:
            verdict = "No changes made"
        if report["refused"] or report.get("provider_errors"):
            result_verdict.classes(add="text-amber-900")
        elif verdict == "Changes accepted":
            result_verdict.classes(add="text-teal-800")
        result_verdict.set_text(verdict)
        if verdict == "Changes accepted":
            result_hint.set_text("Automated checks passed. Review the changes.")
        with edits_area:
            if yolo is not None:
                show_yolo_stages(yolo)
            if report["refused"]:
                for reason in report["refusal_reasons"]:
                    ui.label(reason).classes(
                        "text-sm text-amber-900 [overflow-wrap:anywhere]"
                    )
            elif report["text"] == original.value:
                if yolo is None:
                    ui.label(
                        "No accepted rewrite was found."
                        if rewrite is not None
                        else "The local rules made no edits to this passage."
                    ).classes("text-sm text-slate-600")
            provider_errors = report.get("provider_errors", [])
            if provider_errors:
                if not report["refused"]:
                    ui.label(
                        "Some provider requests failed. The accepted text uses only "
                        "candidates that passed the checks."
                    ).classes("text-amber-900")
                statuses = sorted(
                    {
                        error["http_status"]
                        for error in provider_errors
                        if isinstance(error.get("http_status"), int)
                    }
                )
                if statuses:
                    ui.label(
                        "Provider HTTP responses: " + ", ".join(map(str, statuses))
                    ).classes("text-amber-900")
            if diff:
                show_inline_diff("Accepted output diff", diff, expanded=True)
            review_stages = (
                yolo["stages"]
                if yolo is not None
                else ({workflow.value: rewrite} if rewrite is not None else {})
            )
            stage_source = original.value or ""
            for method, stage in review_stages.items():
                stage_source = stage.get("source_text", stage_source)
                stage_output = stage.get("text", stage_source)
                if stage_output != stage_source:
                    selected_review = stage.get("final_review", {})
                    for candidate in stage.get("candidates", []):
                        if candidate.get("model") and candidate["model"] == stage.get(
                            "selected_model"
                        ):
                            selected_review = candidate
                            break
                    details = _review_details(
                        selected_review, stage_source, stage_output
                    )
                    if (
                        details["findings"]
                        or details["advisory_issues"]
                        or selected_review.get("review_status")
                    ):
                        title = (
                            f"{_WORKFLOW_OPTIONS[method]} · "
                            if yolo is not None
                            else ""
                        ) + "Preservation review"
                        with ui.expansion(title, icon="fact_check").classes(
                            "w-full min-w-0"
                        ):
                            with ui.column().classes("w-full min-w-0 gap-2 px-3 pb-2"):
                                render_review_evidence(details)
                stage_source = stage_output
            if alternatives:
                with ui.expansion(
                    "Review alternatives",
                    icon="visibility",
                    value=bool(state["repairs"]),
                ).classes("w-full min-w-0"):
                    ui.label(
                        "Compare drafts with your input. Use as draft replaces "
                        "your input and can be undone."
                    ).classes("px-3 text-sm text-slate-600")
                    for row in alternatives:
                        text = row["text"]
                        with ui.expansion(
                            row["title"], value=row.get("expanded", False)
                        ).classes("w-full min-w-0"):
                            with ui.column().classes("w-full min-w-0 gap-2 px-3 pb-2"):
                                render_review_evidence(row)
                                if row["markup"]:
                                    ui.label(row["diff_label"]).classes(
                                        "text-sm font-medium"
                                    )
                                    render_inline_diff(row["diff_label"], row["markup"])
                                ui.textarea(label="Candidate text", value=text).props(
                                    "readonly outlined rows=4"
                                ).classes("w-full")
                                with ui.row().classes("w-full gap-2"):
                                    buttons = [
                                        ui.button(
                                            "Copy candidate",
                                            icon="content_copy",
                                            on_click=lambda text=text: copy(text),
                                        ).props("flat no-caps"),
                                        ui.button(
                                            "Use as draft",
                                            on_click=lambda text=text: use_candidate(
                                                text
                                            ),
                                        ).props("outline no-caps"),
                                    ]
                                    material = [
                                        item
                                        for item in row["findings"]
                                        if item["severity"] == "material"
                                    ]
                                    can_repair = bool(
                                        material
                                        and hasattr(service, "repair_candidate")
                                        and max(len(row["source"]), len(text))
                                        <= MAX_TECHNICAL_CHARACTERS
                                    )
                                    if can_repair:
                                        buttons.append(
                                            ui.button(
                                                "Repair cited issues",
                                                icon="build",
                                                on_click=lambda row=row: repair_draft(
                                                    row
                                                ),
                                            ).props("outline no-caps")
                                        )
                                    for button in buttons:
                                        button.set_enabled(not busy)
                                    review_controls.extend(buttons)
                                if can_repair:
                                    ui.label(
                                        "Repair sends this draft, its source and cited issues to the provider, "
                                        "then requests a fresh independent review. Usage charges can apply."
                                    ).classes("text-sm text-slate-600")
                                elif material and hasattr(service, "repair_candidate"):
                                    ui.label(
                                        f"Repair supports source and draft text up to {MAX_TECHNICAL_CHARACTERS:,} characters each."
                                    ).classes("text-sm text-slate-600")
            if report["changes"]:
                with ui.expansion("Changes by rule", icon="list_alt").classes("w-full"):
                    for change in report["changes"]:
                        with ui.expansion(change["label"], icon="difference").classes(
                            "w-full"
                        ):
                            for edit in change["edits"]:
                                with ui.row().classes("items-center gap-3"):
                                    ui.label(edit["before"] or "(empty)").classes(
                                        "whitespace-pre-wrap break-all text-rose-800"
                                    )
                                    ui.icon("arrow_forward").classes("text-slate-400")
                                    ui.label(edit["after"] or "(empty)").classes(
                                        "whitespace-pre-wrap break-all text-teal-800"
                                    )

    async def repair_draft(row: dict) -> None:
        if busy:
            return
        findings = [item for item in row["findings"] if item["severity"] == "material"]
        if not findings:
            return
        set_busy(True)
        spinner.set_visibility(True)
        status.set_text("Repairing cited issues and requesting a fresh review…")
        try:
            result = await run.io_bound(
                service.repair_candidate,
                row["source"],
                row["text"],
                findings,
            )
            if result is None:
                return
            summaries = (state["cost_summary"], result.get("cost_summary"))
            state["cost_summary"] = (
                combine_cost_summaries(summaries)
                if all(summary is not None for summary in summaries)
                else None
            )
            render_costs()
            if result.get("generation_status") == "generated":
                state["repairs"].append({**result, "source_text": row["source"]})
                render_inspection(*inspection_previews)
                status.set_text("Repair draft is ready in Review alternatives.")
            else:
                status.set_text(
                    "Repair could not produce a new draft. Your candidates are still available."
                )
            render_reports()
        except (LLMConfigurationError, LLMModelCatalogError, ValueError) as error:
            status.set_text(str(error))
        except Exception:
            _LOG.exception("Candidate repair failed")
            status.set_text(
                "Repair failed. Your candidates are still available. Try again."
            )
        finally:
            spinner.set_visibility(False)
            set_busy(False)

    def show_yolo_stages(result: dict) -> None:
        stages = result["stages"]
        with ui.expansion("YOLO stage results", icon="list_alt", value=True).classes(
            "w-full"
        ):
            for method in ("adaptive", "technical-update-experimental"):
                stage = stages[method]
                if method == "adaptive" and result.get("adaptive_output_over_limit"):
                    outcome = (
                        "Output exceeded the next stage’s limit. Previous text kept."
                    )
                else:
                    outcome = {
                        "provider_error": "Provider request failed",
                        "reverted_final_review": "Preservation review refused the edits",
                        "unchanged_no_qualified_improvement": "No rewrite qualified",
                    }.get(
                        stage["status"],
                        "Edits kept" if stage["changed"] else "No edits kept",
                    )
                ui.label(f"{_WORKFLOW_OPTIONS[method]} · {outcome}")
            frozen = stages["frozen"]
            outcome = "No edits made"
            if frozen["refused"]:
                outcome = "Preservation checks refused the edits"
            elif frozen["text"] != stages["technical-update-experimental"]["text"]:
                outcome = "Edits kept"
            ui.label(f"Frozen rules · {outcome}")
            if frozen["refused"]:
                for reason in frozen["refusal_reasons"]:
                    ui.label(reason).classes("text-sm text-amber-900 break-words")

    def update_rank_button() -> None:
        text = original.value or ""
        rank_button.set_enabled(
            not busy
            and bool(rule_switches)
            and bool(text.strip())
            and len(text) <= MAX_DOCUMENT_CHARACTERS
        )

    def transform_limit() -> int:
        return (
            MAX_TECHNICAL_CHARACTERS
            if workflow.value in {"technical-update-experimental", "yolo"}
            else MAX_DOCUMENT_CHARACTERS
        )

    def update_actions() -> None:
        text = original.value or ""
        has_input = bool(text.strip())
        has_output = bool(output.value)
        for control, available in (
            (transform_button, has_input and len(text) <= transform_limit()),
            (score_input_button, has_input),
            (copy_input_button, bool(text)),
            (clear_button, bool(text)),
            (copy_output_button, has_output),
            (score_output, has_output),
            (view_result_button, has_output),
            (undo_button, draft_replacement is not None),
        ):
            control.set_enabled(not busy and available)
        undo_button.set_visibility(draft_replacement is not None)
        view_result_button.set_visibility(has_output)
        update_rank_button()

    def update_rule_visibility() -> None:
        query = (rule_search.value or "").strip().casefold()
        allowed_impacts = _IMPACT_FILTERS[impact_filter.value]
        impact_counts: Counter[str] = Counter()
        shown = 0
        for category, examples in grouped_examples.items():
            category_shown = False
            for example in examples:
                rule = example["id"]
                matches_other_filters = (
                    (rule_category_filter.value in {"All categories", category})
                    and (not selected_only.value or rule_switches[rule].value)
                    and (not query or query in " ".join(example.values()).casefold())
                )
                label = impact_labels[rule]
                if matches_other_filters:
                    impact_counts[label] += 1
                visible = matches_other_filters and (
                    allowed_impacts is None or label in allowed_impacts
                )
                rule_rows[rule].set_visibility(visible)
                category_shown |= visible
                shown += visible
            rule_headings[category].set_visibility(
                category_shown and state["rule_ranking"] is None
            )
        options = {}
        for label, bands in _IMPACT_FILTERS.items():
            count = (
                impact_counts.total()
                if bands is None
                else sum(impact_counts[band] for band in bands)
            )
            options[label] = f"{label} ({count})"
        if impact_filter.options != options:
            impact_filter.set_options(options)
        selected = sum(bool(switch.value) for switch in rule_switches.values())
        rule_summary.set_text(
            f"{len(available_rules)} rules across {len(grouped_examples)} categories "
            f"· {selected} selected · {shown} shown"
        )
        no_rules_match.set_visibility(shown == 0)

    def set_impact(
        rule: str, label: str, detail: str, source: str = "", *, figures: str = ""
    ) -> None:
        impact_labels[rule] = label
        impact_badges[rule].set_text(f"{source} · {label}" if source else label)
        impact_badges[rule].props(f"color={impact_color(label)}")
        impact_figures[rule].set_content(figures)
        impact_figures[rule].set_visibility(bool(figures))
        impact_details[rule].set_text(detail)
        impact_panels[rule].set_visibility(bool(detail or figures))

    def render_ranking() -> None:
        ranking = state["rule_ranking"]
        if ranking is None:
            for category, examples in grouped_examples.items():
                rule_headings[category].move()
                for example in examples:
                    rule = example["id"]
                    rule_rows[rule].move()
                    rank_badges[rule].set_visibility(False)
                    recorded = recorded_impacts.get(rule)
                    set_impact(
                        rule,
                        recorded_impact_label(recorded),
                        recorded["detail"] if recorded else "",
                        "Recorded" if recorded else "",
                        figures=_recorded_impact_figures(recorded) if recorded else "",
                    )
            update_rule_visibility()
            return
        unranked = {
            "unchanged": "No edit on this input",
            "refused": "Preservation check refused",
            "no_comparable_scores": "Comparable scores unavailable",
        }
        for result in ranking:
            rule = result["rule"]
            rule_rows[rule].move()
            badge = rank_badges[rule]
            badge.set_text(
                f"#{result['rank']} · {result['mean_delta']:+.4f} mean change"
                if result["status"] == "ranked"
                else unranked[result["status"]]
            )
            badge.set_visibility(result["status"] == "ranked")
            if result["status"] == "ranked":
                deltas = {
                    name: values["delta"]
                    for name, values in result.get("detectors", {}).items()
                }
                set_impact(
                    rule,
                    impact_label(result["mean_delta"]),
                    f"Current input · mean detector change {result['mean_delta']:+.4f} "
                    "on the 0–1 score scale (output minus input). "
                    "Each rule is tested alone. Combined effects can differ.",
                    "On this input",
                    figures=_impact_figure("Current input · score change", deltas),
                )
            else:
                set_impact(
                    rule,
                    unranked[result["status"]],
                    " ".join(result.get("refusal_reasons", [])),
                )
        update_rule_visibility()

    def change_rules() -> None:
        clear_output()
        update_rule_visibility()

    def restore_rules() -> None:
        for rule, switch in rule_switches.items():
            switch.set_value(rule in configured_rules)

    def clear_output() -> None:
        nonlocal inspection_previews
        inspection_previews = ("", [])
        state.update(
            inspection=None,
            output_scores=None,
            rewrite=None,
            yolo=None,
            repairs=[],
            cost_summary=None,
        )
        output.set_value("")
        render_inspection("")
        render_reports()
        status.set_text("")
        analysis_status.set_text("")
        update_actions()

    def reset() -> None:
        state["input_scores"] = None
        if state["rule_ranking"] is not None:
            state["rule_ranking"] = None
            render_ranking()
        length = len(original.value or "")
        limit = transform_limit()
        excess = length - limit
        count.set_text(f"{length:,} characters · Transform limit {limit:,}")
        validation_hint.set_text(
            f"Remove {excess:,} {'character' if excess == 1 else 'characters'} to edit this text."
            if excess > 0
            else ""
        )
        original.props["aria-invalid"] = "true" if excess > 0 else "false"
        clear_output()

    def input_changed() -> None:
        nonlocal draft_replacement
        if draft_replacement is not None and original.value != draft_replacement[1]:
            draft_replacement = None
        reset()

    def replace_draft(text: str) -> None:
        nonlocal draft_replacement
        previous = original.value or ""
        if busy or text == previous:
            return
        draft_replacement = (previous, text) if previous else None
        original.set_value(text)
        if previous:
            status.set_text(
                "Draft replaced. Undo is available."
                if text
                else "Draft cleared. Undo is available."
            )

    def use_candidate(text: str) -> None:
        if busy:
            return
        replace_draft(text)
        original.run_method("focus")
        original.client.run_javascript(
            f'requestAnimationFrame(() => document.getElementById("c{original.id}")'
            '?.scrollIntoView({block: "center"}))'
        )

    def undo_draft() -> None:
        nonlocal draft_replacement
        if busy or draft_replacement is None:
            return
        previous, _ = draft_replacement
        draft_replacement = None
        original.set_value(previous)
        status.set_text("Draft restored.")
        original.run_method("focus")

    def change_workflow() -> None:
        method = workflow.value
        use_llm = method != "frozen"
        rules_panel.set_visibility(not use_llm)
        rank_button.set_visibility(not use_llm)
        analysis_panel.props["caption"] = (
            "Compare detector scores for your draft."
            if use_llm
            else _LOCAL_ANALYSIS_CAPTION
        )
        summaries = {"frozen": _LOCAL_SUMMARY, "yolo": _YOLO_SUMMARY}
        summary = (
            _LLM_GUIDES[method]["summary"]
            if method in _LLM_GUIDES
            else summaries[method]
        )
        workflow_summary.set_text(summary)
        for strategy, panel in workflow_guides.items():
            panel.set_visibility(method == strategy)
        provider_note.set_visibility(use_llm)
        rounds.props["label"] = (
            "Sentence editing rounds" if method == "yolo" else "Rewrite rounds"
        )
        rounds.set_visibility(method in {"adaptive", "yolo"})
        review_policy.set_visibility(method in {"adaptive", "yolo"})
        review_policy_note.set_visibility(method in {"adaptive", "yolo"})
        reset()

    async def paste() -> None:
        source = original.value
        try:
            text = await ui.run_javascript(
                "try { return await navigator.clipboard.readText(); } "
                "catch { return null; }",
                timeout=30,
            )
        except TimeoutError:
            text = None
        if busy:
            return
        if text is None:
            status.set_text(
                "Allow clipboard access, or focus the draft and press Ctrl/Cmd+V."
            )
        elif source != original.value:
            status.set_text("The draft changed while pasting. Try Paste again.")
        elif not text:
            status.set_text("The clipboard is empty.")
        else:
            replace_draft(text)

    async def transform() -> None:
        nonlocal inspection_previews
        source = original.value or ""
        if busy or not source.strip() or len(source) > transform_limit():
            return
        method = workflow.value
        clear_output()
        set_busy(True)
        spinner.set_visibility(True)
        try:
            if method == "yolo":
                status.set_text("Running the three YOLO stages…")
                result = await run.io_bound(
                    service.run_yolo,
                    source,
                    rounds=int(rounds.value),
                    **(
                        {"review_policy": "contextual"}
                        if review_policy.value == "contextual"
                        else {}
                    ),
                )
                if result is None:
                    return
                state["yolo"] = result
                stages = result["stages"]
                summaries = [
                    stages[stage].get("cost_summary")
                    for stage in ("adaptive", "technical-update-experimental")
                ]
                state["cost_summary"] = (
                    combine_cost_summaries(summaries)
                    if all(summary is not None for summary in summaries)
                    else None
                )
                report = {
                    "text": result["text"],
                    "refused": False,
                    "changes": [],
                    "provider_errors": [
                        error
                        for stage in stages.values()
                        for error in stage.get("provider_errors", [])
                    ],
                }
            elif method != "frozen":
                status.set_text(
                    "Rewriting with the configured LLM provider and checking meaning…"
                )
                result = await run.io_bound(
                    service.rewrite_text,
                    source,
                    strategy=method,
                    rounds=int(rounds.value),
                    **(
                        {"review_policy": "contextual"}
                        if method == "adaptive" and review_policy.value == "contextual"
                        else {}
                    ),
                )
                if result is None:
                    return
                report = await run.io_bound(_rewrite_inspection, result, source)
                if report is None:
                    return
                state["rewrite"] = result
                state["cost_summary"] = result.get("cost_summary")
                stages = {method: result}
                for destination, key in (
                    ("input", "original"),
                    ("output", "rewritten"),
                ):
                    state[f"{destination}_scores"] = {
                        "method": result["scores"]["method"],
                        "scores": {
                            name: values[key]
                            for name, values in result["scores"]["detectors"].items()
                        },
                    }
            else:
                status.set_text(
                    "Checking edits and meaning… First use may download a model."
                )
                enabled = [
                    rule for rule in available_rules if rule_switches[rule].value
                ]
                report = await run.io_bound(
                    service.inspect_transformation,
                    source,
                    enabled_rules=enabled,
                    include_catalog=True,
                )
                if report is None:
                    return
                report["enabled_rules"] = enabled
                stages = {method: report}
                state["rewrite"] = None
                state["output_scores"] = None
            previews = await run.io_bound(
                _result_previews, source, report["text"], stages
            )
            if previews is None:
                return
            state["inspection"] = report
            output.set_value(report["text"])
            inspection_previews = previews
            render_inspection(*previews)
            render_reports()
            status.set_text("")
        except (LLMConfigurationError, LLMModelCatalogError) as error:
            status.set_text(str(error))
        except Exception:
            _LOG.exception("Text transformation failed")
            status.set_text("Processing failed. Your draft is still here. Try again.")
        finally:
            spinner.set_visibility(False)
            set_busy(False)

    async def rank_rules() -> None:
        source = original.value or ""
        if (
            busy
            or workflow.value != "frozen"
            or not source.strip()
            or len(source) > MAX_DOCUMENT_CHARACTERS
        ):
            return
        set_busy(True)
        analysis_status.set_text("Checking and scoring each rule separately…")
        try:
            ranking = await run.io_bound(
                service.rank_rules, source, include_catalog=True
            )
            if ranking is None:
                return
            state["rule_ranking"] = ranking
            render_ranking()
            render_reports()
            analysis_status.set_text("Rule ranking complete. See Local rules below.")
        except Exception:
            _LOG.exception("Rule ranking failed")
            analysis_status.set_text("Rule ranking failed. Try again.")
        finally:
            set_busy(False)

    async def score(which: str) -> None:
        text = (original.value if which == "input" else output.value) or ""
        if busy or not text.strip():
            return
        set_busy(True)
        analysis_status.set_text(
            f"Scoring {which}… First use may download detector models."
        )
        try:
            report = await run.io_bound(service.score_text, text, detailed=True)
            if report is None:
                return
            state[f"{which}_scores"] = report
            render_reports()
            analysis_status.set_text(f"{which.capitalize()} scored")
        except Exception:
            _LOG.exception("Detector scoring failed")
            analysis_status.set_text("Scoring failed. Try again.")
        finally:
            set_busy(False)

    async def score_input():
        await score("input")

    async def score_accepted():
        await score("output")

    async def copy_report():
        await copy(report_text())

    async def view_result() -> None:
        await ui.run_javascript(
            f'document.getElementById("c{result_heading.id}").focus()'
        )

    ui.add_css(INLINE_DIFF_CSS + WORKBENCH_CSS)
    ui.colors(
        primary="#0f766e", secondary="#334155", positive="#15803d", info="#0369a1"
    )
    ui.query("body").classes("bg-slate-50 text-slate-800")
    with ui.column().classes("workbench w-full max-w-7xl mx-auto gap-6 p-4 md:p-8"):
        with ui.row().classes("w-full items-center justify-between"):
            with ui.column().classes("gap-1"):
                ui.label("fakehuman").classes(
                    "text-3xl font-bold tracking-tight"
                ).props("role=heading aria-level=1")
                ui.label("Text processing workbench").classes("text-slate-600")
            ui.badge("LOCAL CHECKS", color="teal-8").props("outline role=none")
        ui.label(
            "Paste a draft, edit it, and review the changes before copying the result."
        ).classes("text-base text-slate-600")
        with ui.grid().classes(
            "w-full min-w-0 items-start gap-6 grid-cols-1 lg:grid-cols-2"
        ):
            with ui.card().classes("workbench-panel w-full p-5 gap-4"):
                with ui.row().classes(
                    "panel-heading w-full items-center justify-between"
                ):
                    ui.label("Draft").classes("text-lg font-semibold").props(
                        "role=heading aria-level=2"
                    )
                    with ui.row().classes("gap-2"):
                        controls.append(
                            ui.button(
                                "Paste", icon="content_paste", on_click=paste
                            ).props("flat no-caps dense")
                        )
                        copy_input_button = copy_button(
                            "Copy input", lambda: original.value or ""
                        )
                original = (
                    ui.textarea(
                        label="Original text",
                        placeholder="Paste a passage here, or load the example below…",
                    )
                    .props("outlined stack-label rows=8 input-class=leading-relaxed")
                    .classes("workbench-editor w-full")
                )
                with ui.row().classes(
                    "editor-tools w-full items-center justify-between"
                ):
                    count = ui.label(
                        f"0 characters · Transform limit {MAX_DOCUMENT_CHARACTERS:,}"
                    ).classes("text-xs text-slate-600 tabular-nums")
                    with ui.row().classes("gap-0"):
                        controls.append(
                            ui.button(
                                "Load example",
                                on_click=lambda: replace_draft(
                                    _LLM_GUIDES[workflow.value]["before"]
                                    if workflow.value in _LLM_GUIDES
                                    else _EXAMPLE
                                ),
                            ).props("flat no-caps")
                        )
                        clear_button = ui.button(
                            "Clear", on_click=lambda: replace_draft("")
                        ).props("flat no-caps color=secondary")
                        controls.append(clear_button)
                        undo_button = ui.button(
                            "Undo", icon="undo", on_click=undo_draft
                        ).props("flat no-caps")
                validation_hint = (
                    ui.label()
                    .classes("input-validation text-sm text-rose-800")
                    .props('role="status" aria-live="polite" aria-atomic="true"')
                )
                original.props(f'aria-describedby="c{count.id} c{validation_hint.id}"')
                workflow = (
                    ui.select(
                        _WORKFLOW_OPTIONS,
                        value="frozen",
                        label="Editing method",
                    )
                    .props("outlined")
                    .classes("w-full")
                )
                controls.append(workflow)
                workflow_summary = ui.label(_LOCAL_SUMMARY).classes(
                    "text-sm text-slate-600"
                )
                provider_note = ui.label(
                    "LLM rewriting sends your text to the configured provider "
                    "and can incur usage charges."
                ).classes("provider-note text-sm text-amber-900")
                provider_note.set_visibility(False)
                rounds = (
                    ui.select(
                        {1: "1 round", 2: "2 rounds", 3: "3 rounds"},
                        value=3,
                        label="Rewrite rounds",
                    )
                    .props("outlined")
                    .classes("w-full")
                )
                rounds.set_visibility(False)
                controls.append(rounds)
                review_policy = (
                    ui.select(
                        {
                            "strict": "Strict preservation",
                            "contextual": "Contextual review (experimental)",
                        },
                        value="strict",
                        label="Sentence preservation policy",
                    )
                    .props("outlined")
                    .classes("w-full")
                )
                review_policy.set_visibility(False)
                controls.append(review_policy)
                review_policy_note = ui.label(
                    "Contextual review uses source-cited LLM findings to decide preservation. "
                    "Local meaning and wording checks become advisory; review drafts carefully. "
                    "In YOLO, this applies to the sentence editing stage."
                ).classes("text-sm text-slate-600")
                review_policy_note.set_visibility(False)
                with ui.row().classes("editor-actions gap-3"):
                    transform_button = (
                        ui.button(
                            "Transform text", icon="auto_fix_high", on_click=transform
                        )
                        .props("unelevated no-caps")
                        .classes("px-5")
                    )
                    view_result_button = (
                        ui.button(
                            "View result", icon="arrow_downward", on_click=view_result
                        )
                        .props("flat no-caps")
                        .classes("lg:hidden")
                    )
                    controls.append(transform_button)
                with ui.row().classes("workbench-status items-center gap-2"):
                    spinner = ui.spinner(size="sm").props('aria-hidden="true"')
                    spinner.set_visibility(False)
                    status = (
                        ui.label()
                        .classes("text-sm text-slate-600")
                        .props('role="status" aria-live="polite" aria-atomic="true"')
                    )
            with ui.card().classes("workbench-panel w-full p-5 gap-4"):
                with ui.row().classes(
                    "panel-heading w-full items-center justify-between"
                ):
                    result_heading = (
                        ui.label("Result")
                        .classes("text-lg font-semibold")
                        .props("role=heading aria-level=2 tabindex=-1")
                    )
                output = (
                    ui.textarea(
                        label="Accepted text",
                        placeholder="Your result will appear here.",
                    )
                    .props(
                        "readonly outlined stack-label rows=8 input-class=leading-relaxed"
                    )
                    .classes("workbench-editor accepted-editor w-full")
                )
                with ui.column().classes("w-full min-w-0 gap-2"):
                    with ui.row().classes("w-full items-center gap-2"):
                        with ui.column().classes("min-w-0 flex-1 gap-0"):
                            result_verdict = (
                                ui.label()
                                .classes("font-semibold [overflow-wrap:anywhere]")
                                .props(
                                    'role="status" aria-live="polite" aria-atomic="true"'
                                )
                            )
                            result_hint = ui.label().classes("text-sm text-slate-600")
                        copy_output_button = copy_button(
                            "Copy output", lambda: output.value or ""
                        ).classes("shrink-0")
                    cost_area = ui.column().classes("w-full min-w-0 gap-1")
                    cost_area.set_visibility(False)
                    edits_area = ui.column().classes(
                        "review-details w-full min-w-0 gap-2"
                    )
        with ui.expansion(
            "Rules & analysis",
            icon="analytics",
            caption=_LOCAL_ANALYSIS_CAPTION,
        ).classes("workbench-panel analysis-panel w-full") as analysis_panel:
            with ui.column().classes("w-full min-w-0 gap-3 px-4 pb-4 md:px-5 md:pb-5"):
                ui.label("Detector scores").classes("text-base font-semibold").props(
                    "role=heading aria-level=3"
                )
                with ui.row().classes("w-full gap-2"):
                    score_input_button = ui.button(
                        "Score input", icon="analytics", on_click=score_input
                    ).props("outline no-caps")
                    score_output = ui.button(
                        "Score output", icon="analytics", on_click=score_accepted
                    ).props("outline no-caps")
                    rank_button = ui.button(
                        "Rank rules for input", icon="leaderboard", on_click=rank_rules
                    ).props("outline no-caps")
                    controls.extend((score_input_button, score_output))
                analysis_status = (
                    ui.label()
                    .classes("analysis-status text-sm text-slate-600")
                    .props('role="status" aria-live="polite" aria-atomic="true"')
                )
                score_area = ui.column().classes("w-full min-w-0")
                ui.label(
                    "Scores are classifier outputs, not calibrated authorship probabilities. "
                    "Lower scores do not establish human authorship or better writing. "
                    "Manual scoring averages overlapping windows for longer inputs. "
                    "LLM workflows report scores for the sections used during rewriting."
                ).classes("max-w-prose text-sm text-slate-600")
                raw_area = ui.column().classes("w-full min-w-0")
                with ui.column().classes(
                    "w-full min-w-0 gap-3 border-t border-slate-200 pt-4"
                ) as rules_panel:
                    ui.label("Local rules").classes("text-base font-semibold").props(
                        "role=heading aria-level=3"
                    )
                    rule_summary = ui.label().classes(
                        "text-sm font-medium text-slate-600"
                    )
                    ui.label(
                        f"The full catalog is available; {len(configured_rules)} default rules "
                        "are selected initially. Switch on any additional rules you want to try. "
                        "Default rules run first in pipeline order, followed by additional rules "
                        "in catalog order. Filtering does not change your selections."
                    ).classes("text-sm text-slate-600")
                    ui.label(
                        f"Recorded impacts cover {len(recorded_impacts)} rules. Rank rules for "
                        "input measures every rule separately on your passage, including rules "
                        "that are switched off or hidden by filters. More negative mean score "
                        "changes rank first."
                    ).classes("text-sm text-slate-600")
                    with ui.expansion(
                        "How detector impact is rated", icon="info"
                    ).classes("w-full"):
                        ui.label(
                            "Impact is the mean change across detectors on their 0–1 score scale. "
                            "Score increase: above +0.001. Negligible change: −0.001 to +0.001. "
                            "Small reduction: below −0.001 to above −0.01. Moderate reduction: "
                            "−0.01 to above −0.05. Large reduction: −0.05 or lower. "
                            "These are descriptive display bands, not statistical significance. "
                            "A lower score does not establish better writing or human authorship. "
                            "Open Impact details to see measurements and their source. "
                            "Recorded ratings average accepted-output changes over every AI "
                            "benchmark passage; unchanged and refused outputs count as zero. "
                            "Impact details also shows human controls and separate authored "
                            "matching examples, which do not determine the recorded rating. "
                            "No AI matches means the rule never triggered on that sample. "
                            "All edits refused means every attempted AI-corpus edit failed "
                            "preservation checks. "
                            "Not yet measured means no source-matched individual evaluation is available."
                        ).classes("text-sm text-slate-600")
                    with ui.grid().classes("w-full gap-3 grid-cols-1 md:grid-cols-3"):
                        rule_search = (
                            ui.input(
                                "Find a rule",
                                placeholder="Search names, descriptions or examples",
                            )
                            .props("outlined clearable stack-label")
                            .classes("w-full")
                        )
                        rule_category_filter = (
                            ui.select(
                                ["All categories", *grouped_examples],
                                value="All categories",
                                label="Rule category",
                            )
                            .props("outlined")
                            .classes("w-full")
                        )
                        impact_filter = (
                            ui.select(
                                list(_IMPACT_FILTERS),
                                value="All impacts",
                                label="Detector impact",
                            )
                            .props("outlined")
                            .classes("w-full")
                        )
                    with ui.row().classes("w-full items-center justify-between"):
                        selected_only = ui.checkbox("Selected rules only")
                        controls.append(
                            ui.button("Restore defaults", on_click=restore_rules).props(
                                "flat no-caps"
                            )
                        )
                    for control in (
                        rule_search,
                        rule_category_filter,
                        impact_filter,
                        selected_only,
                    ):
                        control.on_value_change(update_rule_visibility)
                        controls.append(control)
                    no_rules_match = ui.label("No rules match these filters.").classes(
                        "text-slate-600"
                    )
                    with (
                        ui.column()
                        .classes("rule-list w-full min-w-0 gap-2")
                        .props(
                            'tabindex="0" role="region" aria-label="Editing rule catalog"'
                        )
                    ):
                        for category, examples in grouped_examples.items():
                            rule_headings[category] = (
                                ui.label(category)
                                .classes(
                                    "rule-category text-sm font-semibold text-slate-700"
                                )
                                .props("role=heading aria-level=3")
                            )
                            for example in examples:
                                rule = example["id"]
                                with ui.column().classes(
                                    "rule-row w-full min-w-0 gap-1"
                                ) as row:
                                    rule_rows[rule] = row
                                    with ui.row().classes(
                                        "w-full min-w-0 items-center justify-between gap-2"
                                    ):
                                        switch = ui.switch(
                                            example["label"],
                                            value=rule in configured_rules,
                                        )
                                        switch.on_value_change(change_rules)
                                        rule_switches[rule] = switch
                                        controls.append(switch)
                                        badge = ui.badge(color="slate-700").props(
                                            "role=none"
                                        )
                                        badge.set_visibility(False)
                                        rank_badges[rule] = badge
                                        impact_badges[rule] = ui.badge().props(
                                            "outline role=none"
                                        )
                                    ui.label(example["description"]).classes(
                                        "text-sm text-slate-600"
                                    )
                                    with ui.expansion(
                                        "Impact details", icon="analytics"
                                    ).classes("w-full") as impact_panel:
                                        impact_panels[rule] = impact_panel
                                        impact_figures[rule] = (
                                            ui.html()
                                            .classes("mb-3 grid w-full min-w-0 gap-2")
                                            .style(
                                                "grid-template-columns: repeat(auto-fit, minmax(min(17rem, 100%), 1fr))"
                                            )
                                        )
                                        impact_details[rule] = ui.label().classes(
                                            "text-sm text-slate-600 break-words whitespace-pre-line w-full"
                                        )
                                    if diff := inline_diff_html(
                                        example["before"], example["after"]
                                    ):
                                        show_inline_diff("Example diff", diff)
        for strategy, guide in _LLM_GUIDES.items():
            with ui.card().classes("workbench-panel w-full p-5 gap-4") as guide_panel:
                workflow_guides[strategy] = guide_panel
                guide_panel.set_visibility(False)
                ui.label(guide["title"]).classes("text-lg font-semibold").props(
                    "role=heading aria-level=2"
                )
                with ui.grid().classes("w-full gap-4 grid-cols-1 md:grid-cols-3"):
                    for number, (title, detail) in enumerate(guide["steps"], start=1):
                        with ui.column().classes("gap-1"):
                            ui.label(f"{number}. {title}").classes(
                                "text-sm font-semibold text-slate-700"
                            )
                            ui.label(detail).classes("text-sm text-slate-600")
                ui.label(
                    "Illustrative example · actual edits depend on your text and the checks."
                ).classes("text-sm text-slate-600")
                show_inline_diff(
                    "Example diff",
                    inline_diff_html(guide["before"], guide["after"]),
                    expanded=True,
                )
                ui.label(guide["caption"]).classes("text-sm text-slate-600")
                controls.append(
                    ui.button(
                        "Use example as input",
                        on_click=lambda example=guide["before"]: replace_draft(example),
                    ).props("outline no-caps")
                )
        ui.label(
            "Detection and meaning checks run locally. First use downloads pinned "
            "checkpoints unless offline. LLM rewriting uses your configured provider."
        ).classes("text-xs text-slate-600")
    original.on_value_change(input_changed)
    workflow.on_value_change(change_workflow)
    rounds.on_value_change(reset)
    review_policy.on_value_change(reset)
    render_reports()
    render_ranking()
    update_actions()


def run_ui(manifest: dict, *, host: str = "127.0.0.1", port: int = 8080) -> None:
    """Launch the browser workbench without affecting the default stdio entry point."""
    try:
        from nicegui import ui
    except ModuleNotFoundError as exc:
        if exc.name != "nicegui":
            raise
        raise RuntimeError(
            "The browser UI requires NiceGUI. Install it with "
            "'uv sync --extra cpu --extra ui', then run "
            "'uv run --extra cpu --extra ui fakehuman ui'."
        ) from exc

    import torch

    from .service import TextService

    torch.set_num_threads(4)
    service = TextService(manifest)

    @ui.page("/")
    def index():
        _build_page(service)

    try:
        ui.run(
            host=host,
            port=port,
            title="fakehuman · Text workbench",
            reload=False,
            show=False,
        )
    except KeyboardInterrupt:
        pass
