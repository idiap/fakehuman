# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: William Droz <william.droz@idiap.ch>
# SPDX-License-Identifier: GPL-3.0-only

import json

import pytest

from fakehuman.adaptive import (
    generation_body,
    review_body,
    sentence_spans,
)
from fakehuman.document_guards import (
    editable_indices,
    enrich_request,
    format_issues,
)


def test_editor_can_change_prose_and_bullet_but_not_greeting_or_signature():
    source = (
        "Dear Daniel and Ravinithesh,\n\n"
        "An update follows. Further details are available.\n\n"
        "- **Editing:** Authors can recover drafts.\n"
        "- **News:** Editors can reorder stories.\n\n"
        "Best,\\\n&#x20;      William"
    )
    eligible = editable_indices(source)
    selected = [
        source[a:b] for i, (a, b) in enumerate(sentence_spans(source)) if i in eligible
    ]
    assert selected == [
        "Further details are available.",
        "- **Editing:** Authors can recover drafts.",
        "- **News:** Editors can reorder stories.",
    ]
    revised = source.replace(
        "Authors can recover drafts.", "Authors may recover drafts."
    )
    assert format_issues(source, revised) == []
    assert "protected_line_changed" in format_issues(
        source, source.replace("William", "Will")
    )
    assert "protected_line_changed" in format_issues(
        source, source.replace("Dear Daniel", "Hi Daniel")
    )


@pytest.mark.parametrize(
    ("before", "after", "issue"),
    [
        ("One.\n\nTwo.", "One.\nTwo.", "line_boundaries_changed"),
        ("One.\r\nTwo.", "One.\nTwo.", "line_boundaries_changed"),
        ("- **Label:** A sentence.", "+ **Label:** A sentence.", "line_prefix_changed"),
        ("- **Label:** A sentence.", "- **Other:** A sentence.", "line_prefix_changed"),
        ("A **bold** sentence.", "A bold sentence.", "emphasis_markers_changed"),
        (
            "Read [details](https://example.org).",
            "Read [details](https://other.org).",
            "protected_markup_changed",
        ),
        ("Run `prepare()` first.", "Run `delete()` first.", "protected_markup_changed"),
        ("Use <br> here.", "Use <hr> here.", "protected_markup_changed"),
        ("A line.  \n", "A line.\n", "line_ending_format_changed"),
        ("# Original heading\n", "# New heading\n", "protected_line_changed"),
        (
            "```python\nx = 1\n```\n",
            "```python\nx = 2\n```\n",
            "protected_line_changed",
        ),
    ],
)
def test_document_format_changes_are_rejected(before, after, issue):
    assert issue in format_issues(before, after)
    assert format_issues(before, before) == []


def test_nonprose_sentences_are_ineligible():
    source = (
        "# Heading.\n\n"
        "```text\nCode stays.\n```\n\n"
        "- **Bullet:** A sentence.\n"
        "Another prose sentence.\n\n"
        "Best,\nWilliam."
    )
    eligible = editable_indices(source)
    assert all(
        "Code" not in source[a:b] and "William" not in source[a:b]
        for i, (a, b) in enumerate(sentence_spans(source))
        if i in eligible
    )
    assert any(
        source[a:b] == "Another prose sentence."
        for i, (a, b) in enumerate(sentence_spans(source))
        if i in eligible
    )


def test_generation_keeps_original_indices_and_supplies_context_without_mutation():
    source = (
        "Intro.\n\nDear Daniel,\n\nFirst sentence. Second sentence.\n\nBest,\nWilliam"
    )
    chunk = "Dear Daniel,\n\nFirst sentence. Second sentence."
    start = source.index(chunk)
    body = generation_body(chunk, chunk, 0.5, 1, 1)
    before = json.dumps(body)
    enriched = enrich_request(
        body, source_document=source, chunk_span=(start, start + len(chunk))
    )
    payload = json.loads(enriched["messages"][1]["content"])
    assert payload["sentences"] == [{"sentence_index": 1, "text": "Second sentence."}]
    assert payload["document_context"] == {
        "scope": "full_document",
        "original": source,
        "current": source,
    }
    assert json.dumps(body) == before


def test_review_has_full_documents_and_cannot_leak_selection_metadata():
    source = (
        "October updates:\n\nAuthors can recover work.\n\nSeptember updates follow."
    )
    chunk = "Authors can recover work."
    revised = "Authors can recover their work."
    start = source.index(chunk)
    body = review_body(chunk, revised)
    payload = json.loads(body["messages"][1]["content"])
    payload.update({"current_fakespot_score": 0.3, "round": 1, "format_issues": []})
    body["messages"][1]["content"] = json.dumps(payload)
    enriched = enrich_request(
        body, source_document=source, chunk_span=(start, start + len(chunk))
    )
    payload = json.loads(enriched["messages"][1]["content"])
    assert set(payload) == {"before", "after", "document_context"}
    assert payload["document_context"] == {
        "scope": "full_document",
        "before": source,
        "after": source.replace(chunk, revised),
    }


def test_large_document_context_keeps_neighbors_and_reference_binding():
    source = "A" * 10_000 + "Original sentence." + "Z" * 10_000
    chunk = "Original sentence."
    span = (10_000, 10_000 + len(chunk))
    enriched = enrich_request(
        review_body(chunk, "Revised sentence."), source_document=source, chunk_span=span
    )
    context = json.loads(enriched["messages"][1]["content"])["document_context"]
    assert context["scope"] == "neighboring_context"
    assert context["before"] == "A" * 2_000 + chunk + "Z" * 2_000
    assert context["after"] == "A" * 2_000 + "Revised sentence." + "Z" * 2_000
    with pytest.raises(ValueError, match="does not match"):
        enrich_request(
            review_body("Different reference.", chunk),
            source_document=source,
            chunk_span=span,
        )


@pytest.mark.parametrize(
    "source", ["```\nKeep this code.\n```", "Best,\nKeep this code."]
)
def test_generation_inherits_protected_structure_outside_chunk(source):
    chunk = "Keep this code."
    start = source.index(chunk)
    enriched = enrich_request(
        generation_body(chunk, chunk, 0.5, 1, 1),
        source_document=source,
        chunk_span=(start, start + len(chunk)),
    )
    assert json.loads(enriched["messages"][1]["content"])["sentences"] == []
