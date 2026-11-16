# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: William Droz <william.droz@idiap.ch>
# SPDX-License-Identifier: GPL-3.0-only

import pytest

from fakehuman.formatted_text import (
    detect_format,
    prepare_formatted_text,
)


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("An ordinary email paragraph.", "plain"),
        ("Use sample_id and p_value.", "plain"),
        ("A *short* update.", "markdown"),
        ("Title\n=====\n", "markdown"),
        ("Read [details](https://example.org).", "markdown"),
        ("```html\n<p>Example.</p>\n```", "markdown"),
        ("<html><body><p>An update.</p></body></html>", "html"),
        ("A <strong>short</strong> update.", "html"),
        ("Compare a < b > c.", "plain"),
    ],
)
def test_format_detection(source, expected):
    assert detect_format(source) == expected


@pytest.mark.parametrize(
    ("source", "format"),
    [
        (
            "Dear Alex,\r\n\r\nAn email with\r\nsoft wrapping.\r\n\r\nBest,\r\nWilliam",
            "plain",
        ),
        ("# Head\r\n\r\n- A **bold** item that\r\n  continues here.\r\n", "markdown"),
        (
            "<!DOCTYPE html>\r\n<HTML><body><p data-x='a>b'>One &amp; two.</p><!-- x --></body></HTML>",
            "html",
        ),
        (
            "<p>Text.</p><script>if (a < 2) { alert('x'); }</script><pre>  exact\n code</pre>",
            "html",
        ),
        ("", "plain"),
        ("", "markdown"),
        ("", "html"),
    ],
)
def test_identity_round_trip_is_byte_exact(source, format):
    document = prepare_formatted_text(source, format)
    assert document.source == source
    assert document.issues(document.text) == []
    assert document.restore(document.text) == source
    assert document.metadata()["format"] == format


def test_wrapped_nested_lists_and_quotes_keep_hierarchy_and_unedited_source():
    source = (
        "# Updates\n\n- First item has\n  a continuation.\n"
        "  - Nested item has\n    another continuation.\n"
        "    12. A deeper item has\n        a continuation.\n\n"
        "> - A quoted item has\n>   a continuation.\n"
    )
    document = prepare_formatted_text(source)
    assert document.text == (
        "# Updates\n\n- First item has a continuation.\n"
        "  - Nested item has another continuation.\n"
        "    12. A deeper item has a continuation.\n\n"
        "> - A quoted item has a continuation.\n"
    )
    edited = document.text.replace("Nested item", "Inner item")
    assert document.issues(edited) == []
    assert document.restore(edited) == source.replace(
        "  - Nested item has\n    another continuation.",
        "  - Inner item has another continuation.",
    )
    assert document.issues(edited.replace("  - Inner", "- Inner"))


def test_hard_breaks_paragraphs_setext_and_signature_are_kept():
    source = "Title\n=====\n\nFirst line.  \nSecond line.\\\nThird.\n\nBest,\nWilliam\n"
    document = prepare_formatted_text(source, "markdown")
    assert document.text == source
    assert document.issues(document.text.replace("Title", "Other"))
    assert document.issues(document.text.replace("William", "Someone"))
    assert document.issues(document.text.replace("line.  \n", "line.\n"))


def test_markdown_rules_cannot_touch_code_destinations_or_table_cells():
    source = (
        "# Update\n\nWe utilize `utilize()` and [details](https://example.org/utilize_(x)).\n\n"
        "```python\nutilize = 12\n```\n\n| utilize | cell |\n"
    )
    document = prepare_formatted_text(source)
    assert "utilize()" not in document.text
    assert "https://" not in document.text
    assert "utilize = 12" not in document.text
    edited = document.text.replace("utilize", "use")
    assert document.issues(edited) == []
    assert document.restore(edited) == source.replace("We utilize", "We use")
    assert "utilize = 12" not in document.visible_text()
    assert "details" in document.visible_text()
    assert document.issues(edited.replace(document._tokens[0], "changed"))


def test_markdown_visible_text_preserves_identifiers_escaped_markers_and_entities():
    source = "# Heading\nUse sample_id and p_value, \\*literal\\*, &lt;literal&gt; and **bold**."
    visible = prepare_formatted_text(source).visible_text()
    assert "sample_id and p_value" in visible
    assert "*literal*" in visible
    assert "<literal>" in visible
    assert "**bold**" not in visible
    assert "bold" in visible


def test_markdown_multiline_script_and_style_are_opaque():
    source = "# Title\n<script>\nsecret = 42;\n</script>\n<style>\nsecret-rule {}\n</style>\nVisible prose."
    document = prepare_formatted_text(source, "markdown")
    assert "secret" not in document.text
    assert "secret" not in document.visible_text()
    assert document.restore(
        document.text.replace("Visible prose", "Readable prose")
    ) == source.replace("Visible prose", "Readable prose")


def test_html_edits_preserve_attributes_entities_styles_and_scripts():
    source = (
        "<!DOCTYPE html><html><head><style>.x { color: red }</style></head><body>"
        '<p class="utilize" data-number="99">We utilize <strong>helpful</strong> tools &amp; checks.</p>'
        "<script>utilize(99)</script><p>Another\n wrapped paragraph.</p></body></html>"
    )
    document = prepare_formatted_text(source)
    assert "data-number" not in document.text
    assert "utilize(99)" not in document.text
    assert "99" not in document.visible_text()
    assert "We utilize helpful tools & checks." in document.visible_text()
    edited = document.text.replace("We utilize", "We use")
    assert document.issues(edited) == []
    assert document.restore(edited) == source.replace("We utilize", "We use")


def test_html_entities_decode_once_without_losing_visible_punctuation():
    document = prepare_formatted_text(
        "<p>&gt; 5 values, &lt;literal&gt;, &#38; and &amp;copy;.</p>"
    )
    assert document.visible_text() == "> 5 values, <literal>, & and &copy;."


def test_html_new_metacharacters_are_escaped_and_tags_are_rejected():
    document = prepare_formatted_text("<p>Use this text.</p>")
    revised = document.text.replace("Use this text.", "Use x < 5 & y > 1.")
    assert document.restore(revised) == "<p>Use x &lt; 5 &amp; y &gt; 1.</p>"
    revised = document.text.replace("Use this text.", "Use x < 5 & y.")
    assert document.restore(revised) == "<p>Use x &lt; 5 &amp; y.</p>"
    injected = document.text.replace("Use this text.", "<script>alert(1)</script>")
    assert "html_markup_added" in document.issues(injected)
    with pytest.raises(ValueError, match="html_markup_added"):
        document.restore(injected)


@pytest.mark.parametrize(
    "source",
    [
        "<script>alert(1);</script><p>Hello.</p>",
        "<pre><code>x = 1</code></pre><p>Hello.</p>",
        "<style>.x{color:red}</style><p>Hello.</p>",
    ],
)
def test_html_opaque_regions_cannot_be_extended_through_empty_gaps(source):
    document = prepare_formatted_text(source, "html")
    edited = document.text.replace(
        document._tokens[0], document._tokens[0] + "INJECTED;", 1
    )
    assert "protected_html_changed" in document.issues(edited)
    with pytest.raises(ValueError):
        document.restore(edited)


def test_unterminated_html_script_is_opaque():
    source = "<p>Hello.</p><script>secret = 99"
    document = prepare_formatted_text(source, "html")
    assert "secret" not in document.text
    assert "99" not in document.visible_text()
    assert document.restore(document.text.replace("Hello", "Hi")) == source.replace(
        "Hello", "Hi"
    )


def test_partial_projections_do_not_require_all_tokens_or_expose_metadata():
    document = prepare_formatted_text(
        "<p>First <strong>sentence</strong>.</p><p>Second.</p>"
    )
    left = document.text.index("First")
    right = document.text.index("Second")
    assert "First sentence." in document.visible_text(document.text[left:right])
    token = document._tokens[0]
    assert "hf-" not in document.visible_text(token[:10])
    assert document.visible_text(token[10:] + "Words.") == "Words."


def test_plain_supports_layout_changes_and_identity_preserves_wrapping():
    source = "A soft\nwrap.\n\nSecond paragraph."
    document = prepare_formatted_text(source, "plain")
    assert document.text == "A soft wrap.\n\nSecond paragraph."
    assert document.restore(document.text) == source
    assert document.issues("New paragraph.") == []
    assert document.restore("New paragraph.") == "New paragraph."


def test_invalid_format_and_nonstring_inputs_fail_locally():
    with pytest.raises(ValueError, match="format"):
        prepare_formatted_text("text", "rtf")
    with pytest.raises(TypeError):
        prepare_formatted_text(None)  # ty: ignore[invalid-argument-type] -- deliberate invalid input
