# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: William Droz <william.droz@idiap.ch>
# SPDX-License-Identifier: GPL-3.0-only

"""Lossless formatting adapters for editing prose without editing its markup.

Markdown/email soft wraps become editable logical lines. Unchanged logical
lines restore their exact source, including CRLF and wrapping. Changed lines
are emitted without soft wrapping. Unsupported blocks stay opaque.

HTML is never rendered or executed. Tags, attributes, entities and code become
immutable placeholders; only text nodes are editable. Restoring an edited text
node escapes HTML metacharacters. This protects syntax, not factual meaning.
"""

from __future__ import annotations

import hashlib
import html
import re
from dataclasses import dataclass
from html.parser import HTMLParser

_FORMATS = {"plain", "markdown", "html"}
_HTML_TAG = re.compile(
    r"<(?:!doctype\s+html\b|/?(?:html|head|body|p|div|span|br|hr|a|b|i|em|"
    r"strong|u|s|ul|ol|li|blockquote|table|thead|tbody|tr|td|th|h[1-6]|pre|"
    r"code|img|style|script|section|article|font|meta|link)\b[^<>]*>)",
    re.I,
)
_MARKDOWN = re.compile(
    r"(?m)^[ \t]*(?:#{1,6}\s|>\s?|[-+*]\s|\d+[.)]\s|`{3,}|~{3,}|\[[^\]]+\]:)"
    r"|^[=-]{3,}[ \t]*$|!?\[[^\]\n]+\]\([^\n]+\)|(?<!\\)(?P<delimiter>\*{1,2}|(?<!\w)_{1,2}|~~|`)[^\n]+?(?P=delimiter)"
)
_GREETING = re.compile(r"^\s*(?:Dear|Hi|Hello|Hey)\b[^.!?\n]*[,!]\s*$", re.I)
_SIGNOFF = re.compile(
    r"^\s*(?:Best|Regards|Kind regards|Best regards|Warm regards|Sincerely|"
    r"Yours sincerely|Many thanks|Thanks)[,!]?\s*\\?\s*$",
    re.I,
)
_QUOTE = re.compile(r"^(?P<indent>[ \t]*)(?P<quote>(?:>[ \t]*)*)")
_LIST = re.compile(r"(?:[-+*]|\d+[.)])[ \t]+(?:\[[ xX]\][ \t]+)?")
_MARKERS = re.compile(r"(?<!\\)(?:\*{1,3}|(?<!\w)_{1,3}|_{1,3}(?!\w)|~~)")
_FRAGMENTS = re.compile(
    r"(`+)[^\n]*?\1"
    r"|!?\[[^\]\n]*\]\([^\n]*?\)"
    r"|!?\[[^\]\n]*\]\[[^\]\n]*\]"
    r"|<[^<>\n]+>"
    r"|&(?:#[0-9]+|#x[0-9a-fA-F]+|[A-Za-z][A-Za-z0-9]+);"
    r"|https?://[^\s<>]+"
    r"|\\[^\w\s]"
)
_BLOCK_TAGS = {
    "html",
    "body",
    "p",
    "div",
    "br",
    "hr",
    "ul",
    "ol",
    "li",
    "blockquote",
    "table",
    "thead",
    "tbody",
    "tr",
    "td",
    "th",
    "section",
    "article",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "pre",
}
_OPAQUE_TAGS = {
    "head",
    "script",
    "style",
    "code",
    "pre",
    "template",
    "textarea",
    "svg",
    "math",
}
_VOID_TAGS = {
    "area",
    "base",
    "br",
    "col",
    "embed",
    "hr",
    "img",
    "input",
    "link",
    "meta",
    "param",
    "source",
    "track",
    "wbr",
}
_PLACEHOLDER = re.compile(r"<hf-[^<>\r\n]+>")


def detect_format(text: str) -> str:
    """Conservatively detect common HTML or Markdown, otherwise plain text.

    A Markdown link/autolink containing a URL is not HTML. Explicit format
    selection remains available for ambiguous fragments or literal markup.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if re.search(r"(?m)^[ \t]*(?:`{3,}|~{3,}|#{1,6}\s)", text):
        return "markdown"
    if _HTML_TAG.search(text):
        return "html"
    return "markdown" if _MARKDOWN.search(text) else "plain"


def _body(raw: str) -> str:
    return re.sub(r"(?:\r\n|\r|\n)$", "", raw)


def _ending(raw: str) -> str:
    return raw[len(_body(raw)) :]


def _parts(line: str) -> tuple[str, str, str, str]:
    """Return indentation, quote prefix, list marker and remaining prose."""
    match = _QUOTE.match(line)
    assert match is not None
    indent, quote = match["indent"], match["quote"]
    tail = line[match.end() :]
    marker = _LIST.match(tail)
    return (
        indent,
        quote,
        marker[0] if marker else "",
        tail[marker.end() :] if marker else tail,
    )


def _prefix(line: str) -> str:
    indent, quote, marker, tail = _parts(line)
    heading = re.match(r"#{1,6}[ \t]+", tail)
    return indent + quote + marker + (heading[0] if heading else "")


def _hard_break(line: str) -> bool:
    return bool(re.search(r"(?: {2,}|\\)$", line))


def _inline_signature(line: str) -> tuple:
    trailing = re.search(r"[ \t]*\\?$", line)
    assert trailing is not None  # The pattern always matches at least an empty suffix.
    return (
        _prefix(line),
        tuple(_MARKERS.findall(line)),
        tuple(match[0] for match in _FRAGMENTS.finditer(line)),
        trailing[0],
    )


@dataclass(frozen=True)
class _LogicalLine:
    source: str
    text: str
    protected: bool = False
    opaque: bool = False


def _logical_lines(source: str) -> tuple[_LogicalLine, ...]:
    """Join only unambiguous soft wraps, retaining list/quote block structure."""
    lines = source.splitlines(keepends=True)
    result: list[_LogicalLine] = []
    fence: tuple[str, int] | None = None
    signature = False
    html_opaque: list[str] = []
    index = 0
    while index < len(lines):
        raw = lines[index]
        line = _body(raw)
        indent, quote, marker, content = _parts(line)
        stripped = content.lstrip()
        fence_match = re.match(r"(`{3,}|~{3,})", stripped)
        in_fence = fence is not None
        if fence_match:
            if fence is None:
                fence = (fence_match[1][0], len(fence_match[1]))
            elif fence_match[1][0] == fence[0] and len(fence_match[1]) >= fence[1]:
                fence = None
        signature = signature or bool(_SIGNOFF.fullmatch(line))
        opaque_html = bool(html_opaque)
        for tag_match in re.finditer(r"<(\/?)([a-zA-Z][a-zA-Z0-9]*)\b[^>]*>", line):
            tag = tag_match[2].lower()
            if tag in _OPAQUE_TAGS:
                opaque_html = True
                if tag_match[1]:
                    if tag in html_opaque:
                        html_opaque = html_opaque[
                            : len(html_opaque) - 1 - html_opaque[::-1].index(tag)
                        ]
                else:
                    html_opaque.append(tag)
        setext = bool(
            index + 1 < len(lines)
            and re.fullmatch(r"[ \t]*[=-]{3,}[ \t]*", _body(lines[index + 1]))
        )
        protected = bool(
            in_fence
            or fence_match
            or opaque_html
            or setext
            or signature
            or not stripped
            or _GREETING.fullmatch(line)
            or re.match(r"#{1,6}(?:[ \t]|$)", stripped)
            or re.fullmatch(r"(?:[-*_][ \t]*){3,}|={3,}", stripped)
            or re.match(r"\[[^\]]+\]:[ \t]*\S", stripped)
            or "|" in line
            or stripped.startswith(("<", "$$"))
            or (not marker and len(indent.expandtabs(4)) >= 4)
            or (line.count("`") % 2 != 0)
        )
        if protected:
            opaque = bool(
                in_fence
                or fence_match
                or opaque_html
                or "|" in line
                or stripped.startswith(("<", "$$"))
                or (not marker and len(indent.expandtabs(4)) >= 4)
                or (line.count("`") % 2 != 0)
            )
            result.append(_LogicalLine(raw, raw, True, opaque))
            index += 1
            continue
        originals = [raw]
        normalized = line
        previous = line
        index += 1
        while index < len(lines) and not _hard_break(previous):
            following_raw = lines[index]
            following = _body(following_raw)
            next_indent, next_quote, next_marker, next_content = _parts(following)
            next_stripped = next_content.lstrip()
            if (
                not next_stripped
                or next_marker
                or next_quote.count(">") != quote.count(">")
                or _GREETING.fullmatch(following)
                or _SIGNOFF.fullmatch(following)
                or re.match(
                    r"(?:#{1,6}(?:[ \t]|$)|`{3,}|~{3,}|\[[^\]]+\]:)", next_stripped
                )
                or re.fullmatch(r"(?:[-*_][ \t]*){3,}|={3,}", next_stripped)
                or "|" in following
                or next_stripped.startswith(("<", "$$"))
                or following.count("`") % 2 != 0
            ):
                break
            if marker:
                # A list continuation may be lazy (same indentation) or use
                # the list's exact content indentation; deeper content is code.
                quote_indent = (
                    max(0, len(quote.rsplit(">", 1)[-1].expandtabs(4)) - 1)
                    if quote
                    else 0
                )
                next_quote_indent = (
                    max(0, len(next_quote.rsplit(">", 1)[-1].expandtabs(4)) - 1)
                    if next_quote
                    else 0
                )
                allowed = {
                    len(indent.expandtabs(4)) + quote_indent,
                    len((indent + marker).expandtabs(4)) + quote_indent,
                }
                if len(next_indent.expandtabs(4)) + next_quote_indent not in allowed:
                    break
            elif next_indent != indent or next_quote != quote:
                break
            if not marker and len(next_indent.expandtabs(4)) >= 4:
                break
            normalized = normalized.rstrip(" \t") + " " + next_content.lstrip(" \t")
            originals.append(following_raw)
            previous = following
            index += 1
        result.append(
            _LogicalLine("".join(originals), normalized + _ending(originals[-1]))
        )
    return tuple(result)


@dataclass(frozen=True)
class _HTMLToken:
    start: int
    end: int
    kind: str
    visible: str
    block: bool = False


class _HTMLScanner(HTMLParser):
    """Record source spans; never serialize an HTML parse tree."""

    def __init__(self, source: str):
        super().__init__(convert_charrefs=False)
        self.source = source
        self.offsets = [0] + [match.end() for match in re.finditer("\n", source)]
        self.tokens: list[_HTMLToken] = []
        self.stack: list[str] = []

    def source_offset(self) -> int:
        row, column = self.getpos()
        return self.offsets[row - 1] + column

    def add(self, length: int, kind: str, visible: str = "", block: bool = False):
        start = self.source_offset()
        self.tokens.append(_HTMLToken(start, start + length, kind, visible, block))

    def opaque(self) -> bool:
        return any(tag in _OPAQUE_TAGS for tag in self.stack)

    def handle_starttag(self, tag, attrs):
        source_tag = self.get_starttag_text()
        assert source_tag is not None
        self.add(
            len(source_tag),
            tag,
            "\n" if tag in _BLOCK_TAGS else "",
            tag in _BLOCK_TAGS,
        )
        if tag not in _VOID_TAGS:
            self.stack.append(tag)

    def handle_startendtag(self, tag, attrs):
        source_tag = self.get_starttag_text()
        assert source_tag is not None
        self.add(
            len(source_tag),
            tag,
            "\n" if tag in _BLOCK_TAGS else "",
            tag in _BLOCK_TAGS,
        )

    def handle_endtag(self, tag):
        start = self.source_offset()
        end = self.source.find(">", start)
        self.add(
            (end + 1 if end >= 0 else len(self.source)) - start,
            tag,
            "\n" if tag in _BLOCK_TAGS else "",
            tag in _BLOCK_TAGS,
        )
        if tag in self.stack:
            self.stack = self.stack[: len(self.stack) - 1 - self.stack[::-1].index(tag)]

    def handle_data(self, data):
        if self.opaque():
            self.add(len(data), "opaque")

    def handle_entityref(self, name):
        start = self.source_offset()
        length = (
            len(name)
            + 1
            + int(self.source[start + len(name) + 1 : start + len(name) + 2] == ";")
        )
        raw = self.source[start : start + length]
        self.add(length, "entity", "" if self.opaque() else html.unescape(raw))

    def handle_charref(self, name):
        start = self.source_offset()
        length = (
            len(name)
            + 2
            + int(self.source[start + len(name) + 2 : start + len(name) + 3] == ";")
        )
        raw = self.source[start : start + length]
        self.add(length, "entity", "" if self.opaque() else html.unescape(raw))

    def handle_comment(self, data):
        start = self.source_offset()
        match = re.search(r"--\s*>", self.source[start + 4 :])
        self.add(4 + match.end() if match else len(self.source) - start, "comment")

    def handle_decl(self, decl):
        self.add(len(decl) + 3, "declaration")

    def handle_pi(self, data):
        self.add(len(data) + 2, "instruction")

    def unknown_decl(self, data):
        start = self.source_offset()
        end = self.source.find("]>", start)
        self.add((end + 2 if end >= 0 else len(self.source)) - start, "declaration")


def _html_parts(source: str):
    scanner = _HTMLScanner(source)
    try:
        scanner.feed(source)
        scanner.close()
    except (ValueError, AssertionError):
        # Invalid/unsupported HTML is safe to round-trip but never to rewrite.
        return (_HTMLToken(0, len(source), "opaque", ""),)
    if scanner.opaque():
        start = scanner.tokens[-1].end if scanner.tokens else 0
        if start < len(source):
            scanner.tokens.append(_HTMLToken(start, len(source), "opaque", ""))
    previous_end = 0
    for token in scanner.tokens:
        if not previous_end <= token.start <= token.end <= len(source):
            return (_HTMLToken(0, len(source), "opaque", ""),)
        previous_end = token.end
    return tuple(scanner.tokens)


def _markdown_visible(text: str) -> str:
    result = []
    fence: str | None = None
    for raw in text.splitlines():
        line = raw
        match = re.match(r"^[ \t]*(?:>[ \t]*)*(`{3,}|~{3,})", line)
        if match:
            fence = None if fence == match[1][0] else match[1][0]
            continue
        if fence is not None or re.match(r"^\s*\[[^\]]+\]:", line):
            continue
        indent, quote, marker, line = _parts(line)
        if len(indent.expandtabs(4)) >= 4 and not marker:
            continue
        line = re.sub(r"^#{1,6}\s+|\s+#+\s*$", "", line)
        line = re.sub(r"(`+).*?\1", "", line)
        line = re.sub(r"!?\[([^\]]*)\]\([^\n]*?\)", r"\1", line)
        line = re.sub(r"!?\[([^\]]*)\]\[[^\]]*\]", r"\1", line)
        line = re.sub(r"<((?:https?://|mailto:)[^<>]+)>", r"\1", line)
        line = re.sub(r"<(?!hf-)[^<>]*>", "", line)
        line = _MARKERS.sub("", line)
        line = re.sub(r"\\([^\w\s])", r"\1", line)
        result.append(html.unescape(line))
    return "\n".join(result)


def _mask_markdown_fragments(text, protect):
    """Protect inline fragments, including balanced link-destination parentheses."""
    result = []
    position = 0
    while match := _FRAGMENTS.search(text, position):
        start, end = match.span()
        raw = match[0]
        if raw.startswith(("[", "![")) and "](" in raw:
            opening = text.index("](", start) + 1
            depth = 1
            cursor = opening + 1
            while cursor < len(text) and depth:
                if text[cursor] == "\\":
                    cursor += 2
                    continue
                if text[cursor] == "(":
                    depth += 1
                elif text[cursor] == ")":
                    depth -= 1
                cursor += 1
            end = cursor
            raw = text[start:end]
        result.append(text[position:start])
        label = re.match(r"^!?\[([^\]]*)\]\(", raw)
        visible = _markdown_visible(label[1] if label else raw)
        result.append(protect(raw, visible))
        position = end
    result.append(text[position:])
    return "".join(result)


@dataclass(frozen=True)
class FormattedText:
    """A prepared document and a lossless mapping back to its source format."""

    source: str
    format: str
    text: str
    _lines: tuple[_LogicalLine, ...] = ()
    _tokens: tuple[str, ...] = ()
    _raw_tokens: tuple[str, ...] = ()
    _visible_tokens: tuple[str, ...] = ()
    _gaps: tuple[str, ...] = ()
    _editable_gaps: tuple[str, ...] = ()
    _decorations: tuple[tuple[str, str], ...] = ()

    def metadata(self) -> dict:
        return {
            "format": self.format,
            "soft_wraps_normalized": any(
                line.source
                != _PLACEHOLDER.sub(
                    lambda match: dict(
                        zip(self._tokens, self._raw_tokens, strict=True)
                    ).get(match[0], match[0]),
                    line.text,
                )
                for line in self._lines
            ),
            "opaque_fragments": len(self._tokens),
            "restoration": "Unchanged fragments retain exact source; edited HTML text is escaped.",
        }

    def issues(self, edited: str) -> list[str]:
        """Return structural rejection reasons; semantic review is separate."""
        if not isinstance(edited, str):
            raise TypeError("edited must be a string")
        if edited == self.text or self.format == "plain":
            return []
        issues = []
        if re.findall(r"\r\n|\r|\n", edited) != re.findall(r"\r\n|\r|\n", self.text):
            issues.append("line_boundaries_changed")
        if self.format == "html":
            if tuple(_PLACEHOLDER.findall(edited)) != self._tokens:
                return issues + ["protected_html_changed"]
            gaps = _PLACEHOLDER.split(edited)
            if any(re.search(r"<(?:/?[A-Za-z][^<>]*|![^<>]*)>", gap) for gap in gaps):
                issues.append("html_markup_added")
            for old, new, raw, (prefix, suffix) in zip(
                self._editable_gaps, gaps, self._gaps, self._decorations, strict=True
            ):
                if not raw.strip() and new != old:
                    issues.append("protected_html_changed")
                if (prefix and not new.startswith(prefix)) or (
                    suffix and not new.endswith(suffix)
                ):
                    issues.append("html_block_boundaries_changed")
                # Artificial separators cannot migrate to another text node.
                if re.findall(r"\r\n|\r|\n", old) != re.findall(r"\r\n|\r|\n", new):
                    issues.append("html_block_boundaries_changed")
            return list(dict.fromkeys(issues))
        if tuple(_PLACEHOLDER.findall(edited)) != self._tokens:
            issues.append("protected_markup_changed")
        lines = edited.splitlines(keepends=True)
        if len(lines) != len(self._lines):
            return issues + ["line_boundaries_changed"]
        for block, new in zip(self._lines, lines, strict=True):
            if block.protected and new != block.text:
                issues.append("protected_block_changed")
            old_signature = _inline_signature(_body(block.text))
            new_signature = _inline_signature(_body(new))
            for before, after, reason in zip(
                old_signature,
                new_signature,
                (
                    "line_prefix_changed",
                    "emphasis_markers_changed",
                    "protected_markup_changed",
                    "line_ending_format_changed",
                ),
                strict=True,
            ):
                if before != after:
                    issues.append(reason)
        return list(dict.fromkeys(issues))

    def restore(self, edited: str) -> str:
        """Restore a valid candidate, rejecting changed structure before output."""
        issues = self.issues(edited)
        if issues:
            raise ValueError("Formatted text cannot be restored: " + ", ".join(issues))
        if edited == self.text:
            return self.source
        if self.format == "plain":
            return edited
        if self.format != "html":
            return "".join(
                block.source
                if new == block.text
                else _PLACEHOLDER.sub(
                    lambda match: dict(
                        zip(self._tokens, self._raw_tokens, strict=True)
                    )[match[0]],
                    new,
                )
                for block, new in zip(
                    self._lines, edited.splitlines(keepends=True), strict=True
                )
            )
        gaps = _PLACEHOLDER.split(edited)
        result = []
        for index, (new, old, raw, (prefix, suffix)) in enumerate(
            zip(gaps, self._editable_gaps, self._gaps, self._decorations, strict=True)
        ):
            if new == old:
                result.append(raw)
            else:
                content = new[len(prefix) : len(new) - len(suffix) if suffix else None]
                result.append(html.escape(content, quote=False))
            if index < len(self._raw_tokens):
                result.append(self._raw_tokens[index])
        return "".join(result)

    def visible_text(self, edited: str | None = None) -> str:
        """Project readable prose, also for partial detector/chunk inputs.

        This deliberately does not validate structure. Call ``issues`` before
        accepting or restoring a candidate. Markup, opaque code and metadata
        never become detector input; HTML entities become their visible values.
        """
        text = self.text if edited is None or edited == self.source else edited
        if not isinstance(text, str):
            raise TypeError("edited must be a string")
        if self.format == "html":
            visible = dict(zip(self._tokens, self._visible_tokens, strict=True))
            # Remove only sufficiently long suffixes from cut placeholders,
            # before decoding entities such as &gt; into visible punctuation.
            for token in self._tokens:
                lengths = range(min(len(token) - 1, len(text)), 3, -1)
                match_length = next(
                    (length for length in lengths if text.startswith(token[-length:])),
                    None,
                )
                if match_length is not None:
                    text = text[match_length:]
                    break
            text = re.sub(r"<hf-[^>]*$", "", text)
            text = _PLACEHOLDER.sub(lambda match: visible.get(match[0], ""), text)
            text = re.sub(r"[ \t\f\v]+", " ", text)
            return re.sub(r" *\n(?: *\n)+ *", "\n\n", text).strip()
        if self.format == "markdown":
            visible = dict(zip(self._tokens, self._visible_tokens, strict=True))
            text = _markdown_visible(text)
            return _PLACEHOLDER.sub(lambda match: visible.get(match[0], ""), text)
        return text


def prepare_formatted_text(text: str, format: str = "auto") -> FormattedText:
    """Prepare plain text, Markdown or HTML for guarded prose editing."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if format not in {"auto", *_FORMATS}:
        raise ValueError("format must be auto, plain, markdown or html")
    resolved = detect_format(text) if format == "auto" else format
    if resolved != "html":
        lines = _logical_lines(text)
        if resolved == "plain":
            return FormattedText(
                text, resolved, "".join(line.text for line in lines), _lines=lines
            )
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
        while f"<hf-{digest}-" in text:
            digest += "x"
        tokens, raw_tokens, visible_tokens, masked_lines = [], [], [], []

        def protect(raw: str, visible: str = "") -> str:
            token = f"<hf-{digest}-{len(tokens):04d}-md>"
            tokens.append(token)
            raw_tokens.append(raw)
            visible_tokens.append(visible)
            return token

        for line in lines:
            if line.opaque:
                masked = protect(_body(line.text)) + _ending(line.text)
            else:
                masked = _mask_markdown_fragments(line.text, protect)
            masked_lines.append(
                _LogicalLine(line.source, masked, line.protected, line.opaque)
            )
        return FormattedText(
            text,
            resolved,
            "".join(line.text for line in masked_lines),
            _lines=tuple(masked_lines),
            _tokens=tuple(tokens),
            _raw_tokens=tuple(raw_tokens),
            _visible_tokens=tuple(visible_tokens),
        )
    spans = _html_parts(text)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
    while f"<hf-{digest}-" in text:
        digest += "x"
    html_tokens = tuple(
        f"<hf-{digest}-{index:04d}-{span.kind}>" for index, span in enumerate(spans)
    )
    gaps = []
    previous = 0
    for span in spans:
        gaps.append(text[previous : span.start])
        previous = span.end
    gaps.append(text[previous:])
    editable_gaps = []
    decorations = []
    pieces = []
    pending_break = False
    had_content = False
    for index, raw in enumerate(gaps):
        normalized = re.sub(r"[ \t\r\n\f]+", " ", raw)
        prefix = suffix = ""
        if normalized.strip():
            if pending_break and had_content:
                prefix = "\n"
            pending_break = False
            had_content = True
        elif pending_break:
            normalized = ""
        if index < len(spans) and spans[index].visible.strip():
            if pending_break and had_content:
                suffix = "\n"
            had_content = True
            pending_break = False
        editable = prefix + normalized + suffix
        editable_gaps.append(editable)
        decorations.append((prefix, suffix))
        pieces.append(editable)
        if index < len(spans):
            pieces.append(html_tokens[index])
            pending_break = pending_break or spans[index].block
    return FormattedText(
        text,
        resolved,
        "".join(pieces),
        _tokens=html_tokens,
        _raw_tokens=tuple(text[span.start : span.end] for span in spans),
        _visible_tokens=tuple(span.visible for span in spans),
        _gaps=tuple(gaps),
        _editable_gaps=tuple(editable_gaps),
        _decorations=tuple(decorations),
    )
