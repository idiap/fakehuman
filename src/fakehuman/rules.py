# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: William Droz <william.droz@idiap.ch>
# SPDX-License-Identifier: GPL-3.0-only

"""Small, inspectable text transformations for detector robustness experiments.

These rules are experiments, not evidence that a text was written by a human.
``quality_issues`` is a deliberately conservative mechanical screen; it cannot
prove preservation of meaning, factual accuracy, or acceptable writing quality.
"""

from __future__ import annotations

import hashlib
import random
import re
from collections import Counter
from collections.abc import Callable, Sequence

Rule = Callable[[str, random.Random], str]

# Keep source literals and markup outside prose edits. Quoted wording is also
# protected; normalize_quotes alone may normalize its typographic delimiters.
_PROTECTED = re.compile(
    r"(?m:^[ \t]{0,3}(?:>[ \t]*)*(?P<fence>`{3,}|~{3,})[^\r\n]*(?:\r?\n|\r)"
    r"[\s\S]*?(?:^[ \t]{0,3}(?:>[ \t]*)*(?P=fence)(?:(?<=`)`*|(?<=~)~*)"
    r"[ \t]*(?:\r?\n|\r|$)|\Z))"
    r"|(?P<ticks>`+)(?!`)[\s\S]+?(?<!`)(?P=ticks)(?!`)"
    r"|(?m:^(?: {4}| {0,3}\t)[^\r\n]*(?:\r?\n|\r|$))"
    r"|(?m:^[ \t]{0,3}>[^\r\n]*(?:\r?\n|\r|$))"
    r"|<!--(?s:.*?)(?:-->|\Z)"
    r"|<(?P<opaque>pre|code|script|style|textarea|svg|math|template)\b[^>]*>"
    r"(?s:.*?)(?:</(?P=opaque)\s*>|\Z)"
    r"|</?[A-Za-z][A-Za-z0-9:-]*(?:\s+(?:[^<>\"']|\"[^\"]*\"|'[^']*')*)?/?>"
    r"|&(?:\#[0-9]+|\#x[0-9a-f]+|[A-Za-z][A-Za-z0-9]+);"
    r"|(?m:^[ \t]{0,3}\[[^\]\r\n]+\]:[^\r\n]*)"
    r"|!?\[[^\]\r\n]*\]\((?:[^()\r\n]|\([^()\r\n]*\))*\)"
    r"|!?\[[^\]\r\n]*\]\[[^\]\r\n]*\]"
    r"|https?://[^\s<>]+|www\.[^\s<>]+"
    r"|\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b"
    r"|\b10\.\d{4,9}/[^\s<>]+"
    r"|\[[^\]\n]+\]"
    r"|\([^()\n]*(?:\b(?:19|20)\d{2}[a-z]?\b|et al\.|doi:)[^()\n]*\)"
    r'''|(?P<quoted>"(?:\\.|[^"\\])*"|“[^”]*”|‘[^’]*’|(?<!\w)'(?:\\.|[^'\\])*'(?!\w))''',
    re.IGNORECASE,
)
_PLAIN_QUOTES = str.maketrans("“”‘’", "\"\"''")
_WORD = re.compile(r"\b[\w]+(?:['’][\w]+)*\b")
_NUMBER_SOURCE = r"(?<![\w.])[+\-−]?\d+(?:[.,]\d+)*(?:[eE][+\-−]?\d+)?"
_NUMBERS = re.compile(_NUMBER_SOURCE)
_UNIT_SOURCE = (
    r"(?:mmHg|mmol|µmol|μmol|mol|kHz|Hz|kPa|Pa|IU|"
    r"µg|μg|mg|ng|kg|g|µM|μM|mM|nM|M|µm|μm|nm|mm|cm|"
    r"mL|ml|µL|μL|L|l|mV|µV|μV|V|ms|min|s|h|°C|°F|K|m|%)"
    r"(?:[²³⁻¹]+|[−-]\d+)?"
)
_QUANTITIES = re.compile(
    _NUMBER_SOURCE
    + r"[ \t]*(?P<unit>"
    + _UNIT_SOURCE
    + r"(?:/"
    + _UNIT_SOURCE
    + r")?)(?![A-Za-zµμ])"
)
_SYMBOLS = re.compile(r"[µμ±≤≥<>≈≠×÷−°αβγδ²³⁻¹₀₁₂₃₄₅₆₇₈₉]")
_SCIENTIFIC_TOKENS = re.compile(
    r"\b(?:[A-Z][A-Z0-9]+(?:[-/][A-Za-z0-9]+)*"
    r"|[A-Za-z]+[-/]\d[\w-]*|[A-Za-z]*\d+[A-Za-z]+[\w-]*"
    r"|[A-Za-z]+\d[\w-]*|[a-z]+[A-Z][A-Za-z0-9]*)\b"
)
_NEGATIONS = re.compile(
    r"\b(?:not|never|no|neither|nor|without|cannot)\b|\b\w+n['’]t\b",
    re.IGNORECASE,
)
_HEDGES = re.compile(
    r"\b(?:may|might|could|possibly|probably|approximately|uncertain|"
    r"uncertainty|potentially)\b",
    re.IGNORECASE,
)


def _outside_protected(
    text: str, transform: Callable[[str], str], *, normalize_quoted: bool = False
) -> str:
    parts: list[str] = []
    start = 0

    def prose(chunk: str, offset: int) -> str:
        # A protected span is not a sentence boundary. A sentinel suppresses
        # synthetic ^ matches while preserving real newline anchors in chunk.
        if offset == 0:
            return transform(chunk)
        prefix = "\n" if text[offset - 1] in "\r\n" else "\ue000"
        return transform(prefix + chunk)[1:]

    for match in _PROTECTED.finditer(text):
        protected = match.group()
        if normalize_quoted and match.lastgroup == "quoted":
            protected = protected.translate(_PLAIN_QUOTES)
        parts.extend((prose(text[start : match.start()], start), protected))
        start = match.end()
    parts.append(prose(text[start:], start))
    return "".join(parts)


def _case_like(original: str, replacement: str) -> str:
    if original.isupper():
        return replacement.upper()
    if original[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def _replace_phrases(text: str, phrases: dict[str, str]) -> str:
    pattern = re.compile(
        r"(?<![\w'’/\\-])(?:"
        + "|".join(re.escape(p) for p in sorted(phrases, key=len, reverse=True))
        + r")(?![\w'’/\\-])",
        re.IGNORECASE,
    )
    return _outside_protected(
        text,
        lambda chunk: pattern.sub(
            lambda match: _case_like(match.group(), phrases[match.group().lower()]),
            chunk,
        ),
    )


def _capitalize_word(word: str) -> str:
    # Preserve single-letter variables, mixed-case names, and numbered identifiers.
    return (
        word.capitalize()
        if word.isalpha() and word.islower() and len(word) > 1
        else word
    )


def normalize_dashes(text: str, rng: random.Random) -> str:
    """Replace prose em dashes, retaining numeric ranges and minus signs."""
    def transform(chunk: str) -> str:
        def replace(match: re.Match[str]) -> str:
            before = match[1]
            after = re.match(r"[A-Za-z]+", chunk[match.end():])[0]
            if not all(word.islower() or word.istitle() for word in (before, after)):
                return match.group()
            return before + ", "

        return re.sub(
            r"\b([A-Za-z]{2,})[ \t]*—[ \t]*(?=[A-Za-z]{2,}\b)", replace, chunk
        )

    return _outside_protected(
        text, transform
    )


def normalize_quotes(text: str, rng: random.Random) -> str:
    """Normalize typographic quotation marks, not scientific Unicode symbols."""
    return _outside_protected(
        text, lambda chunk: chunk.translate(_PLAIN_QUOTES), normalize_quoted=True
    )


def normalize_spaces(text: str, rng: random.Random) -> str:
    """Collapse interior prose spaces; preserve indentation, tabs and hard breaks."""
    return _outside_protected(
        text, lambda chunk: re.sub(r"(?<=\S) {2,}(?=\S)", " ", chunk)
    )


def trim_openers(text: str, rng: random.Random) -> str:
    pattern = re.compile(
        r"(^|[.!?]\s+)(?:Furthermore|Moreover|Additionally|In conclusion|"
        r"In summary|Overall|Importantly|Notably),[ \t]+([A-Za-z][\w-]*)",
        re.IGNORECASE | re.MULTILINE,
    )
    return _outside_protected(
        text,
        lambda chunk: pattern.sub(
            lambda match: match[1] + _capitalize_word(match[2] or ""), chunk
        ),
    )


def remove_stock_framing(text: str, rng: random.Random) -> str:
    pattern = re.compile(
        r"(^|[.!?]\s+)(?:It is important to note that|It is worth noting that|"
        r"It should be noted that|It is essential to recognize that)[ \t]+([A-Za-z][\w-]*)",
        re.IGNORECASE | re.MULTILINE,
    )
    return _outside_protected(
        text,
        lambda chunk: pattern.sub(
            lambda match: match[1] + _capitalize_word(match[2] or ""), chunk
        ),
    )


def simplify_verbs(text: str, rng: random.Random) -> str:
    return _replace_phrases(
        text,
        {
            "utilize": "use",
            "utilizes": "uses",
            "utilized": "used",
            "utilizing": "using",
            "commence": "start",
            "commences": "starts",
            "commenced": "started",
            "endeavor to": "try to",
            "endeavors to": "tries to",
            "endeavored to": "tried to",
        },
    )


def simplify_connectors(text: str, rng: random.Random) -> str:
    return _replace_phrases(
        text,
        {
            "due to the fact that": "because",
            "in order to": "to",
            "with regard to": "about",
            "in the event that": "if",
            "despite the fact that": "although",
            "prior to": "before",
            "subsequent to": "after",
        },
    )


def shorten_phrases(text: str, rng: random.Random) -> str:
    return _replace_phrases(
        text,
        {
            "at the present time": "now",
            "at this point in time": "now",
            "for the purpose of": "for",
            "on a regular basis": "regularly",
            "in a timely manner": "promptly",
            "in the majority of cases": "in most cases",
            "a large number of": "many",
            "a wide range of": "many",
        },
    )


def active_phrases(text: str, rng: random.Random) -> str:
    return _replace_phrases(
        text,
        {
            "conduct an analysis of": "analyze",
            "conducted an analysis of": "analyzed",
            "perform an assessment of": "assess",
            "performed an assessment of": "assessed",
            "provide an explanation of": "explain",
            "provided an explanation of": "explained",
            "make a comparison between": "compare",
            "made a comparison between": "compared",
            "carry out an evaluation of": "evaluate",
            "carried out an evaluation of": "evaluated",
        },
    )


_CONTRACTIONS = {
    "is not": "isn't",
    "are not": "aren't",
    "was not": "wasn't",
    "were not": "weren't",
    "do not": "don't",
    "does not": "doesn't",
    "did not": "didn't",
    "cannot": "can't",
    "will not": "won't",
    "has not": "hasn't",
    "have not": "haven't",
    "should not": "shouldn't",
    "could not": "couldn't",
    "would not": "wouldn't",
}


def contractions(text: str, rng: random.Random) -> str:
    return _replace_phrases(text, _CONTRACTIONS)


def expand_contractions(text: str, rng: random.Random) -> str:
    phrases = {value: key for key, value in _CONTRACTIONS.items()}
    phrases.update(
        {value.replace("'", "’"): key for key, value in _CONTRACTIONS.items()}
    )
    return _replace_phrases(text, phrases)


def split_semicolons(text: str, rng: random.Random) -> str:
    # A semicolon can separate list items rather than complete clauses. Require
    # an explicit subject and a recognizable finite verb on its right side.
    finite = (
        r"(?:am|is|are|was|were|has|have|had|do|does|did|can|could|may|might|"
        r"must|shall|should|will|would|began|ended|remained|continued|worked|"
        r"changed|increased|decreased|failed|succeeded|started|stopped|waited)"
    )
    clause = re.compile(
        r"(?:the |a |an |these |those |this |that )?"
        r"(?!and\b|or\b|but\b)[A-Za-z][\w-]*[ \t]+" + finite + r"\b",
        re.IGNORECASE,
    )

    def transform(chunk: str) -> str:
        return re.sub(
            r"(?<=[A-Za-z]);[ \t]+([A-Za-z][\w-]*)",
            lambda match: ". " + _capitalize_word(match[1])
            if clause.match(chunk, match.start(1))
            else match.group(),
            chunk,
        )

    return _outside_protected(
        text, transform
    )


def strip_heading_markup(text: str, rng: random.Random) -> str:
    def transform(chunk: str) -> str:
        return re.sub(
            r"(?m)^#{1,6}[ \t]+([^\r\n]*)",
            lambda match: re.sub(r"[ \t]+#+[ \t]*$", "", match[1]),
            chunk,
        )

    return _outside_protected(
        text, transform
    )


def seeded_synonyms(text: str, rng: random.Random) -> str:
    """A tiny explicit vocabulary: never unrestricted thesaurus substitution."""
    choices = {
        "additionally": ("also", "in addition"),
        "furthermore": ("also", "in addition"),
        "helpful": ("useful", "helpful"),
        "numerous": ("many", "numerous"),
        "therefore": ("thus", "therefore"),
        "however": ("however", "nevertheless"),
        "consequently": ("as a result", "consequently"),
    }
    pattern = re.compile(
        r"(?<![\w'’/\\-])(?:" + "|".join(choices) + r")(?![\w'’/\\-])",
        re.IGNORECASE,
    )

    def transform(chunk: str) -> str:
        def replace(match: re.Match[str]) -> str:
            if match.group().lower() == "however" and not re.match(
                r",[ \t]+", chunk[match.end():]
            ):
                return match.group()
            return _case_like(match.group(), rng.choice(choices[match.group().lower()]))

        return pattern.sub(replace, chunk)

    return _outside_protected(
        text, transform
    )


def simple_list_transitions(text: str, rng: random.Random) -> str:
    return _replace_phrases(
        text,
        {
            "firstly": "first",
            "secondly": "second",
            "thirdly": "third",
            "lastly": "last",
        },
    )


def affirmative_contractions(text: str, rng: random.Random) -> str:
    """Contract an explicit pronoun + auxiliary only before a following word."""
    phrases = {
        "it is": "it's",
        "that is": "that's",
        "there is": "there's",
        "we are": "we're",
        "they are": "they're",
        "you are": "you're",
        "i am": "I'm",
        "we will": "we'll",
        "they will": "they'll",
        "you will": "you'll",
        "i will": "I'll",
    }
    pattern = re.compile(
        r"(?<![\w'’/\\-])(?:" + "|".join(phrases)
        + r")(?![\w'’/\\-])(?=[ \t]+[A-Za-z])",
        re.IGNORECASE,
    )

    def transform(chunk: str) -> str:
        def replace(match: re.Match[str]) -> str:
            # Relative/comparative auxiliaries can be stressed or stand alone:
            # "where we are now" cannot become "where we're now".
            if re.search(r"\b(?:what|where|how|as|than)[ \t]+$", chunk[:match.start()], re.I):
                return match.group()
            return _case_like(match.group(), phrases[match.group().lower()])

        return pattern.sub(replace, chunk)

    return _outside_protected(
        text, transform
    )


def auxiliary_have_contractions(text: str, rng: random.Random) -> str:
    """Leave possession alone; contract have only before listed participles."""
    pattern = re.compile(
        r"\b(we|they|you|I) have (?=(?:been|seen|observed|measured|found|used|"
        r"tested|checked|reviewed|written|read|collected|recorded|discussed|"
        r"prepared|completed|learned|learnt|noticed|identified|considered)\b)",
        re.IGNORECASE,
    )
    return _outside_protected(
        text,
        lambda chunk: pattern.sub(
            lambda match: _case_like(match.group(), match[1] + "'ve "), chunk
        ),
    )


def common_plain_verbs(text: str, rng: random.Random) -> str:
    """Use ordinary verbs; preserve conditional 'provided that' and passives."""
    phrases = {
        "require": "need",
        "requires": "needs",
        "required": "needed",
        "requiring": "needing",
        "provide": "give",
        "provides": "gives",
        "providing": "giving",
        "assist": "help",
        "assists": "helps",
        "assisted": "helped",
        "assisting": "helping",
    }
    pattern = re.compile(
        r"(?<![\w'’/\\-])(?:" + "|".join(phrases)
        + r")(?![\w'’/\\-])(?!\s+(?:that|for|with)\b)",
        re.IGNORECASE,
    )

    def transform(chunk: str) -> str:
        def replace(match: re.Match[str]) -> str:
            word = match.group().lower()
            following = chunk[match.end():]
            if word == "required" and re.match(r"\s+(?:to|of|by)\b", following, re.I):
                return match.group()
            if word.startswith("provid"):
                if word == "providing" and not re.search(
                    r"\b(?:am|is|are|was|were|be|been)[ \t]+$", chunk[:match.start()], re.I
                ):
                    return match.group()
                if re.match(r"(?:[ \t]+[\w'’-]+){1,6}[ \t]+with\b", following, re.I):
                    return match.group()
            return _case_like(match.group(), phrases[word])

        chunk = pattern.sub(replace, chunk)
        # Past-tense "provided" needs "gave"; the passive needs "given".
        # Only rewrite the explicitly active pronoun construction here.
        return re.sub(
            r"\b(we|they|you|I) provided\b(?!\s+(?:that|for|with)\b)"
            r"(?!(?:[ \t]+[\w'’-]+){1,6}[ \t]+with\b)",
            lambda match: _case_like(match.group(), match[1] + " gave"),
            chunk,
            flags=re.IGNORECASE,
        )

    return _outside_protected(text, transform)


def example_alternatives(text: str, rng: random.Random) -> str:
    return _replace_phrases(
        text,
        {"for example": "for instance", "as an example": "as an illustration"},
    )


def contrast_transitions(text: str, rng: random.Random) -> str:
    """Reword initial contrasts without repeating the target in the same clause."""
    pattern = re.compile(
        r"(^|[.!?;]\s+)(However|Nevertheless|Nonetheless),[ \t]+",
        re.MULTILINE | re.IGNORECASE,
    )
    def transform(chunk: str) -> str:
        def replace(match: re.Match[str]) -> str:
            replacement = "still" if match[2].lower() == "however" else "even so"
            clause = re.split(r"[.!?;\r\n]", chunk[match.end():], maxsplit=1)[0]
            repeated = r"\bstill\b" if replacement == "still" else r"\beven[ \t]+so\b"
            if re.search(repeated, clause, flags=re.IGNORECASE):
                return match.group()
            return match[1] + _case_like(match[2], replacement) + ", "

        return pattern.sub(replace, chunk)

    return _outside_protected(text, transform)


def result_transitions(text: str, rng: random.Random) -> str:
    """Preserve the causal link without creating 'they so did' constructions."""
    pattern = re.compile(
        r"(^|[.!?;]\s+)(Therefore|Consequently|As a result),[ \t]+",
        re.MULTILINE | re.IGNORECASE,
    )
    return _outside_protected(
        text,
        lambda chunk: pattern.sub(
            lambda match: match[1] + _case_like(match[2], "so") + " ", chunk
        ),
    )


def two_sentence_paragraphs(text: str, rng: random.Random) -> str:
    """Insert paragraph breaks only; skip decimal points and common abbreviations.

    This is a presentation experiment. A detector that normalizes whitespace
    may produce exactly the same score, which is a valid no-effect result.
    """
    boundary = re.compile(r"([.!?])([ \t]+)(?=[A-Z][a-z])")
    abbreviations = {"dr", "mr", "mrs", "ms", "prof", "fig", "eq", "vs", "etc", "al"}
    count = 0

    def transform(chunk: str) -> str:
        nonlocal count

        def paragraph(match: re.Match[str]) -> str:
            nonlocal count
            preceding = re.search(r"([A-Za-z]+)$", chunk[: match.start()])
            if match[1] == "." and preceding:
                word = preceding[1].lower()
                if len(word) == 1 or word in abbreviations:
                    return match.group()
            count += 1
            return match[1] + ("\n\n" if count % 2 == 0 else match[2])

        return boundary.sub(paragraph, chunk)

    return _outside_protected(text, transform)


def ability_phrases(text: str, rng: random.Random) -> str:
    """Shorten present ability; retain past tense and modal qualifications."""
    pattern = re.compile(
        r"\b(?:is able to|are able to|am able to|has the ability to|have the ability to)\b",
        re.IGNORECASE,
    )

    def transform(chunk: str) -> str:
        def replace(match: re.Match[str]) -> str:
            preceding = re.split(r"[.!?;\n]", chunk[: match.start()])[-1]
            if re.search(
                r"(?:\b(?:may|might|could|would|should|must|can|cannot|not|to|will|do|does|did)"
                r"|n['’]t)\b",
                preceding,
                flags=re.IGNORECASE,
            ):
                return match.group()
            return _case_like(match.group(), "can")

        return pattern.sub(replace, chunk)

    return _outside_protected(text, transform)


def recommendation_phrases(text: str, rng: random.Random) -> str:
    """Reword explicit advice while leaving should/must and negation intact."""
    return _replace_phrases(
        text,
        {
            "a useful recommendation is to": "one useful suggestion is to",
            "my recommendation is to": "my suggestion is to",
            "our recommendation is to": "our suggestion is to",
            "it is advisable to": "it is a good idea to",
        },
    )


def shorter_time_phrases(text: str, rng: random.Random) -> str:
    return _replace_phrases(
        text,
        {
            "in advance of": "before",
            "in advance": "beforehand",
            "at a later date": "later",
            "during the course of": "during",
            "before the start of": "before",
            "after the end of": "after",
            "on a daily basis": "daily",
            "on a weekly basis": "weekly",
        },
    )


def direct_nominal_phrases(text: str, rng: random.Random) -> str:
    return _replace_phrases(
        text,
        {
            "make a decision": "decide",
            "makes a decision": "decides",
            "made a decision": "decided",
            "making a decision": "deciding",
            "have a discussion about": "discuss",
            "had a discussion about": "discussed",
            "give consideration to": "consider",
            "gave consideration to": "considered",
            "keep a record of": "record",
            "kept a record of": "recorded",
            "conduct a review of": "review",
            "conducted a review of": "reviewed",
        },
    )


def broader_seeded_synonyms(text: str, rng: random.Random) -> str:
    """Phrase-level alternatives retain articles and avoid technical synonyms."""
    choices = {
        "a useful": ("a helpful", "a useful"),
        "an important": ("a key", "an important"),
        "important step": ("key step", "important step"),
        "useful example": ("helpful example", "useful example"),
        "brief note": ("short note", "brief note"),
        "a difficult task": ("a hard task", "a difficult task"),
        "routine task": ("routine job", "routine task"),
        "additional information": ("more information", "additional information"),
        "obtain feedback": ("get feedback", "obtain feedback"),
        "ensure that": ("make sure that", "ensure that"),
    }
    pattern = re.compile(
        r"(?<![\w'’/\\-])(?:"
        + "|".join(re.escape(phrase) for phrase in choices)
        + r")(?![\w'’/\\-])",
        re.IGNORECASE,
    )
    return _outside_protected(
        text,
        lambda chunk: pattern.sub(
            lambda match: _case_like(
                match.group(), rng.choice(choices[match.group().lower()])
            ),
            chunk,
        ),
    )


RULES: dict[str, Rule] = {
    "normalize_dashes": normalize_dashes,
    "normalize_quotes": normalize_quotes,
    "normalize_spaces": normalize_spaces,
    "trim_openers": trim_openers,
    "remove_stock_framing": remove_stock_framing,
    "simplify_verbs": simplify_verbs,
    "simplify_connectors": simplify_connectors,
    "shorten_phrases": shorten_phrases,
    "active_phrases": active_phrases,
    "contractions": contractions,
    "expand_contractions": expand_contractions,
    "split_semicolons": split_semicolons,
    "strip_heading_markup": strip_heading_markup,
    "seeded_synonyms": seeded_synonyms,
    "simple_list_transitions": simple_list_transitions,
    "affirmative_contractions": affirmative_contractions,
    "auxiliary_have_contractions": auxiliary_have_contractions,
    "common_plain_verbs": common_plain_verbs,
    "example_alternatives": example_alternatives,
    "contrast_transitions": contrast_transitions,
    "result_transitions": result_transitions,
    "two_sentence_paragraphs": two_sentence_paragraphs,
    "ability_phrases": ability_phrases,
    "recommendation_phrases": recommendation_phrases,
    "shorter_time_phrases": shorter_time_phrases,
    "direct_nominal_phrases": direct_nominal_phrases,
    "broader_seeded_synonyms": broader_seeded_synonyms,
}

RULE_LABELS: dict[str, str] = {
    "normalize_dashes": "Replace prose em dashes",
    "normalize_quotes": "Use plain quotation marks",
    "normalize_spaces": "Normalize repeated spaces",
    "trim_openers": "Remove stock transitions",
    "remove_stock_framing": "Remove stock framing",
    "simplify_verbs": "Use simpler verbs",
    "simplify_connectors": "Use shorter connectors",
    "shorten_phrases": "Shorten wordy phrases",
    "active_phrases": "Use direct verb phrases",
    "contractions": "Contract negative auxiliaries",
    "expand_contractions": "Expand negative contractions",
    "split_semicolons": "Split semicolon clauses",
    "strip_heading_markup": "Remove heading markup",
    "seeded_synonyms": "Try seeded synonym choices",
    "simple_list_transitions": "Simplify list transitions",
    "affirmative_contractions": "Contract affirmative auxiliaries",
    "auxiliary_have_contractions": "Contract auxiliary have",
    "common_plain_verbs": "Use common plain verbs",
    "example_alternatives": "Reword example phrases",
    "contrast_transitions": "Reword contrast transitions",
    "result_transitions": "Reword result transitions",
    "two_sentence_paragraphs": "Add two-sentence paragraphs",
    "ability_phrases": "Shorten present ability phrases",
    "recommendation_phrases": "Reword recommendation phrases",
    "shorter_time_phrases": "Shorten time phrases",
    "direct_nominal_phrases": "Use direct everyday verbs",
    "broader_seeded_synonyms": "Try phrase-level seeded synonyms",
}


def apply_rules(text: str, rules: Sequence[str], seed: int = 42) -> str:
    """Apply a named pipeline with independent, stable per-text/per-rule RNGs.

    The seed depends on the *original* text, so adding an earlier rule or changing
    batch order does not itself change the random stream of a later rule.
    This function does not silently accept or reject quality: call the screen
    separately and inspect the original and transformed text.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    unknown = set(rules).difference(RULES)
    if unknown:
        raise ValueError("Unknown rule(s): " + ", ".join(sorted(unknown)))
    result = text
    for name in rules:
        digest = hashlib.sha256(f"{seed}\0{name}\0{text}".encode("utf-8")).digest()
        rng = random.Random(int.from_bytes(digest, "big"))
        result = RULES[name](result, rng)
    return result


def _negation_counts(text: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    for match in _NEGATIONS.finditer(text):
        token = match.group().lower().replace("’", "'")
        counts["not" if token == "cannot" or token.endswith("n't") else token] += 1
    return counts


def _hedge_counts(text: str) -> Counter[str]:
    # "Couldn't" carries the same modal and negation as "could not".
    expanded = re.sub(r"\bcouldn['’]t\b", "could not", text, flags=re.IGNORECASE)
    return Counter(token.lower() for token in _HEDGES.findall(expanded))


def quality_issues(original: str, transformed: str) -> list[str]:
    """Flag mechanical damage; an empty list is NOT semantic verification.

    Numeric spelling/order, known measurement units, scientific symbols,
    acronyms, citations, URLs, explicit negations, and common hedges are checked.
    Unknown units, negation scope, changed implications, and factual errors can
    still escape this screen. A human must compare before/after text.
    """
    issues: list[str] = []
    if not transformed.strip():
        issues.append("empty output")
    original_words = len(_WORD.findall(original))
    output_words = len(_WORD.findall(transformed))
    if original_words and output_words < original_words * 0.65:
        issues.append("excessive removal: fewer than 65% of original words remain")
    if original_words and output_words > original_words * 1.35:
        issues.append("excessive expansion: more than 135% of original words")
    if _NUMBERS.findall(original) != _NUMBERS.findall(transformed):
        issues.append("numeric values or their order changed")
    original_units = [match["unit"] for match in _QUANTITIES.finditer(original)]
    output_units = [match["unit"] for match in _QUANTITIES.finditer(transformed)]
    if original_units != output_units:
        issues.append("measurement units changed")
    if Counter(_SYMBOLS.findall(original)) != Counter(_SYMBOLS.findall(transformed)):
        issues.append("scientific symbols changed")
    if Counter(_SCIENTIFIC_TOKENS.findall(original)) != Counter(
        _SCIENTIFIC_TOKENS.findall(transformed)
    ):
        issues.append("protected scientific tokens or acronyms changed")
    def protected_values(text: str) -> list[str]:
        return [
            match.group().translate(_PLAIN_QUOTES)
            if match.lastgroup == "quoted"
            else match.group()
            for match in _PROTECTED.finditer(text)
        ]

    if protected_values(original) != protected_values(transformed):
        issues.append("citation, URL, DOI, code, markup, or quoted wording changed")
    if _negation_counts(original) != _negation_counts(transformed):
        issues.append("negation changed")
    if _hedge_counts(original) != _hedge_counts(transformed):
        issues.append("uncertainty language changed")
    return issues
