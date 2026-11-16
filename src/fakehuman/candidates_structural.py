# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: William Droz <william.droz@idiap.ch>
# SPDX-License-Identifier: GPL-3.0-only

"""Conservative, deterministic alternatives for text post-processing.

These Python post-processing rules make small style or layout edits. They do
not call a language model. Phrase substitutions are intentionally explicit;
they cannot establish semantic equivalence for every context, so outputs still
need human review.
"""

from __future__ import annotations

import random
import re

from .candidates_lexical import _QUANTITY, _candidate
from .rules import Rule, _case_like, _outside_protected


def _phrase_rule(phrases: dict[str, str]) -> Rule:
    replacements = [_candidate(source, target) for source, target in phrases.items()]

    def transform(text: str, rng: random.Random) -> str:
        for replace in replacements:
            text = replace(text, rng)
        return text

    return transform


def _replace_sentence_initial(old: str, new: str) -> Rule:
    pattern = re.compile(
        r"(^|[.!?]\s+)" + re.escape(old) + r"(?=,\s)",
        re.IGNORECASE | re.MULTILINE,
    )

    def transform(text: str, rng: random.Random) -> str:
        return _outside_protected(
            text,
            lambda chunk: pattern.sub(
                lambda m: m.group(1) + _case_like(m.group(0)[len(m.group(1)) :], new),
                chunk,
            ),
        )

    return transform


def _prose_parenthetical(text: str, *, commas: bool) -> str:
    """Change only simple mid-sentence prose asides, not notation or nesting."""
    pattern = re.compile(r"(?<=\w)[ \t]+\(([^()\n]{1,100})\)(?=[ \t]+[A-Za-z])")

    def replace_chunk(chunk: str) -> str:
        def replace(match: re.Match[str]) -> str:
            aside = match[1]
            before = chunk[: match.start()]
            if (
                before.count("(") != before.count(")")
                or not re.match(r"[a-z]", aside)
                or re.search(r"[,;!?=+*/<>^|\\]|(?<!\d)\.|\.(?!\d)", aside)
                or (
                    " " not in aside
                    and aside
                    not in {
                        "briefly",
                        "temporarily",
                        "initially",
                        "originally",
                        "apparently",
                    }
                )
            ):
                return match.group()
            return f", {aside}," if commas else f" — {aside} —"

        return pattern.sub(replace, chunk)

    return _outside_protected(text, replace_chunk)


def parenthetical_to_dashes(text: str, rng: random.Random) -> str:
    """Change a simple mid-sentence prose aside to em dashes."""
    return _prose_parenthetical(text, commas=False)


def parenthetical_to_commas(text: str, rng: random.Random) -> str:
    """Use commas around a simple parenthetical aside, retaining its wording."""
    return _prose_parenthetical(text, commas=True)


def comma_parenthetical_to_dashes(text: str, rng: random.Random) -> str:
    """Change a short comma-delimited aside to em dashes without editing words."""
    return _outside_protected(
        text,
        lambda chunk: re.sub(
            r", ((?:still|which|who|although|though|however|for example|for instance|"
            r"in fact|as expected|as noted)\b[^,;.!?\n]{0,60}),(?=[ \t]+[A-Za-z])",
            r" — \1 —",
            chunk,
            flags=re.IGNORECASE,
        ),
    )


def e_g_expansion(text: str, rng: random.Random) -> str:
    return _phrase_rule({"e.g.": "for example"})(text, rng)


def i_e_expansion(text: str, rng: random.Random) -> str:
    return _phrase_rule({"i.e.": "that is"})(text, rng)


def viz_expansion(text: str, rng: random.Random) -> str:
    return _phrase_rule({"viz.": "namely"})(text, rng)


def that_is_to_say(text: str, rng: random.Random) -> str:
    return _phrase_rule({"that is to say": "in other words"})(text, rng)


def put_another_way(text: str, rng: random.Random) -> str:
    return _phrase_rule({"put another way": "in other words"})(text, rng)


def as_a_consequence(text: str, rng: random.Random) -> str:
    return _phrase_rule({"as a consequence": "as a result"})(text, rng)


def as_a_result_of(text: str, rng: random.Random) -> str:
    return _phrase_rule({"as a result of": "because of"})(text, rng)


def due_to_equivalent(text: str, rng: random.Random) -> str:
    """Replace only the causal preposition phrase, preserving its complement."""
    return _phrase_rule({"due to": "because of"})(text, rng)


def for_the_reason_that(text: str, rng: random.Random) -> str:
    return _phrase_rule({"for the reason that": "because"})(text, rng)


def except_for_equivalent(text: str, rng: random.Random) -> str:
    return _phrase_rule({"with the exception of": "except for"})(text, rng)


def in_addition_to_equivalent(text: str, rng: random.Random) -> str:
    return _phrase_rule({"in addition to": "besides"})(text, rng)


def along_with_equivalent(text: str, rng: random.Random) -> str:
    return _phrase_rule({"along with": "together with"})(text, rng)


def in_other_words_to_that_is(text: str, rng: random.Random) -> str:
    return _phrase_rule({"in other words": "that is"})(text, rng)


def in_particular_to_specifically(text: str, rng: random.Random) -> str:
    return _phrase_rule({"in particular": "specifically"})(text, rng)


def more_specifically_to_specifically(text: str, rng: random.Random) -> str:
    # Retain the comparative qualification instead of silently dropping 'more'.
    return _phrase_rule({"more specifically": "more precisely"})(text, rng)


def roughly_to_about(text: str, rng: random.Random) -> str:
    pattern = re.compile(r"(?<![\w'’/\\\-])roughly(?![\w'’/\\\-])", re.I)
    return _outside_protected(
        text,
        lambda chunk: pattern.sub(
            lambda match: (
                _case_like(match.group(), "about")
                if _QUANTITY.match(chunk[match.end() :])
                else match.group()
            ),
            chunk,
        ),
    )


def in_a_similar_way_to_similarly(text: str, rng: random.Random) -> str:
    return _phrase_rule({"in a similar way": "similarly"})(text, rng)


def as_long_as_to_provided_that(text: str, rng: random.Random) -> str:
    """Reword clear conditions while skipping length and duration comparisons."""
    pattern = re.compile(r"(?<![\w'’/\\\-])as long as(?![\w'’/\\\-])", re.I)

    def replace_chunk(chunk: str) -> str:
        def replace(match: re.Match[str]) -> str:
            before, after = chunk[: match.start()], chunk[match.end() :]
            clause = re.split(r"[.!?;\n]", before)[-1]
            if (
                re.search(
                    r"\b(?:lasts?|lasted|waits?|waited|stays?|stayed|takes?|took|"
                    r"spends?|spent|twice|half|nearly|almost|exactly)\b",
                    clause,
                    re.I,
                )
                or not re.search(
                    r"\b(?:can|could|may|might|will|would|should|must|"
                    r"works?|runs?|succeeds?|applies?|valid|available|"
                    r"acceptable|allowed|possible)\b",
                    clause,
                    re.I,
                )
                or not re.match(
                    r"^[ \t]+(?:[A-Za-z][\w'’-]*[ \t]+){1,5}"
                    r"(?:is|are|remains?|stays?|has|have|does|do|can|will)\b",
                    after,
                    re.I,
                )
            ):
                return match.group()
            return _case_like(match.group(), "provided that")

        return pattern.sub(replace, chunk)

    return _outside_protected(text, replace_chunk)


def provided_that_to_as_long_as(text: str, rng: random.Random) -> str:
    """Reword only sentence-initial provided-that conditions."""
    pattern = re.compile(
        r"(^|[.!?]\s+)(provided that)(?=[ \t]+(?:[A-Za-z][\w'’-]*[ \t]+){1,5}"
        r"(?:is|are|remains?|stays?|has|have|does|do|can|will)\b)",
        re.I | re.M,
    )
    return _outside_protected(
        text,
        lambda chunk: pattern.sub(
            lambda m: m.group(1) + _case_like(m.group(2), "as long as"), chunk
        ),
    )


def sentence_initial_for_this_reason(text: str, rng: random.Random) -> str:
    return _replace_sentence_initial("For this reason", "Therefore")(text, rng)


def sentence_initial_in_contrast(text: str, rng: random.Random) -> str:
    return _replace_sentence_initial("In contrast", "By contrast")(text, rng)


def sentence_initial_to_summarize(text: str, rng: random.Random) -> str:
    return _replace_sentence_initial("To summarize", "In summary")(text, rng)


RULES: dict[str, Rule] = {
    "syn_parenthetical_to_dashes": parenthetical_to_dashes,
    "syn_parenthetical_to_commas": parenthetical_to_commas,
    "syn_comma_parenthetical_to_dashes": comma_parenthetical_to_dashes,
    "syn_e_g_expansion": e_g_expansion,
    "syn_i_e_expansion": i_e_expansion,
    "syn_viz_expansion": viz_expansion,
    "syn_that_is_to_say": that_is_to_say,
    "syn_put_another_way": put_another_way,
    "syn_as_a_consequence": as_a_consequence,
    "syn_as_a_result_of": as_a_result_of,
    "syn_due_to_equivalent": due_to_equivalent,
    "syn_for_the_reason_that": for_the_reason_that,
    "syn_except_for_equivalent": except_for_equivalent,
    "syn_in_addition_to_equivalent": in_addition_to_equivalent,
    "syn_along_with_equivalent": along_with_equivalent,
    "syn_in_other_words_to_that_is": in_other_words_to_that_is,
    "syn_in_particular_to_specifically": in_particular_to_specifically,
    "syn_more_specifically_to_specifically": more_specifically_to_specifically,
    "syn_roughly_to_about": roughly_to_about,
    "syn_in_a_similar_way_to_similarly": in_a_similar_way_to_similarly,
    "syn_as_long_as_to_provided_that": as_long_as_to_provided_that,
    "syn_provided_that_to_as_long_as": provided_that_to_as_long_as,
    "syn_sentence_initial_for_this_reason": sentence_initial_for_this_reason,
    "syn_sentence_initial_in_contrast": sentence_initial_in_contrast,
    "syn_sentence_initial_to_summarize": sentence_initial_to_summarize,
}

RULE_LABELS: dict[str, str] = {
    "syn_parenthetical_to_dashes": "Format parenthetical as an aside",
    "syn_parenthetical_to_commas": "Format parenthetical with commas",
    "syn_comma_parenthetical_to_dashes": "Format comma-delimited aside with dashes",
    "syn_e_g_expansion": "Expand e.g. in prose",
    "syn_i_e_expansion": "Expand i.e. in prose",
    "syn_viz_expansion": "Expand viz. in prose",
    "syn_that_is_to_say": "Rephrase that is to say",
    "syn_put_another_way": "Rephrase put another way",
    "syn_as_a_consequence": "Rephrase as a consequence",
    "syn_as_a_result_of": "Rephrase as a result of",
    "syn_due_to_equivalent": "Rephrase due to",
    "syn_for_the_reason_that": "Shorten for the reason that",
    "syn_except_for_equivalent": "Shorten with the exception of",
    "syn_in_addition_to_equivalent": "Rephrase in addition to",
    "syn_along_with_equivalent": "Rephrase along with",
    "syn_in_other_words_to_that_is": "Shorten in other words",
    "syn_in_particular_to_specifically": "Shorten in particular",
    "syn_more_specifically_to_specifically": "Reword more specifically",
    "syn_roughly_to_about": "Use about for roughly",
    "syn_in_a_similar_way_to_similarly": "Shorten in a similar way",
    "syn_as_long_as_to_provided_that": "Rephrase conditional as long as",
    "syn_provided_that_to_as_long_as": "Rephrase conditional provided that",
    "syn_sentence_initial_for_this_reason": "Rephrase sentence-initial for this reason",
    "syn_sentence_initial_in_contrast": "Rephrase sentence-initial in contrast",
    "syn_sentence_initial_to_summarize": "Rephrase sentence-initial to summarize",
}
