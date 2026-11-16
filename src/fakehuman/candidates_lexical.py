# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: William Droz <william.droz@idiap.ch>
# SPDX-License-Identifier: GPL-3.0-only

"""Narrow lexical alternatives for text-detector experiments.

Each rule rewrites only an explicitly listed phrase. These alternatives are
intended for ordinary research prose; they are not a general thesaurus and do
not establish that a transformed passage preserves meaning in context.
"""

from __future__ import annotations

import random
import re

from .rules import Rule, _case_like, _outside_protected

_NUMBER_WORDS = (
    "zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    "thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|"
    "thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|"
    "half|a quarter|a third"
)
_QUANTITY = re.compile(r"^[ \t]+(?:\d|(?:" + _NUMBER_WORDS + r")\b)", re.I)
_DISCOURSE = {
    "for this reason",
    "in other words",
    "put differently",
    "that is to say",
    "as an example",
    "all the same",
    "even so",
    "put another way",
    "as a result",
    "in addition",
    "in contrast",
    "by contrast",
    "in particular",
}
_SHORT_NOUNS = re.compile(
    r"^[ \t]+(?:break|pause|interval|period|visit|stay|meeting|session|report|"
    r"note|message|summary|account|description|explanation|introduction|"
    r"presentation|discussion|conversation|questionnaire|survey|document|"
    r"letter|email|speech|statement|overview|review)s?\b",
    re.I,
)


def _safe_context(source: str, before: str, after: str) -> bool:
    """Skip known ambiguous grammatical uses instead of guessing their sense."""
    if source == "due to":
        # A causal noun phrase is different from a scheduled infinitive:
        # 'due to the delay' can change; 'due to arrive' must remain intact.
        clause = re.split(r"[.!?;\n]", before)[-1]
        if re.search(
            r"\b(?:payments?|money|amounts?|balances?|sums?|fees?|tax|taxes|rent|"
            r"royalties|credit|debts?|wages|salary|respect|recognition)\b",
            clause,
            re.I,
        ):
            # 'The payment due to the bank' describes an obligation, not a cause.
            return False
        return bool(
            re.match(
                r"^[ \t]+(?:the|a|an|this|that|these|those|my|our|your|their|his|her|its|"
                r"some|any|no|rain|noise|weather|fatigue|illness|illnesses|delays?|"
                r"errors?|failures?|damage|overheating|maintenance|traffic|"
                r"[A-Za-z]+\d+[A-Za-z0-9]*|\d+)\b",
                after,
                re.I,
            )
        )
    if source in _DISCOURSE:
        # Require punctuation around an explicit discourse marker. Avoid e.g.
        # 'they are all the same' or 'describe it in other words'.
        return bool(re.search(r"(?:^|[.!?;,])[ \t]*$", before)) and bool(
            re.match(r",[ \t]+\S", after)
        )
    if source == "in advance":
        return not re.match(r"^[ \t]+of\b", after, re.I)
    if source == "along with":
        return not re.search(
            r"\b(?:get|gets|got|getting|gotten|go|goes|went|going|"
            r"play|plays|played|playing)[ \t]+$",
            before,
            re.I,
        )
    if source == "at least":
        return bool(_QUANTITY.match(after))
    if source in {"a short", "a brief"}:
        return bool(_SHORT_NOUNS.match(after))
    if source == "at the same time":
        # Sentence-final temporal use; do not change a concessive opener or
        # 'at the same time as', which needs a different construction.
        return bool(before.strip()) and bool(re.match(r"^[.!?](?:\s|$)", after))
    if source == "at the end of":
        return bool(
            re.match(
                r"^[ \t]+(?:(?:the|a|an|this|that|each|every|our|their)[ \t]+)?"
                r"(?:study|test|trial|session|meeting|course|year|day|month|week|"
                r"chapter|report|presentation|discussion|experiment|project)\b",
                after,
                re.I,
            )
        )
    if source in {"carry out", "carries out", "carried out"}:
        return bool(
            re.match(
                r"^[ \t]+(?:(?:the|a|an|this|that|these|those|our|their|each|several|"
                r"some|many|\d+)[ \t]+)?(?:tests?|trials?|experiments?|analyses|analysis|"
                r"studies|study|surveys?|checks?|inspections?|measurements?|reviews?|"
                r"tasks?|procedures?|operations?|assessments?|evaluations?)\b",
                after,
                re.I,
            )
        )
    return True


def _contextual_candidate(source: str, target: str) -> Rule:
    """Apply a phrase only where its narrow grammatical use is supported."""
    pattern = re.compile(
        r"(?<![\w'’/\\\-])" + re.escape(source) + r"(?![\w'’/\\\-])", re.I
    )

    def transform(text: str, rng: random.Random) -> str:
        def replace(chunk: str) -> str:
            return pattern.sub(
                lambda match: (
                    _case_like(match.group(), target)
                    if _safe_context(
                        source, chunk[: match.start()], chunk[match.end() :]
                    )
                    else match.group()
                ),
                chunk,
            )

        return _outside_protected(text, replace)

    return transform


def _candidate(source: str, target: str) -> Rule:
    """Create a case-aware, protected-span-aware exact phrase substitution."""
    transform = _contextual_candidate(source, target)

    transform.__name__ = "lex_" + source.replace(" ", "_").replace("'", "")
    transform.__doc__ = f"Replace the exact phrase {source!r} with {target!r}."
    return transform


# Keep these as separate, inspectable candidates so detector results can be
# attributed to one lexical choice. Mappings are explicit and intentionally
# modest: no technical nouns, claims, numbers, negation, or uncertainty terms.
_PAIRS: tuple[tuple[str, str, str], ...] = (
    ("lex_due_to", "due to", "because of"),
    ("lex_because_of", "because of", "owing to"),
    ("lex_as_a_result_of", "as a result of", "because of"),
    ("lex_for_this_reason", "for this reason", "therefore"),
    ("lex_in_other_words", "in other words", "that is"),
    ("lex_put_differently", "put differently", "in other words"),
    ("lex_that_is_to_say", "that is to say", "that is"),
    ("lex_with_respect_to", "with respect to", "regarding"),
    ("lex_in_relation_to", "in relation to", "regarding"),
    ("lex_in_terms_of", "in terms of", "with regard to"),
    ("lex_prior_to", "prior to", "before"),
    ("lex_subsequent_to", "subsequent to", "after"),
    ("lex_at_the_start_of", "at the start of", "at the beginning of"),
    ("lex_at_the_beginning_of", "at the beginning of", "at the start of"),
    ("lex_at_the_end_of", "at the end of", "at the conclusion of"),
    ("lex_over_the_course_of", "over the course of", "during"),
    ("lex_at_the_same_time", "at the same time", "simultaneously"),
    ("lex_for_the_most_part", "for the most part", "mostly"),
    ("lex_to_a_certain_extent", "to a certain extent", "to some extent"),
    ("lex_in_a_similar_manner", "in a similar manner", "similarly"),
    ("lex_in_the_same_way", "in the same way", "in the same manner"),
    ("lex_in_particular", "in particular", "specifically"),
    ("lex_more_specifically", "more specifically", "more precisely"),
    ("lex_for_instance", "for instance", "for example"),
    ("lex_as_an_example", "as an example", "for example"),
    ("lex_in_contrast", "in contrast", "by contrast"),
    ("lex_by_contrast", "by contrast", "in contrast"),
    ("lex_even_so", "even so", "nevertheless"),
    ("lex_all_the_same", "all the same", "nevertheless"),
    ("lex_in_addition", "in addition", "additionally"),
    ("lex_in_the_meantime", "in the meantime", "meanwhile"),
    ("lex_also_known_as", "also known as", "also called"),
    ("lex_is_composed_of", "is composed of", "consists of"),
    ("lex_are_composed_of", "are composed of", "consist of"),
    ("lex_consists_of", "consists of", "is composed of"),
    ("lex_consist_of", "consist of", "comprise"),
    ("lex_give_rise_to", "give rise to", "cause"),
    ("lex_gives_rise_to", "gives rise to", "causes"),
    ("lex_result_in", "result in", "lead to"),
    ("lex_results_in", "results in", "leads to"),
    ("lex_resulted_in", "resulted in", "led to"),
    ("lex_result_from", "result from", "arise from"),
    ("lex_results_from", "results from", "arises from"),
    ("lex_resulted_from", "resulted from", "arose from"),
    ("lex_carry_out", "carry out", "perform"),
    ("lex_carries_out", "carries out", "performs"),
    ("lex_carried_out", "carried out", "performed"),
    ("lex_take_into_account", "take into account", "consider"),
    ("lex_takes_into_account", "takes into account", "considers"),
    ("lex_took_into_account", "took into account", "considered"),
    # Frequent, context-checked wording from the training examples.
    (
        "lex_a_useful_recommendation",
        "a useful recommendation",
        "a helpful recommendation",
    ),
    ("lex_useful_recommendation", "useful recommendation", "helpful recommendation"),
    (
        "lex_my_useful_recommendation",
        "my useful recommendation",
        "my helpful recommendation",
    ),
    ("lex_recommendation_is_to", "recommendation is to", "suggestion is to"),
    (
        "lex_a_useful_recommendation_is_to",
        "a useful recommendation is to",
        "a helpful suggestion is to",
    ),
    (
        "lex_my_useful_recommendation_is_to",
        "my useful recommendation is to",
        "my helpful suggestion is to",
    ),
    ("lex_for_example_instance", "for example", "for instance"),
    ("lex_therefore_thus", "therefore", "thus"),
    ("lex_consequently_as_a_result", "consequently", "as a result"),
    ("lex_moreover_in_addition", "moreover", "in addition"),
    ("lex_nevertheless_even_so", "nevertheless", "even so"),
    ("lex_a_limitation_is_that", "a limitation is that", "one limitation is that"),
    ("lex_an_important_limitation_key", "an important limitation", "a key limitation"),
    ("lex_useful_example_helpful", "useful example", "helpful example"),
    ("lex_a_useful_example_helpful", "a useful example", "a helpful example"),
    ("lex_a_short_brief", "a short", "a brief"),
    ("lex_short_questionnaire_brief", "short questionnaire", "brief questionnaire"),
    ("lex_a_brief_short", "a brief", "a short"),
    ("lex_brief_note_short", "brief note", "short note"),
    ("lex_concrete_example_specific", "concrete example", "specific example"),
    ("lex_a_concrete_example_specific", "a concrete example", "a specific example"),
    (
        "lex_a_simple_framework_straightforward",
        "a simple framework",
        "a straightforward framework",
    ),
    (
        "lex_a_simple_structure_straightforward",
        "a simple structure",
        "a straightforward structure",
    ),
    ("lex_a_simple_table_basic", "a simple table", "a basic table"),
    ("lex_simple_bar_chart_basic", "simple bar chart", "basic bar chart"),
    ("lex_simple_line_chart_basic", "simple line chart", "basic line chart"),
    ("lex_at_least_minimum", "at least", "a minimum of"),
    ("lex_one_page_single", "one-page", "single-page"),
    ("lex_in_advance_beforehand", "in advance", "beforehand"),
    ("lex_the_same_idea_identical", "the same idea", "an identical idea"),
    ("lex_the_same_package_identical", "the same package", "the identical package"),
    ("lex_as_a_result_therefore", "as a result", "therefore"),
)

RULES: dict[str, Rule] = {
    name: _candidate(source, target) for name, source, target in _PAIRS
}


def _initial_transition(source: str, target: str) -> Rule:
    """Change a sentence-initial transition only when followed by a comma."""
    pattern = re.compile(
        r"(^|[.!?;]\s+)" + re.escape(source) + r"(?=,\s+)",
        re.IGNORECASE | re.MULTILINE,
    )

    def transform(text: str, rng: random.Random) -> str:
        return _outside_protected(
            text,
            lambda chunk: pattern.sub(
                lambda match: (
                    match[1] + _case_like(match.group()[len(match[1]) :], target)
                ),
                chunk,
            ),
        )

    transform.__name__ = "lex_initial_" + source
    transform.__doc__ = (
        f"Replace sentence-initial {source!r} before a comma with {target!r}."
    )
    return transform


RULES["lex_initial_however_nevertheless"] = _initial_transition(
    "However", "Nevertheless"
)

RULE_LABELS: dict[str, str] = {
    name: f"Replace {source!r} with {target!r}" for name, source, target in _PAIRS
}
RULE_LABELS["lex_initial_however_nevertheless"] = (
    "Reword sentence-initial 'However,' as 'Nevertheless,'"
)
