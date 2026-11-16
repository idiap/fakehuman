# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: William Droz <william.droz@idiap.ch>
# SPDX-License-Identifier: GPL-3.0-only

"""Illustrate individual frozen rules with examples from the rule engine."""

from collections.abc import Sequence

from .candidates_lexical import _PAIRS
from .research_rules import RULE_LABELS, apply_rules
from .rule_categories import rule_category

_LEXICAL_SENTENCES = {
    "lex_due_to": "The match was postponed due to rain.",
    "lex_because_of": "The match was postponed because of rain.",
    "lex_as_a_result_of": "The start was delayed as a result of rain.",
    "lex_for_this_reason": "For this reason, we repeated the test.",
    "lex_in_other_words": "In other words, the test can start.",
    "lex_put_differently": "Put differently, the test can start.",
    "lex_that_is_to_say": "That is to say, the test can start.",
    "lex_with_respect_to": "We raised a question with respect to the schedule.",
    "lex_in_relation_to": "We raised a question in relation to the schedule.",
    "lex_in_terms_of": "The options differ in terms of cost.",
    "lex_prior_to": "We checked the seals prior to departure.",
    "lex_subsequent_to": "We checked the seals subsequent to arrival.",
    "lex_at_the_start_of": "We checked the seals at the start of the test.",
    "lex_at_the_beginning_of": "We checked the seals at the beginning of the test.",
    "lex_at_the_end_of": "We stopped at the end of the trial.",
    "lex_over_the_course_of": "The temperature rose over the course of the test.",
    "lex_at_the_same_time": "Both tests ran at the same time.",
    "lex_for_the_most_part": "For the most part, the samples remained intact.",
    "lex_to_a_certain_extent": "The two routes overlap to a certain extent.",
    "lex_in_a_similar_manner": "The two devices operate in a similar manner.",
    "lex_in_the_same_way": "Both teams measured the samples in the same way.",
    "lex_in_particular": "In particular, check the main entrance.",
    "lex_more_specifically": "More specifically, check the seal around the lid.",
    "lex_for_instance": "For instance, the small tray holds one sample.",
    "lex_as_an_example": "As an example, the small tray holds one sample.",
    "lex_in_contrast": "In contrast, the second sample stayed intact.",
    "lex_by_contrast": "By contrast, the second sample stayed intact.",
    "lex_even_so": "Even so, we kept the second sample.",
    "lex_all_the_same": "All the same, we kept the second sample.",
    "lex_in_addition": "In addition, we saved a copy.",
    "lex_in_the_meantime": "In the meantime, we prepared the next sample.",
    "lex_also_known_as": "The herb is also known as sweet basil.",
    "lex_is_composed_of": "The assembly is composed of three sections.",
    "lex_are_composed_of": "The assemblies are composed of three sections each.",
    "lex_consists_of": "The assembly consists of three sections.",
    "lex_consist_of": "The assemblies consist of three sections each.",
    "lex_give_rise_to": "These faults give rise to repeated delays.",
    "lex_gives_rise_to": "This fault gives rise to repeated delays.",
    "lex_result_in": "These faults result in repeated delays.",
    "lex_results_in": "This fault results in repeated delays.",
    "lex_resulted_in": "The fault resulted in repeated delays.",
    "lex_result_from": "These delays result from equipment faults.",
    "lex_results_from": "This delay results from an equipment fault.",
    "lex_resulted_from": "The delay resulted from an equipment fault.",
    "lex_carry_out": "We carry out tests every week.",
    "lex_carries_out": "The team carries out tests every week.",
    "lex_carried_out": "The team carried out tests last week.",
    "lex_take_into_account": "We take into account the cost of transport.",
    "lex_takes_into_account": "The estimate takes into account the cost of transport.",
    "lex_took_into_account": "The estimate took into account the cost of transport.",
    "lex_a_useful_recommendation": "The report contains a useful recommendation.",
    "lex_useful_recommendation": "We followed the useful recommendation in the report.",
    "lex_my_useful_recommendation": "The team followed my useful recommendation.",
    "lex_recommendation_is_to": "Our recommendation is to keep a second copy.",
    "lex_a_useful_recommendation_is_to": "A useful recommendation is to keep a second copy.",
    "lex_my_useful_recommendation_is_to": "My useful recommendation is to keep a second copy.",
    "lex_for_example_instance": "For example, the small tray holds one sample.",
    "lex_therefore_thus": "The room was booked. Therefore, we met in the lobby.",
    "lex_consequently_as_a_result": "Consequently, we moved the meeting to the lobby.",
    "lex_moreover_in_addition": "Moreover, the second sample was intact.",
    "lex_nevertheless_even_so": "Nevertheless, the second sample was intact.",
    "lex_a_limitation_is_that": "A limitation is that the sensor requires daylight.",
    "lex_an_important_limitation_key": "The short battery life is an important limitation.",
    "lex_useful_example_helpful": "The report includes another useful example.",
    "lex_a_useful_example_helpful": "The report includes a useful example.",
    "lex_a_short_brief": "We wrote a short report.",
    "lex_short_questionnaire_brief": "Participants completed a short questionnaire.",
    "lex_a_brief_short": "We took a brief break before the next session.",
    "lex_brief_note_short": "The team left a brief note beside the equipment.",
    "lex_concrete_example_specific": "The report includes another concrete example.",
    "lex_a_concrete_example_specific": "The report includes a concrete example.",
    "lex_a_simple_framework_straightforward": "We used a simple framework to compare the options.",
    "lex_a_simple_structure_straightforward": "The report follows a simple structure.",
    "lex_a_simple_table_basic": "We recorded the results in a simple table.",
    "lex_simple_bar_chart_basic": "A simple bar chart shows the sample counts.",
    "lex_simple_line_chart_basic": "A simple line chart shows the daily temperature.",
    "lex_at_least_minimum": "Allow at least three days.",
    "lex_one_page_single": "We prepared a one-page report for the meeting.",
    "lex_in_advance_beforehand": "Please check the equipment in advance.",
    "lex_the_same_idea_identical": "Both speakers expressed the same idea.",
    "lex_the_same_package_identical": "The courier returned the same package.",
    "lex_as_a_result_therefore": "As a result, we kept the second sample.",
}

_EXAMPLES = {
    **{
        name: (
            f"Replaces {source!r} with {target!r} outside protected spans.",
            _LEXICAL_SENTENCES[name],
        )
        for name, source, target in _PAIRS
    },
    "lex_therefore_thus": (
        "Uses 'thus' as an alternative to 'therefore' to introduce a result.",
        "The room was booked. Therefore, we met in the lobby.",
    ),
    "lex_initial_however_nevertheless": (
        "Rewords an opening 'However,' when whitespace follows the comma.",
        "However, the revised schedule still works.",
    ),
    "lex_due_to": (
        "Rewords causal 'due to' before recognized noun phrases; skips scheduled or owed senses.",
        "The match was postponed due to rain.",
    ),
    "lex_a_brief_short": (
        "Changes 'a brief' to 'a short' before selected nouns such as 'break' or 'report'.",
        "We took a brief break before the next session.",
    ),
    "normalize_dashes": (
        "Replaces em dashes between prose words with commas, keeping variable and acronym pairs.",
        "One option—worth testing—is to add another trial.",
    ),
    "normalize_quotes": (
        "Changes curly quotation marks and apostrophes in prose to straight ones; keeps code and markup intact.",
        "The form reads “complete” when it’s ready.",
    ),
    "normalize_spaces": (
        "Collapses repeated spaces inside prose, preserving tabs, indentation, line breaks and Markdown hard breaks.",
        "Please  review the results.",
    ),
    "trim_openers": (
        "Removes selected sentence openers when text follows on the same line.",
        "Furthermore, the second sample was intact.",
    ),
    "remove_stock_framing": (
        "Removes selected same-line lead-ins such as 'It is worth noting that'.",
        "It is worth noting that both teams agreed.",
    ),
    "simplify_verbs": (
        "Shortens selected formal verbs, keeping noun uses such as 'an endeavor'.",
        "We utilized the same process for both samples.",
    ),
    "simplify_connectors": (
        "Shortens selected phrases that connect ideas or express conditions.",
        "We paused in order to check the readings.",
    ),
    "shorten_phrases": (
        "Shortens a fixed set of wordy expressions.",
        "We review the results on a regular basis.",
    ),
    "active_phrases": (
        "Replaces certain verb and noun combinations with a single verb.",
        "We conducted an analysis of the samples.",
    ),
    "contractions": (
        "Contracts selected negative verb forms.",
        "We did not adjust the settings.",
    ),
    "expand_contractions": (
        "Expands selected negative contractions into their full forms.",
        "We didn't adjust the settings.",
    ),
    "split_semicolons": (
        "Splits a semicolon when the following clause has a recognized subject and finite verb.",
        "The first test ended; another began.",
    ),
    "strip_heading_markup": (
        "Removes opening and optional closing hash marks from Markdown headings.",
        "## Results",
    ),
    "seeded_synonyms": (
        "Uses the seed to choose among a small set of alternatives; 'however' changes only before a comma.",
        "Additionally, the second sample was intact.",
    ),
    "simple_list_transitions": (
        "Replaces 'firstly', 'secondly', 'thirdly' and 'lastly' with shorter forms.",
        "Firstly, check that the light is on.",
    ),
    "affirmative_contractions": (
        "Contracts selected pronoun and verb pairs before a following word; skips uses after words such as 'where' or 'how'.",
        "We are ready to start.",
    ),
    "auxiliary_have_contractions": (
        "Contracts 'have' after certain pronouns and before listed participles.",
        "We have measured the samples.",
    ),
    "common_plain_verbs": (
        "Simplifies selected verbs while retaining 'required to', conditional 'providing' and 'provide … with'.",
        "They assisted the team with the setup.",
    ),
    "example_alternatives": (
        "Rewords 'for example' and 'as an example'.",
        "For example, the small tray holds one sample.",
    ),
    "contrast_transitions": (
        "Rewords opening contrasts as 'Still,' or 'Even so,' unless that phrase already occurs in the same clause.",
        "However, the team kept the second sample.",
    ),
    "result_transitions": (
        "Rewords selected opening result transitions as 'So' when prose follows on the same line.",
        "Therefore, we kept the second sample.",
    ),
    "two_sentence_paragraphs": (
        "Inserts paragraph breaks at every second eligible sentence boundary.",
        "We opened the box. The seal was intact. We recorded the result.",
    ),
    "ability_phrases": (
        "Shortens selected present-ability phrases to 'can', checking for modal, negative and infinitive qualifications.",
        "We are able to repeat the test.",
    ),
    "recommendation_phrases": (
        "Rewords a fixed set of phrases that introduce recommendations.",
        "Our recommendation is to keep a second copy.",
    ),
    "shorter_time_phrases": (
        "Shortens selected expressions about timing or frequency.",
        "We check the readings on a daily basis.",
    ),
    "direct_nominal_phrases": (
        "Replaces selected action phrases with a direct verb.",
        "We made a decision to repeat the test.",
    ),
    "broader_seeded_synonyms": (
        "Uses the seed to keep or replace selected phrases. The example shows one possible output.",
        "A useful next step is to obtain feedback and ask for additional information.",
    ),
    "syn_parenthetical_to_dashes": (
        "Uses em dashes for eligible mid-sentence prose asides, skipping notation and nested parentheses.",
        "The sample (still warm) arrived this morning.",
    ),
    "syn_parenthetical_to_commas": (
        "Uses commas for eligible mid-sentence prose asides, skipping notation and nested parentheses.",
        "The sample (still warm) arrived this morning.",
    ),
    "syn_comma_parenthetical_to_dashes": (
        "Uses dashes for short comma-delimited asides with recognized lead-ins such as 'still' or 'which'.",
        "The sample, still warm, arrived this morning.",
    ),
    "syn_e_g_expansion": (
        "Replaces 'e.g.' with 'for example'.",
        "You can sort by a label, e.g. by name.",
    ),
    "syn_i_e_expansion": (
        "Replaces 'i.e.' with 'that is'.",
        "Use the main entrance, i.e. the door beside the ramp.",
    ),
    "syn_viz_expansion": (
        "Replaces 'viz.' with 'namely'.",
        "One item is missing, viz. the lid.",
    ),
    "syn_that_is_to_say": (
        "Replaces 'that is to say' with 'in other words'.",
        "That is to say, the test can start.",
    ),
    "syn_put_another_way": (
        "Replaces 'put another way' with 'in other words'.",
        "Put another way, the test can start.",
    ),
    "syn_as_a_consequence": (
        "Replaces 'as a consequence' with 'as a result'.",
        "As a consequence, we ran another test.",
    ),
    "syn_as_a_result_of": (
        "Replaces 'as a result of' with 'because of'.",
        "The start was delayed as a result of rain.",
    ),
    "syn_due_to_equivalent": (
        "Replaces causal 'due to' before recognized noun phrases, skipping scheduled or owed senses.",
        "The match was postponed due to rain.",
    ),
    "syn_for_the_reason_that": (
        "Replaces 'for the reason that' with 'because'.",
        "We repeated the test for the reason that the first reading was unclear.",
    ),
    "syn_except_for_equivalent": (
        "Replaces 'with the exception of' with 'except for'.",
        "We checked all items with the exception of the lid.",
    ),
    "syn_in_addition_to_equivalent": (
        "Replaces 'in addition to' with 'besides'.",
        "We packed a spare cable in addition to the charger.",
    ),
    "syn_along_with_equivalent": (
        "Replaces 'along with' with 'together with'.",
        "We sent the chart along with the report.",
    ),
    "syn_in_other_words_to_that_is": (
        "Replaces 'in other words' with 'that is'.",
        "In other words, we can start.",
    ),
    "syn_in_particular_to_specifically": (
        "Replaces 'in particular' with 'specifically'.",
        "In particular, check the main entrance.",
    ),
    "syn_more_specifically_to_specifically": (
        "Rewords 'more specifically' as 'more precisely', keeping the comparison.",
        "More specifically, check the main entrance.",
    ),
    "syn_roughly_to_about": (
        "Replaces 'roughly' with 'about' before a numeric quantity; leaves manner uses unchanged.",
        "The test took roughly ten minutes.",
    ),
    "syn_in_a_similar_way_to_similarly": (
        "Replaces 'in a similar way' with 'similarly'.",
        "In a similar way, the second run used a smaller tray.",
    ),
    "syn_as_long_as_to_provided_that": (
        "Rewords 'as long as' as 'provided that' only in recognized conditional contexts.",
        "You can start as long as the light is on.",
    ),
    "syn_provided_that_to_as_long_as": (
        "Rewords sentence-opening 'provided that' when a condition with a recognized verb follows.",
        "Provided that the light is on, you can start.",
    ),
    "syn_sentence_initial_for_this_reason": (
        "Replaces 'For this reason,' with 'Therefore,' at the start of a sentence.",
        "For this reason, we repeated the test.",
    ),
    "syn_sentence_initial_in_contrast": (
        "Replaces 'In contrast,' with 'By contrast,' at the start of a sentence.",
        "In contrast, the second sample stayed intact.",
    ),
    "syn_sentence_initial_to_summarize": (
        "Replaces 'To summarize,' with 'In summary,' at the start of a sentence.",
        "To summarize, we recorded both results.",
    ),
}

# Context-sensitive rules use complete sentences that exercise their safe use.
for _name, _sample in {
    "lex_for_this_reason": "For this reason, we repeated the test.",
    "lex_in_other_words": "In other words, the test can start.",
    "lex_put_differently": "Put differently, the test can start.",
    "lex_that_is_to_say": "That is to say, the test can start.",
    "lex_as_an_example": "As an example, the small tray holds one sample.",
    "lex_all_the_same": "All the same, we kept the second sample.",
    "lex_even_so": "Even so, we kept the second sample.",
    "lex_as_a_result_therefore": "As a result, we kept the second sample.",
    "lex_in_addition": "In addition, we saved a copy.",
    "lex_in_contrast": "In contrast, the second sample stayed intact.",
    "lex_by_contrast": "By contrast, the second sample stayed intact.",
    "lex_in_particular": "In particular, check the main entrance.",
    "lex_at_least_minimum": "Allow at least three days.",
    "lex_a_short_brief": "We wrote a short report.",
    "lex_at_the_same_time": "Both tests ran at the same time.",
    "lex_at_the_end_of": "We stopped at the end of the trial.",
    "lex_carry_out": "We carry out tests every week.",
    "lex_carries_out": "The team carries out tests every week.",
    "lex_carried_out": "The team carried out tests last week.",
}.items():
    _description, _ = _EXAMPLES[_name]
    _EXAMPLES[_name] = (_description + " Ambiguous uses are left unchanged.", _sample)


def rule_examples(names: Sequence[str], seed: int) -> list[dict[str, str]]:
    """Return categories, descriptions, and examples in pipeline order."""
    examples = []
    for name in names:
        description, example = _EXAMPLES[name]
        before = example + "\n"
        after = apply_rules(before, [name], seed)
        if name == "broader_seeded_synonyms" and before == after:
            # This seed illustrates one possible change if the configured seed
            # happens to keep all phrases in the sample.
            after = apply_rules(before, [name], 17)
        examples.append(
            {
                "id": name,
                "label": RULE_LABELS[name],
                "category": rule_category(name),
                "description": description,
                "before": before,
                "after": after,
            }
        )
    return examples
