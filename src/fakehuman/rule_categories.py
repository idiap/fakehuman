# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: William Droz <william.droz@idiap.ch>
# SPDX-License-Identifier: GPL-3.0-only

"""Display categories for rules, based on the kind of edit they make."""

_DEFAULT_CATEGORY = "Word and phrase swaps"

_SPECIALIZED_RULES = {
    "Transitions and framing": {
        "contrast_transitions",
        "example_alternatives",
        "lex_a_limitation_is_that",
        "lex_a_useful_recommendation_is_to",
        "lex_all_the_same",
        "lex_as_a_result_therefore",
        "lex_as_an_example",
        "lex_by_contrast",
        "lex_consequently_as_a_result",
        "lex_even_so",
        "lex_for_example_instance",
        "lex_for_instance",
        "lex_for_this_reason",
        "lex_in_addition",
        "lex_in_contrast",
        "lex_in_other_words",
        "lex_in_particular",
        "lex_in_the_meantime",
        "lex_initial_however_nevertheless",
        "lex_more_specifically",
        "lex_moreover_in_addition",
        "lex_my_useful_recommendation_is_to",
        "lex_nevertheless_even_so",
        "lex_put_differently",
        "lex_recommendation_is_to",
        "lex_that_is_to_say",
        "lex_therefore_thus",
        "plain_additive_openers",
        "plain_concessive_openers",
        "recommendation_phrases",
        "remove_stock_framing",
        "result_transitions",
        "simple_list_transitions",
        "syn_as_a_consequence",
        "syn_in_other_words_to_that_is",
        "syn_in_particular_to_specifically",
        "syn_more_specifically_to_specifically",
        "syn_put_another_way",
        "syn_sentence_initial_for_this_reason",
        "syn_sentence_initial_in_contrast",
        "syn_sentence_initial_to_summarize",
        "syn_that_is_to_say",
        "trim_openers",
    },
    "Contractions and abbreviations": {
        "affirmative_contractions",
        "auxiliary_have_contractions",
        "contractions",
        "expand_contractions",
        "syn_e_g_expansion",
        "syn_i_e_expansion",
        "syn_viz_expansion",
    },
    "Sentence and paragraph structure": {
        "ability_phrases",
        "active_phrases",
        "direct_nominal_phrases",
        "split_semicolons",
        "two_sentence_paragraphs",
    },
    "Punctuation and formatting": {
        "normalize_dashes",
        "normalize_quotes",
        "normalize_spaces",
        "strip_heading_markup",
        "syn_comma_parenthetical_to_dashes",
        "syn_parenthetical_to_commas",
        "syn_parenthetical_to_dashes",
    },
}

RULE_CATEGORIES = (_DEFAULT_CATEGORY, *_SPECIALIZED_RULES)
_CATEGORY_BY_RULE = {
    name: category for category, names in _SPECIALIZED_RULES.items() for name in names
}


def rule_category(name: str) -> str:
    """Return a display category for a rule ID."""
    return _CATEGORY_BY_RULE.get(name, _DEFAULT_CATEGORY)
