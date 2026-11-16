# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: William Droz <william.droz@idiap.ch>
# SPDX-License-Identifier: GPL-3.0-only

"""Exercise browser interactions when the optional NiceGUI extra is installed."""

import asyncio
import json
import re
import threading

import pytest

pytest.importorskip("nicegui")

from nicegui import ui  # noqa: E402
from nicegui.testing import user_simulation  # noqa: E402

from fakehuman.cli import load_manifest  # noqa: E402
from fakehuman.costs import summarize_responses  # noqa: E402
from fakehuman.document_chunks import MAX_TECHNICAL_CHARACTERS  # noqa: E402
from fakehuman.llm_client import LLMConfigurationError  # noqa: E402
from fakehuman.research_rules import RULE_LABELS, RULES, apply_rules  # noqa: E402
from fakehuman.rule_catalog import catalog_rule_names  # noqa: E402
from fakehuman.rule_categories import RULE_CATEGORIES, rule_category  # noqa: E402
from fakehuman.ui import _build_page  # noqa: E402


class FixtureService:
    manifest = load_manifest()

    def __init__(self, *, refused=False):
        self.refused = refused
        self.score_calls = []

    def inspect_transformation(self, text, *, enabled_rules, include_catalog):
        assert include_catalog
        self.enabled_rules = enabled_rules
        if not text.strip():
            raise ValueError("Provide nonempty text up to 100,000 characters")
        attempted = apply_rules(text, enabled_rules, self.manifest["seed"])
        return {
            "text": text if self.refused else attempted,
            "attempted_text": attempted,
            "refused": self.refused,
            "refusal_reasons": ["Fixture meaning refusal"] if self.refused else [],
            "changes": [],
            "meaning": {"passed": not self.refused},
            "quality_issues": [],
        }

    def score_text(self, text, *, detailed):
        assert detailed
        self.score_calls.append(text)
        return {"scores": {"fixture": 0.6 if text.startswith("Thus") else 0.8}}


def test_text_actions_follow_available_input_and_output_after_processing():
    async def exercise():
        async with user_simulation(lambda: _build_page(FixtureService())) as user:
            await user.open("/")
            await user.should_not_see("Provider cost:")

            def enabled(label):
                return next(
                    iter(user.find(kind=ui.button, content=label).elements)
                ).enabled

            input_actions = ("Transform text", "Score input", "Copy input", "Clear")
            output_actions = ("Copy output", "Score output")
            assert not any(
                enabled(label) for label in (*input_actions, *output_actions)
            )

            user.find("Original text").type("   ")
            assert not enabled("Transform text")
            assert not enabled("Score input")
            assert enabled("Copy input")
            assert enabled("Clear")
            user.find("Clear").click()
            assert not any(enabled(label) for label in input_actions)

            user.find("Original text").type("Therefore, the team prepared a report.")
            assert all(enabled(label) for label in input_actions)
            assert not any(enabled(label) for label in output_actions)

            user.find("Score input").click()
            await user.should_see("Input scored", retries=20)
            assert all(enabled(label) for label in input_actions)
            assert not any(enabled(label) for label in output_actions)

            user.find("Transform text").click()
            await user.should_see("Changes accepted", retries=20)
            await user.should_see("Provider cost: $0.00 USD · local processing")
            assert next(iter(user.find("Accepted text").elements)).value
            assert all(enabled(label) for label in (*input_actions, *output_actions))

            user.find("Clear").click()
            await user.should_not_see("Provider cost:")
            assert next(iter(user.find("Original text").elements)).value == ""
            assert next(iter(user.find("Accepted text").elements)).value == ""
            assert not any(
                enabled(label) for label in (*input_actions, *output_actions)
            )

    asyncio.run(exercise())


def test_editing_input_clears_output_and_both_score_reports():
    async def exercise():
        service = FixtureService()
        async with user_simulation(lambda: _build_page(service)) as user:
            await user.open("/")
            user.find("Original text").type("Therefore, the team prepared a report.")
            user.find("Transform text").click()
            await user.should_see("Changes accepted", retries=20)
            accepted = next(iter(user.find("Accepted text").elements))
            assert accepted.value == "Thus, the team prepared a report."
            await user.should_see("Accepted output diff")
            await user.should_see(
                kind=ui.html,
                content='<del>Therefore</del><ins class="diff-replacement">Thus</ins>'
                ", the team prepared a report.",
            )
            user.find("Score input").click()
            await user.should_see("Input scored", retries=20)
            user.find("Score output").click()
            await user.should_see("Output scored", retries=20)
            table = next(iter(user.find(ui.table).elements))
            assert table.rows == [
                {
                    "detector": "fixture",
                    "input": "0.8000",
                    "output": "0.6000",
                    "change": "-0.2000",
                }
            ]
            user.find("Original text").type(" Another sentence.")
            assert accepted.value == ""
            await user.should_not_see("Accepted output diff")
            await user.should_not_see(ui.table)
            await user.should_not_see("Full report · JSON")
            assert not next(iter(user.find("Score output").elements)).enabled

    asyncio.run(exercise())


@pytest.mark.parametrize("action", ["Clear", "Load example", "Paste"])
def test_replacing_draft_can_be_undone_until_the_replacement_is_edited(action):
    async def exercise():
        async with user_simulation(lambda: _build_page(FixtureService())) as user:
            await user.open("/")
            original = next(iter(user.find("Original text").elements))
            accepted = next(iter(user.find("Accepted text").elements))
            draft = (
                "Therefore, the team prepared a report.\n\n  Keep café and spacing.  "
            )
            pasted = "Text from the clipboard."
            user.javascript_rules[
                re.compile(r"try \{ return await navigator\.clipboard\.readText\(\);")
            ] = lambda _: pasted

            if action != "Clear":
                user.find(kind=ui.button, content=action).click()
                if action == "Paste":
                    await user.should_see(pasted, retries=20)
                await user.should_not_see(kind=ui.button, content="Undo")
                user.find("Original text").clear()

            user.find("Original text").type(draft)
            user.find("Transform text").click()
            await user.should_see("Changes accepted", retries=20)
            assert accepted.value

            user.find(kind=ui.button, content=action).click()
            await user.should_see(kind=ui.button, content="Undo", retries=20)
            assert original.value != draft
            assert accepted.value == ""

            if original.value:
                user.find("Score input").click()
                await user.should_see("Input scored", retries=20)
                await user.should_see(ui.table)
            user.find(kind=ui.button, content="Undo").click()
            assert original.value == draft
            assert accepted.value == ""
            await user.should_not_see(ui.table)
            await user.should_not_see(kind=ui.button, content="Undo")

            user.find(kind=ui.button, content=action).click()
            await user.should_see(kind=ui.button, content="Undo", retries=20)
            user.find("Original text").type("My edit to the replacement.")
            await user.should_not_see(kind=ui.button, content="Undo")
            assert original.value.endswith("My edit to the replacement.")

    asyncio.run(exercise())


def test_editing_limit_follows_method_and_keeps_over_limit_draft_scoreable():
    async def exercise():
        service = FixtureService()
        async with user_simulation(lambda: _build_page(service)) as user:
            await user.open("/")
            user.find("Editing method").click()
            user.find("LLM technical rewrite (experimental)").click()
            original = next(iter(user.find("Original text").elements))
            transform = next(iter(user.find("Transform text").elements))
            score = next(iter(user.find("Score input").elements))

            draft = "x" * MAX_TECHNICAL_CHARACTERS
            user.find("Original text").type(draft)
            assert transform.enabled
            await user.should_not_see("Remove 1 character to edit this text.")

            user.find("Original text").type("y")
            await user.should_see("Remove 1 character to edit this text.")
            assert original.value == draft + "y"
            assert not transform.enabled
            assert score.enabled

            user.find("Original text").type("z")
            await user.should_see("Remove 2 characters to edit this text.")
            user.find("Score input").click()
            await user.should_see("Input scored", retries=20)
            assert service.score_calls == [draft + "yz"]
            assert not transform.enabled

            user.find("Editing method").click()
            user.find("Local editing rules").click()
            assert original.value == draft + "yz"
            assert transform.enabled
            await user.should_not_see("Remove 2 characters to edit this text.")

    asyncio.run(exercise())


def test_refusal_retains_original_and_shows_reason_and_attempted_candidate():
    async def exercise():
        service = FixtureService(refused=True)
        async with user_simulation(lambda: _build_page(service)) as user:
            await user.open("/")
            original = "Therefore, the team prepared a report."
            user.find("Original text").type(original)
            user.find("Transform text").click()
            await user.should_see("Original retained", retries=20)
            assert next(iter(user.find("Accepted text").elements)).value == original
            await user.should_see("Fixture meaning refusal")
            await user.should_see("Review alternatives")
            user.find("Review alternatives").click()
            await user.should_see("Attempted candidate")
            user.find("Attempted candidate").click()
            await user.should_see("Refused candidate diff")
            await user.should_see(
                kind=ui.html,
                content='<del>Therefore</del><ins class="diff-replacement">Thus</ins>'
                ", the team prepared a report.",
            )
            await user.should_not_see("Accepted output diff")
            await user.should_see("Thus, the team prepared a report.")
            user.find(kind=ui.button, content="Use as draft").click()
            assert next(iter(user.find("Original text").elements)).value == (
                "Thus, the team prepared a report."
            )
            assert next(iter(user.find("Accepted text").elements)).value == ""
            user.find("Undo").click()
            assert next(iter(user.find("Original text").elements)).value == original

    asyncio.run(exercise())


def test_frozen_rules_explain_unchanged_text():
    async def exercise():
        async with user_simulation(lambda: _build_page(FixtureService())) as user:
            await user.open("/")
            original = "The team prepared a report."
            user.find("Original text").type(original)
            user.find("Transform text").click()
            await user.should_see("No changes made", retries=20)
            await user.should_see("The local rules made no edits to this passage.")
            await user.should_not_see("Change refused")
            await user.should_not_see("Accepted output diff")
            assert next(iter(user.find("Accepted text").elements)).value == original

    asyncio.run(exercise())


def test_rule_switches_control_output_and_ranking_tracks_input():
    class RankingService(FixtureService):
        def rank_rules(self, text, *, include_catalog):
            assert include_catalog
            assert text == "Therefore, the team prepared a brief report."
            return [
                {
                    "rule": rule,
                    "rank": 1 if index == 0 else None,
                    "mean_delta": -0.125 if index == 0 else None,
                    "status": "ranked" if index == 0 else "unchanged",
                }
                for index, rule in enumerate(
                    reversed(catalog_rule_names(self.manifest["rules"]))
                )
            ]

    async def exercise():
        async with user_simulation(lambda: _build_page(RankingService())) as user:
            await user.open("/")
            switches = sorted(user.find(ui.switch).elements, key=lambda item: item.id)
            assert len(switches) == len(RULES)
            assert {switch.text for switch in switches if switch.value} == {
                RULE_LABELS[rule] for rule in FixtureService.manifest["rules"]
            }
            rank_button = next(
                iter(user.find(kind=ui.button, content="Rank rules for input").elements)
            )
            assert not rank_button.enabled
            await user.should_see("Example diff")
            await user.should_see(kind=ui.html, content="<del>Therefore</del>")
            user.find("Original text").type(
                "Therefore, the team prepared a brief report."
            )
            assert rank_button.enabled
            user.find("Transform text").click()
            await user.should_see("Changes accepted", retries=20)
            accepted = next(iter(user.find("Accepted text").elements))
            assert accepted.value == "Thus, the team prepared a short report."

            user.find(kind=ui.button, content="Rank rules for input").click()
            await user.should_see("Rule ranking complete", retries=20)
            await user.should_see("#1 · -0.1250 mean change")
            await user.should_see("No edit on this input")
            switches[0].set_value(False)
            assert accepted.value == ""
            await user.should_see("#1 · -0.1250 mean change")
            user.find("Transform text").click()
            await user.should_see("Changes accepted", retries=20)
            assert accepted.value == "Therefore, the team prepared a short report."

            for switch in switches:
                switch.set_value(False)
            user.find("Transform text").click()
            await user.should_see("No changes made", retries=20)
            assert accepted.value == "Therefore, the team prepared a brief report."
            user.find("Original text").type(" The ranking now needs a refresh.")
            await user.should_not_see("#1 · -0.1250 mean change")

    asyncio.run(exercise())


def test_custom_rules_group_for_display_without_changing_pipeline_order():
    configured_rules = [
        "lex_a_brief_short",
        "lex_initial_however_nevertheless",
        "contractions",
        "two_sentence_paragraphs",
        "normalize_spaces",
        "lex_also_known_as",
    ]
    catalog_rules = catalog_rule_names(configured_rules)
    ranked_rules = list(reversed(catalog_rules))
    headings = [
        "Word and phrase swaps",
        "Transitions and framing",
        "Contractions and abbreviations",
        "Sentence and paragraph structure",
        "Punctuation and formatting",
    ]
    grouped_order = [
        item
        for category in headings
        for item in [
            category,
            *(
                RULE_LABELS[rule]
                for rule in catalog_rules
                if rule_category(rule) == category
            ),
        ]
    ]

    class GroupedService(FixtureService):
        manifest = {**FixtureService.manifest, "rules": configured_rules}

        def rank_rules(self, text, *, include_catalog):
            assert include_catalog
            return [
                {
                    "rule": rule,
                    "rank": index,
                    "mean_delta": -0.001 * (len(ranked_rules) - index + 1),
                    "status": "ranked",
                }
                for index, rule in enumerate(ranked_rules, start=1)
            ]

    async def exercise():
        service = GroupedService()
        async with user_simulation(lambda: _build_page(service)) as user:
            await user.open("/")
            first_heading = next(iter(user.find(headings[0]).elements))
            rule_list = first_heading.parent_slot.parent

            def visible_rule_order():
                return [
                    element.text
                    for element in rule_list.descendants()
                    if element.visible
                    and (
                        isinstance(element, ui.switch)
                        or isinstance(element, ui.label)
                        and element.text in headings
                    )
                ]

            assert visible_rule_order() == grouped_order
            disabled = next(iter(user.find(RULE_LABELS["contractions"]).elements))
            disabled.set_value(False)
            user.find("Original text").type(
                "However, this is not final. It is also known as a brief report."
            )
            user.find(kind=ui.button, content="Rank rules for input").click()
            await user.should_see("Rule ranking complete", retries=20)
            assert visible_rule_order() == [RULE_LABELS[rule] for rule in ranked_rules]

            user.find("Transform text").click()
            await user.should_see("Changes accepted", retries=20)
            assert service.enabled_rules == [
                rule for rule in configured_rules if rule != "contractions"
            ]

            user.find("Original text").type(" The ranking needs a refresh.")
            assert visible_rule_order() == grouped_order
            assert not disabled.value

    asyncio.run(exercise())


@pytest.mark.parametrize("configured_rules", [[], ["contractions"]])
def test_catalog_categories_are_available_with_sparse_or_empty_defaults(
    configured_rules,
):
    class SparseRulesService(FixtureService):
        manifest = {**FixtureService.manifest, "rules": configured_rules}

    async def exercise():
        async with user_simulation(lambda: _build_page(SparseRulesService())) as user:
            await user.open("/")
            for category in RULE_CATEGORIES:
                await user.should_see(kind=ui.label, content=category)
            switches = user.find(ui.switch).elements
            assert len(switches) == len(RULES)
            assert {switch.text for switch in switches if switch.value} == {
                RULE_LABELS[rule] for rule in configured_rules
            }
            user.find("Original text").type("A passage to inspect.")
            assert next(
                iter(user.find(kind=ui.button, content="Rank rules for input").elements)
            ).enabled

    asyncio.run(exercise())


def test_additional_catalog_rule_can_be_enabled_and_transformed():
    async def exercise():
        service = FixtureService()
        async with user_simulation(lambda: _build_page(service)) as user:
            await user.open("/")
            contraction = next(iter(user.find(RULE_LABELS["contractions"]).elements))
            assert not contraction.value
            user.find(RULE_LABELS["contractions"]).click()
            user.find("Original text").type(
                "Therefore, the team did not send the report."
            )
            user.find("Transform text").click()
            await user.should_see("Changes accepted", retries=20)
            assert next(iter(user.find("Accepted text").elements)).value == (
                "Thus, the team didn't send the report."
            )
            assert service.enabled_rules == [
                *service.manifest["rules"],
                "contractions",
            ]

    asyncio.run(exercise())


def test_rule_filters_hide_empty_headings_and_preserve_hidden_selections():
    async def exercise():
        service = FixtureService()
        async with user_simulation(lambda: _build_page(service)) as user:
            await user.open("/")
            user.find(RULE_LABELS["contractions"]).click()
            user.find("Find a rule").type("Contract negative auxiliaries")
            assert {item.text for item in user.find(ui.switch).elements} == {
                RULE_LABELS["contractions"]
            }
            await user.should_see(
                kind=ui.label, content="Contractions and abbreviations"
            )
            for category in RULE_CATEGORIES:
                if category != "Contractions and abbreviations":
                    await user.should_not_see(kind=ui.label, content=category)

            user.find("Rule category").click()
            user.find("Punctuation and formatting").click()
            await user.should_see("No rules match these filters.")
            await user.should_not_see(ui.switch)
            for category in RULE_CATEGORIES:
                await user.should_not_see(kind=ui.label, content=category)

            user.find("Find a rule").clear()
            await user.should_not_see("No rules match these filters.")
            assert {item.text for item in user.find(ui.switch).elements} == {
                RULE_LABELS[rule]
                for rule in RULES
                if rule_category(rule) == "Punctuation and formatting"
            }
            user.find("Selected rules only").click()
            await user.should_see("No rules match these filters.")
            user.find("Rule category").click()
            user.find("All categories").click()
            assert {item.text for item in user.find(ui.switch).elements} == {
                *(RULE_LABELS[rule] for rule in service.manifest["rules"]),
                RULE_LABELS["contractions"],
            }

            user.find("Find a rule").type("no matching rule exists")
            await user.should_see("No rules match these filters.")
            user.find("Original text").type(
                "Therefore, the team did not send the report."
            )
            user.find("Transform text").click()
            await user.should_see("Changes accepted", retries=20)
            assert next(iter(user.find("Accepted text").elements)).value == (
                "Thus, the team didn't send the report."
            )
            user.find("Find a rule").clear()
            assert all(switch.value for switch in user.find(ui.switch).elements)
            user.find("Selected rules only").click()
            assert len(user.find(ui.switch).elements) == len(RULES)

    asyncio.run(exercise())


def test_live_impact_labels_filter_results_and_reset_to_recorded_evidence():
    catalog_rules = catalog_rule_names(FixtureService.manifest["rules"])
    levels = [
        (-0.1, "Large reduction"),
        (-0.03, "Moderate reduction"),
        (-0.005, "Small reduction"),
        (0.0005, "Negligible change"),
        (0.01, "Score increase"),
    ]

    class ImpactService(FixtureService):
        def rank_rules(self, text, *, include_catalog):
            assert include_catalog
            return [
                {
                    "rule": rule,
                    "rank": index + 1 if index < len(levels) else None,
                    "mean_delta": levels[index][0] if index < len(levels) else None,
                    "detectors": {"fixture": {"delta": levels[index][0]}}
                    if index < len(levels)
                    else {},
                    "status": "ranked"
                    if index < len(levels)
                    else "refused"
                    if index == 5
                    else "no_comparable_scores"
                    if index == 6
                    else "unchanged",
                }
                for index, rule in enumerate(catalog_rules)
            ]

    async def exercise():
        async with user_simulation(lambda: _build_page(ImpactService())) as user:
            await user.open("/")
            await user.should_see("Recorded ·")
            await user.should_not_see(kind=ui.badge, content="Not yet measured")
            await user.should_see(kind=ui.html, content="AI benchmark · mean change")
            await user.should_see(
                kind=ui.html, content="Matching example · refused candidate"
            )
            impact_filter = next(iter(user.find("Detector impact").elements))
            recorded_options = impact_filter.options.copy()
            assert recorded_options["All impacts"] == f"All impacts ({len(RULES)})"
            user.find("Original text").type("Therefore, the report is ready.")
            user.find(kind=ui.button, content="Rank rules for input").click()
            await user.should_see("Rule ranking complete", retries=20)
            for rule, (_, label) in zip(catalog_rules, levels):
                switch = next(iter(user.find(RULE_LABELS[rule]).elements))
                badges = {
                    item.text
                    for item in switch.parent_slot.parent.descendants()
                    if isinstance(item, ui.badge) and item.visible
                }
                assert f"On this input · {label}" in badges
            await user.should_see("No edit on this input")
            await user.should_see("Preservation check refused")
            await user.should_see("Comparable scores unavailable")
            await user.should_not_see("Recorded ·")
            await user.should_see(kind=ui.html, content="Current input · score change")
            await user.should_see(kind=ui.html, content="-10.0000 pp")
            await user.should_not_see(
                kind=ui.html, content="AI benchmark · mean change"
            )

            assert impact_filter.options["At least a small reduction"] == (
                "At least a small reduction (3)"
            )
            assert impact_filter.options["At least a moderate reduction"] == (
                "At least a moderate reduction (2)"
            )
            impact_filter.set_value("At least a small reduction")
            assert {item.text for item in user.find(ui.switch).elements} == {
                RULE_LABELS[rule] for rule in catalog_rules[:3]
            }
            user.find("Selected rules only").click()
            assert impact_filter.options["Score increase"] == "Score increase (0)"
            assert impact_filter.options["All impacts"] == "All impacts (4)"
            user.find("Selected rules only").click()
            assert impact_filter.options["Score increase"] == "Score increase (1)"
            impact_filter.set_value("Small reduction")
            assert {item.text for item in user.find(ui.switch).elements} == {
                RULE_LABELS[catalog_rules[2]]
            }
            impact_filter.set_value("Preservation check refused")
            assert {item.text for item in user.find(ui.switch).elements} == {
                RULE_LABELS[catalog_rules[5]]
            }
            impact_filter.set_value("At least a moderate reduction")
            user.find("Original text").type(" A changed passage needs a new ranking.")
            assert impact_filter.value == "At least a moderate reduction"
            assert impact_filter.options == recorded_options
            impact_filter.set_value("All impacts")
            await user.should_not_see("On this input ·")
            await user.should_not_see("#1 · -0.1000 mean change")
            await user.should_see("Recorded ·")
            await user.should_not_see(
                kind=ui.html, content="Current input · score change"
            )
            await user.should_see(kind=ui.html, content="AI benchmark · mean change")
            await user.should_not_see(kind=ui.badge, content="Not yet measured")
            assert len(user.find(ui.switch).elements) == len(RULES)
            for category in RULE_CATEGORIES:
                await user.should_see(kind=ui.label, content=category)

    asyncio.run(exercise())


def test_failed_transformation_shows_error_and_restores_controls():
    class FailingOnceService(FixtureService):
        def __init__(self):
            super().__init__()
            self.transform_calls = 0

        def inspect_transformation(self, text, *, enabled_rules, include_catalog):
            self.transform_calls += 1
            if self.transform_calls == 1:
                raise RuntimeError(
                    "The local meaning check is temporarily unavailable."
                )
            return super().inspect_transformation(
                text, enabled_rules=enabled_rules, include_catalog=include_catalog
            )

    async def exercise():
        service = FailingOnceService()
        async with user_simulation(lambda: _build_page(service)) as user:
            await user.open("/")
            original = "Therefore, this can be retried."
            user.find("Original text").type(original)
            user.find("Transform text").click()
            await user.should_see(
                "Processing failed. Your draft is still here. Try again.", retries=20
            )
            await user.should_not_see(
                "The local meaning check is temporarily unavailable."
            )
            assert next(iter(user.find("Transform text").elements)).enabled
            assert next(iter(user.find("Original text").elements)).value == original
            assert not next(iter(user.find("Copy output").elements)).enabled
            assert not next(iter(user.find("Score output").elements)).enabled
            user.find("Transform text").click()
            await user.should_see("Changes accepted", retries=20)
            assert service.transform_calls == 2
            assert next(iter(user.find("Accepted text").elements)).value == (
                "Thus, this can be retried."
            )

    asyncio.run(exercise())


def test_busy_processing_disables_edits_and_scoring():
    entered = threading.Event()
    release = threading.Event()

    class BlockingService(FixtureService):
        def inspect_transformation(self, text, *, enabled_rules, include_catalog):
            entered.set()
            if not release.wait(timeout=5):
                raise TimeoutError("The test did not release processing")
            return super().inspect_transformation(
                text, enabled_rules=enabled_rules, include_catalog=include_catalog
            )

    async def exercise():
        service = BlockingService()
        async with user_simulation(lambda: _build_page(service)) as user:
            await user.open("/")
            user.find("Original text").type("Therefore, the passage stays stable.")
            user.find("Transform text").click()
            try:
                await user.should_see("Checking edits and meaning", retries=20)
                assert entered.wait(timeout=1)
                original = next(iter(user.find("Original text").elements))
                assert not original.enabled
                assert not next(iter(user.find("Score input").elements)).enabled
                assert not next(
                    iter(
                        user.find(
                            kind=ui.button, content="Rank rules for input"
                        ).elements
                    )
                ).enabled
                assert all(
                    not switch.enabled for switch in user.find(ui.switch).elements
                )
                user.find("Original text").type(" Stale edit.")
                user.find("Score input").click()
                assert original.value == "Therefore, the passage stays stable."
                assert service.score_calls == []
            finally:
                release.set()
            await user.should_see("Changes accepted", retries=20)
            assert next(iter(user.find("Transform text").elements)).enabled

    asyncio.run(exercise())


def test_clipboard_buttons_preserve_text_and_handle_denied_access():
    async def exercise():
        async with user_simulation(lambda: _build_page(FixtureService())) as user:
            await user.open("/")
            passage = 'Quoted "text" with a newline.\n<script>literal text</script>'
            copied = []
            user.javascript_rules[
                re.compile(r"try \{ return await navigator\.clipboard\.readText\(\);")
            ] = lambda _: passage
            user.javascript_rules[
                re.compile(
                    r"try \{ await navigator\.clipboard\.writeText\((.*)\); return true;",
                    re.DOTALL,
                )
            ] = lambda match: copied.append(json.loads(match[1])) or True
            user.find(kind=ui.button, content="Paste").click()
            await user.should_see(passage, retries=20)
            user.find("Copy input").click()
            await user.should_see("Copied to clipboard.", retries=20)
            assert copied == [passage]
            user.javascript_rules[
                re.compile(r"try \{ return await navigator\.clipboard\.readText\(\);")
            ] = lambda _: None
            user.find(kind=ui.button, content="Paste").click()
            await user.should_see("Allow clipboard access", retries=20)

    asyncio.run(exercise())


@pytest.mark.parametrize(
    ("option", "strategy"),
    [
        ("LLM sentence editing", "adaptive"),
        ("LLM technical rewrite (experimental)", "technical-update-experimental"),
    ],
)
def test_llm_workflow_routes_request_and_shows_paired_scores(option, strategy):
    class RewriteService(FixtureService):
        def __init__(self):
            super().__init__()
            self.rewrite_calls = []

        def score_text(self, text, *, detailed):
            report = super().score_text(text, detailed=detailed)
            report["detectors"] = {"fixture": {"window_scoring": {"mean": 0.6}}}
            return report

        def rewrite_text(self, text, *, strategy, rounds):
            self.rewrite_calls.append((text, strategy, rounds))
            return {
                "text": text.replace("Therefore", "Thus"),
                "changed": True,
                "status": "changed",
                "scores": {
                    "method": "original-word-weighted-section-mean",
                    "detectors": {
                        "fixture": {"original": 0.8, "rewritten": 0.6, "delta": -0.2}
                    },
                },
                "final_review": {"passed": True},
            }

    async def exercise():
        service = RewriteService()
        async with user_simulation(lambda: _build_page(service)) as user:
            await user.open("/")
            await user.should_not_see("LLM rewriting sends your text")
            await user.should_not_see("How sentence editing works")
            await user.should_not_see("How technical rewriting works")
            user.find("Editing method").click()
            user.find(option).click()
            await user.should_see("LLM rewriting sends your text")
            await user.should_not_see("Local rules")
            guide_title = (
                "How sentence editing works"
                if strategy == "adaptive"
                else "How technical rewriting works"
            )
            await user.should_see(guide_title)
            await user.should_see("Illustrative example")
            await user.should_see(kind=ui.html, content="<del>")
            await user.should_see(kind=ui.html, content="<ins")
            user.find("Use example as input").click()
            example_input = next(iter(user.find("Original text").elements)).value
            assert example_input.startswith(
                "The team carried out"
                if strategy == "adaptive"
                else "The new release introduces"
            )
            assert next(iter(user.find("Accepted text").elements)).value == ""
            assert service.rewrite_calls == []
            assert service.score_calls == []
            user.find("Clear").click()
            user.find("Load example").click()
            assert (
                next(iter(user.find("Original text").elements)).value == example_input
            )
            user.find("Clear").click()
            if strategy == "adaptive":
                user.find("Rewrite rounds").click()
                user.find("1 round").click()
            original = "Therefore, the team prepared a report."
            user.find("Original text").type(original)
            user.find("Transform text").click()
            await user.should_see("Changes accepted", retries=20)
            assert service.rewrite_calls == [
                (original, strategy, 1 if strategy == "adaptive" else 3)
            ]
            assert service.score_calls == []
            assert next(iter(user.find("Accepted text").elements)).value.startswith(
                "Thus"
            )
            await user.should_see("Accepted output diff")
            table = next(iter(user.find(ui.table).elements))
            assert table.rows[0]["input"] == "0.8000 (section mean)"
            assert table.rows[0]["output"] == "0.6000 (section mean)"
            assert table.rows[0]["change"] == "-0.2000"
            await user.should_see("These scores combine document sections")
            user.find("Score output").click()
            await user.should_see("Output scored", retries=20)
            await user.should_see("Window means average scores")
            table = next(iter(user.find(ui.table).elements))
            assert table.rows[0]["output"] == "0.6000 (window mean)"
            assert table.rows[0]["change"] == "—"
            user.find("Editing method").click()
            user.find("Local editing rules").click()
            await user.should_not_see("LLM rewriting sends your text")
            await user.should_not_see(guide_title)
            await user.should_not_see("Illustrative example")
            await user.should_see("Local rules")
            await user.should_not_see(ui.table)
            assert next(iter(user.find("Accepted text").elements)).value == ""

    asyncio.run(exercise())


def test_yolo_runs_the_combined_method_and_shows_stage_results():
    class YoloService(FixtureService):
        def run_yolo(self, text, *, rounds):
            self.yolo_call = (text, rounds)
            edited = "Therefore, the team sent the report."
            return {
                "text": "Thus, the team sent the report.",
                "stages": {
                    "adaptive": {"status": "changed", "changed": True, "text": edited},
                    "technical-update-experimental": {
                        "status": "unchanged_no_qualified_improvement",
                        "changed": False,
                        "text": edited,
                    },
                    "frozen": {
                        "refused": False,
                        "text": "Thus, the team sent the report.",
                    },
                },
            }

    async def exercise():
        service = YoloService()
        async with user_simulation(lambda: _build_page(service)) as user:
            await user.open("/")
            user.find("Editing method").click()
            user.find("YOLO · Run all three").click()
            await user.should_see("Transform limit 24,000")
            await user.should_see("LLM rewriting sends your text")
            await user.should_not_see("Local rules")
            user.find("Sentence editing rounds").click()
            user.find("1 round").click()
            original = "Therefore, the team dispatched the report."
            user.find("Original text").type(original)
            user.find("Transform text").click()
            await user.should_see("Changes accepted", retries=20)
            assert service.yolo_call == (original, 1)
            assert next(iter(user.find("Accepted text").elements)).value == (
                "Thus, the team sent the report."
            )
            await user.should_see("YOLO stage results")
            await user.should_see("LLM sentence editing · Edits kept")
            await user.should_see("No rewrite qualified")
            await user.should_see("Frozen rules · Edits kept")
            await user.should_see("Accepted output diff")
            await user.should_not_see(ui.table)

    asyncio.run(exercise())


def test_llm_refusal_retains_original_and_reports_review_reason():
    class RefusedRewrite(FixtureService):
        def rewrite_text(self, text, **kwargs):
            return {
                "text": text,
                "changed": False,
                "status": "reverted_final_review",
                "scores": {
                    "method": "single-section-native",
                    "detectors": {
                        "fixture": {"original": 0.8, "rewritten": 0.8, "delta": 0}
                    },
                },
                "final_review": {
                    "passed": False,
                    "review": {
                        "decision": {"reason": "The revision altered the timing."}
                    },
                },
            }

    async def exercise():
        async with user_simulation(lambda: _build_page(RefusedRewrite())) as user:
            await user.open("/")
            user.find("Editing method").click()
            user.find("LLM sentence editing").click()
            original = "Therefore, the team prepared a report."
            user.find("Original text").type(original)
            user.find("Transform text").click()
            await user.should_see("Original retained", retries=20)
            await user.should_see("The revision altered the timing.")
            assert next(iter(user.find("Accepted text").elements)).value == original
            await user.should_not_see("Attempted candidate")

    asyncio.run(exercise())


@pytest.mark.parametrize(
    "option", ["LLM technical rewrite (experimental)", "YOLO · Run all three"]
)
def test_unselected_rewrite_can_be_reviewed_and_used_as_an_undoable_draft(option):
    source = "The team prepared a report."
    technical_text = "The team assembled the report."
    sentence_edit = "The team put the report together."
    yolo = option.startswith("YOLO")

    class AlternativeService(FixtureService):
        def __init__(self):
            super().__init__()
            self.calls = []

        @staticmethod
        def technical_report(text):
            return {
                "text": text,
                "changed": False,
                "status": "unchanged_no_qualified_improvement",
                "strategy": "technical_update",
                "selected_model": None,
                "scores": {"method": "single-section-native", "detectors": {}},
                "candidates": [
                    {
                        "model": "fixture/editor",
                        "text": "The team assembled the PROTECTED_FRAGMENT.",
                        "formatted_text": technical_text,
                        "acceptable": True,
                        "disposition": "accepted",
                        "format_issues": [],
                        "mechanical": {"passed": True, "issues": []},
                        "resolution": {"accepted": True},
                    },
                    {
                        "model": "fixture/broken",
                        "text": "The team broke the formatting.",
                        "acceptable": False,
                        "disposition": "format_refusal",
                        "format_issues": ["protected_markup_changed"],
                    },
                ],
            }

        def rewrite_text(self, text, *, strategy, rounds):
            self.calls.append((strategy, text, rounds))
            return self.technical_report(text)

        def run_yolo(self, text, *, rounds):
            self.calls.append(("yolo", text, rounds))
            return {
                "text": text,
                "stages": {
                    "adaptive": {
                        "text": text,
                        "attempted_text": sentence_edit,
                        "changed": False,
                        "status": "reverted_final_review",
                        "final_review": {
                            "passed": False,
                            "mechanical_issues": [],
                            "review": {"decision": {"passed": True}},
                        },
                    },
                    "technical-update-experimental": self.technical_report(text),
                    "frozen": {"text": text, "attempted_text": text, "refused": False},
                },
            }

    async def exercise():
        service = AlternativeService()
        async with user_simulation(lambda: _build_page(service)) as user:
            await user.open("/")
            copied = []
            user.javascript_rules[
                re.compile(
                    r"try \{ await navigator\.clipboard\.writeText\((.*)\); return true;",
                    re.DOTALL,
                )
            ] = lambda match: copied.append(json.loads(match[1])) or True
            user.find("Editing method").click()
            user.find(option).click()
            user.find("Original text").type(source)
            user.find("Transform text").click()
            await user.should_see("Review alternatives", retries=20)
            accepted = next(iter(user.find("Accepted text").elements))
            assert accepted.value == source
            user.find("Review alternatives").click()
            if yolo:
                await user.should_see("LLM sentence editing · Attempted candidate")
            await user.should_not_see(kind=ui.expansion, content="fixture/broken")
            user.find(kind=ui.expansion, content="fixture/editor").click()
            with user.scope(kind=ui.expansion, content="fixture/editor"):
                await user.should_see(technical_text)
                await user.should_see(
                    "Passed preservation review. It did not qualify on detector scores."
                )
                await user.should_not_see("PROTECTED_FRAGMENT")
                user.find("Copy candidate").click()
            await user.should_see("Copied to clipboard.", retries=20)
            assert copied == [technical_text]
            assert accepted.value == source
            with user.scope(kind=ui.expansion, content="fixture/editor"):
                user.find(kind=ui.button, content="Use as draft").click()
            assert (
                next(iter(user.find("Original text").elements)).value == technical_text
            )
            assert accepted.value == ""
            user.find("Undo").click()
            assert next(iter(user.find("Original text").elements)).value == source
            assert accepted.value == ""
            assert service.calls == [
                ("yolo" if yolo else "technical-update-experimental", source, 3)
            ]
            assert service.score_calls == []

    asyncio.run(exercise())


def test_technical_rewrite_shows_provider_model_configuration_error():
    class MissingModelService(FixtureService):
        def rewrite_text(self, text, *, strategy, rounds):
            assert strategy == "technical-update-experimental"
            raise LLMConfigurationError(
                "Missing provider model metadata: example-model"
            )

    async def exercise():
        async with user_simulation(lambda: _build_page(MissingModelService())) as user:
            await user.open("/")
            user.find("Editing method").click()
            user.find("LLM technical rewrite (experimental)").click()
            draft = "Therefore, the team sent the report."
            user.find("Original text").type(draft)
            user.find("Transform text").click()
            await user.should_see("Missing provider model metadata: example-model")
            assert next(iter(user.find("Original text").elements)).value == draft
            assert next(iter(user.find("Transform text").elements)).enabled

    asyncio.run(exercise())


def test_llm_processing_disables_workflow_and_round_controls():
    entered = threading.Event()
    release = threading.Event()

    class BlockingRewrite(FixtureService):
        def rewrite_text(self, text, **kwargs):
            entered.set()
            if not release.wait(timeout=5):
                raise TimeoutError("The test did not release rewriting")
            return {
                "text": text,
                "changed": False,
                "status": "unchanged",
                "scores": {
                    "method": "single-section-native",
                    "detectors": {
                        "fixture": {"original": 0.8, "rewritten": 0.8, "delta": 0}
                    },
                },
                "final_review": {"passed": True},
            }

    async def exercise():
        async with user_simulation(lambda: _build_page(BlockingRewrite())) as user:
            await user.open("/")
            user.find("Editing method").click()
            user.find("LLM sentence editing").click()
            user.find("Original text").type("Therefore, keep the input stable.")
            user.find("Transform text").click()
            try:
                await user.should_see(
                    "Rewriting with the configured LLM provider", retries=20
                )
                assert entered.wait(timeout=1)
                assert not next(iter(user.find("Editing method").elements)).enabled
                assert not next(iter(user.find("Rewrite rounds").elements)).enabled
                assert not next(
                    iter(user.find("Use example as input").elements)
                ).enabled
                assert not next(iter(user.find("Original text").elements)).enabled
            finally:
                release.set()
            await user.should_see("No changes made", retries=20)
            assert next(iter(user.find("Editing method").elements)).enabled

    asyncio.run(exercise())


@pytest.mark.parametrize(
    ("status", "changed"),
    [
        ("provider_error", False),
        ("changed_with_provider_errors", True),
        ("unchanged_with_provider_errors", False),
    ],
)
def test_llm_provider_failures_are_visible_without_claiming_no_changes_needed(
    status, changed
):
    class ProviderFailure(FixtureService):
        def rewrite_text(self, text, **kwargs):
            return {
                "text": text.replace("Therefore", "Thus") if changed else text,
                "changed": changed,
                "status": status,
                "provider_errors": [
                    {"stage": "generation", "type": "http_error", "http_status": 401}
                ],
                "scores": {
                    "method": "single-section-native",
                    "detectors": {
                        "fixture": {"original": 0.8, "rewritten": 0.8, "delta": 0}
                    },
                },
                "final_review": {"passed": True},
            }

    async def exercise():
        async with user_simulation(lambda: _build_page(ProviderFailure())) as user:
            await user.open("/")
            user.find("Editing method").click()
            user.find("LLM sentence editing").click()
            original = "Therefore, keep the team informed."
            user.find("Original text").type(original)
            user.find("Transform text").click()
            if status == "provider_error":
                await user.should_see("Provider request failed", retries=20)
                await user.should_see(
                    "Check the provider configuration and model access"
                )
            else:
                await user.should_see("Some provider requests failed", retries=20)
            await user.should_see("Provider HTTP responses: 401")
            await user.should_not_see("The local rules made no edits to this passage.")
            assert next(iter(user.find("Accepted text").elements)).value == (
                original.replace("Therefore", "Thus") if changed else original
            )

    asyncio.run(exercise())


@pytest.mark.parametrize("option", ["LLM sentence editing", "YOLO · Run all three"])
def test_contextual_policy_is_explicit_and_routed_only_when_selected(option):
    class ContextService(FixtureService):
        def rewrite_text(self, text, **kwargs):
            self.call = kwargs
            return {
                "text": text,
                "status": "unchanged",
                "scores": {"method": "single-section-native", "detectors": {}},
            }

        def run_yolo(self, text, **kwargs):
            self.call = kwargs
            return {
                "text": text,
                "stages": {
                    "adaptive": {"text": text, "status": "unchanged"},
                    "technical-update-experimental": {
                        "text": text,
                        "status": "unchanged",
                    },
                    "frozen": {"text": text, "refused": False},
                },
            }

    async def exercise():
        service = ContextService()
        async with user_simulation(lambda: _build_page(service)) as user:
            await user.open("/")
            await user.should_not_see("Sentence preservation policy")
            user.find("Editing method").click()
            user.find(option).click()
            policy = next(iter(user.find("Sentence preservation policy").elements))
            assert policy.value == "strict"
            user.find("Sentence preservation policy").click()
            user.find("Contextual review (experimental)").click()
            user.find("Original text").type("Keep the release date unchanged.")
            user.find("Transform text").click()
            await user.should_see("No changes made", retries=20)
            assert service.call["review_policy"] == "contextual"
            user.find("Editing method").click()
            user.find("LLM technical rewrite (experimental)").click()
            await user.should_not_see("Sentence preservation policy")
            user.find("Transform text").click()
            await user.should_see("No changes made", retries=20)
            assert "review_policy" not in service.call

    asyncio.run(exercise())


@pytest.mark.parametrize(
    ("review_status", "review_label"),
    [
        ("accepted", "Preservation review passed"),
        ("unverified", "Preservation unverified"),
        ("rejected", "Preservation review refused"),
    ],
)
def test_review_evidence_repair_keeps_the_result_and_original_as_separate_drafts(
    review_status, review_label
):
    source = "Only managers can export reports on Friday."
    candidate = "All employees can export reports on Friday."
    repaired = "On Friday, only managers can export reports."
    finding = {
        "severity": "material",
        "category": "scope",
        "source_quote": "Only managers",
        "candidate_quote": "All employees",
        "explanation": "The draft broadens who can export reports.",
    }

    class RepairService(FixtureService):
        def rewrite_text(self, text, **kwargs):
            return {
                "text": text,
                "status": "unchanged_no_qualified_improvement",
                "scores": {"method": "single-section-native", "detectors": {}},
                "candidates": [
                    {
                        "model": "fixture/editor",
                        "formatted_text": candidate,
                        "acceptable": False,
                        "mechanical": {"passed": True},
                        "resolution": {"status": "reject"},
                        "reviews": [{"status": "reject", "issues": [finding]}],
                    }
                ],
            }

        def repair_candidate(self, original, draft, findings):
            self.repair_call = (original, draft, findings)
            return {
                "text": original,
                "attempted_text": repaired,
                "changed": False,
                "generation_status": "generated",
                "review_status": review_status,
                "review": {"status": "accept", "issues": []},
                "provider_errors": [],
            }

    async def exercise():
        service = RepairService()
        async with user_simulation(lambda: _build_page(service)) as user:
            await user.open("/")
            user.find("Editing method").click()
            user.find("LLM technical rewrite (experimental)").click()
            user.find("Original text").type(source)
            user.find("Transform text").click()
            await user.should_see("Review alternatives", retries=20)
            user.find("Review alternatives").click()
            user.find("fixture/editor").click()
            await user.should_see("Material changes flagged")
            await user.should_see("Source passage")
            await user.should_see("Only managers")
            await user.should_see("All employees")
            user.find("Repair cited issues").click()
            await user.should_see("Repair draft is ready", retries=20)
            assert service.repair_call == (source, candidate, [finding])
            assert next(iter(user.find("Original text").elements)).value == source
            assert next(iter(user.find("Accepted text").elements)).value == source
            assert next(
                iter(
                    user.find(kind=ui.expansion, content="Review alternatives").elements
                )
            ).value
            assert next(
                iter(user.find(kind=ui.expansion, content="Repair draft 1").elements)
            ).value
            with user.scope(kind=ui.expansion, content="Repair draft 1"):
                await user.should_see(review_label)
                await user.should_see(repaired)
                user.find("Use as draft").click()
            assert next(iter(user.find("Original text").elements)).value == repaired
            assert next(iter(user.find("Accepted text").elements)).value == ""
            user.find("Undo").click()
            assert next(iter(user.find("Original text").elements)).value == source

    asyncio.run(exercise())


def test_unverified_or_ungrounded_review_does_not_offer_repair():
    from fakehuman.ui import _result_previews

    finding = {
        "severity": "material",
        "category": "scope",
        "source_quote": "Absent quote",
        "candidate_quote": "The team",
        "explanation": "Unfounded objection.",
    }
    _, rows = _result_previews(
        "The team prepared a report.",
        "The team prepared a report.",
        {
            "adaptive": {
                "candidates": [
                    {
                        "candidate_text": "The team assembled a report.",
                        "candidate_key": "one",
                        "review_status": "unverified",
                        "review": {
                            "evidence": {"status": "unavailable", "issues": [finding]}
                        },
                    }
                ]
            },
        },
    )
    assert rows[0]["review_status"] == "unverified"
    assert rows[0]["findings"] == []


def test_preview_handles_missing_review_and_evidence_spanning_html_markup():
    from fakehuman.ui import _result_previews

    source = "<p>Only <b>managers</b> may export.</p>"
    candidate = "<p>All <b>employees</b> may export.</p>"
    finding = {
        "severity": "material",
        "category": "scope",
        "source_quote": "Only managers",
        "candidate_quote": "All employees",
        "explanation": "The draft broadens export access.",
    }
    _, rows = _result_previews(
        source,
        source,
        {
            "adaptive": {
                "candidates": [
                    {
                        "candidate_text": "<p>Managers export.</p>",
                        "review": None,
                        "mechanical_issues": ["token budget exceeded"],
                    },
                    {
                        "candidate_text": candidate,
                        "review_status": "rejected",
                        "review": {
                            "evidence": {"status": "reject", "issues": [finding]}
                        },
                    },
                ]
            }
        },
    )
    assert rows[0]["review_status"] == "unverified"
    assert rows[0]["blocking_issues"] == ["token budget exceeded"]
    assert rows[1]["findings"] == [finding]


def test_accepted_contextual_output_shows_advisory_evidence():
    class AdvisoryService(FixtureService):
        def rewrite_text(self, text, **kwargs):
            return {
                "text": text.replace("Therefore", "Thus"),
                "status": "changed",
                "scores": {"method": "single-section-native", "detectors": {}},
                "final_review": {
                    "passed": True,
                    "review_status": "accepted",
                    "review": {
                        "decision": {"passed": True},
                        "evidence": {"status": "accept", "issues": []},
                    },
                    "advisory_issues": [
                        "The comparison exceeded the local token limit and was not scored."
                    ],
                },
            }

    async def exercise():
        async with user_simulation(lambda: _build_page(AdvisoryService())) as user:
            await user.open("/")
            user.find("Editing method").click()
            user.find("LLM sentence editing").click()
            user.find("Original text").type("Therefore, the team prepared a report.")
            user.find("Transform text").click()
            await user.should_see("Changes accepted", retries=20)
            user.find(kind=ui.expansion, content="Preservation review").click()
            user.find("Local checks · advisory").click()
            await user.should_see(
                "The comparison exceeded the local token limit and was not scored."
            )
            await user.should_not_see("Review alternatives")

    asyncio.run(exercise())


@pytest.mark.parametrize(
    "option", ["LLM sentence editing", "LLM technical rewrite (experimental)"]
)
@pytest.mark.parametrize(
    ("responses", "expected"),
    [
        (
            [
                {
                    "usage": {
                        "cost": 0,
                        "cost_details": {"upstream_inference_cost": 0.0000929},
                    }
                }
            ],
            ("Reported gateway: $0.00", "Reported upstream: $0.0000929"),
        ),
        (
            [
                {"usage": {"cost": 0.125}},
                {"usage": None, "provider_completion_unknown": True},
            ],
            (
                "Reported gateway: $0.125 (partial · 1 of 2 requests)",
                "Reported upstream: Not reported",
                "1 request has an unknown completion; charges may be missing.",
            ),
        ),
        (
            [{"usage": {}}],
            ("Reported gateway: Not reported", "Reported upstream: Not reported"),
        ),
    ],
)
def test_llm_costs_show_reported_and_missing_amounts_on_refused_runs(
    option, responses, expected
):
    summary = summarize_responses(responses)

    class CostService(FixtureService):
        def rewrite_text(self, text, **kwargs):
            return {
                "text": text,
                "status": "unchanged_no_qualified_improvement",
                "scores": {"method": "single-section-native", "detectors": {}},
                "cost_summary": summary,
            }

    async def exercise():
        async with user_simulation(lambda: _build_page(CostService())) as user:
            await user.open("/")
            await user.should_not_see("Run cost · USD")
            user.find("Editing method").click()
            user.find(option).click()
            user.find("Original text").type("Therefore, the team prepared a report.")
            user.find("Transform text").click()
            await user.should_see(
                "Original retained · no rewrite qualified", retries=20
            )
            await user.should_see("Run cost · USD")
            for label in expected:
                await user.should_see(label)
            count = len(responses)
            await user.should_see(
                f"{count} provider {'request' if count == 1 else 'requests'} · "
                "includes all candidates, reviews and repairs for this run."
            )
            exported = json.loads(next(iter(user.find(ui.code).elements)).content)
            assert exported["cost_summary"] == summary
            user.find("Original text").type(" Another sentence.")
            await user.should_not_see("Run cost · USD")
            await user.should_not_see("Reported gateway:")

    asyncio.run(exercise())


def test_yolo_costs_include_both_remote_stages_once():
    class CostService(FixtureService):
        def run_yolo(self, text, **kwargs):
            return {
                "text": text,
                "stages": {
                    "adaptive": {
                        "text": text,
                        "changed": False,
                        "status": "unchanged_no_qualified_improvement",
                        "cost_summary": summarize_responses(
                            [{"usage": {"cost": 0.01}}, {"usage": {"cost": 0.02}}]
                        ),
                    },
                    "technical-update-experimental": {
                        "text": text,
                        "changed": False,
                        "status": "unchanged_no_qualified_improvement",
                        "cost_summary": summarize_responses(
                            [{"usage": {"cost": 0.04}}]
                        ),
                    },
                    "frozen": {"text": text, "refused": False},
                },
            }

    async def exercise():
        async with user_simulation(lambda: _build_page(CostService())) as user:
            await user.open("/")
            user.find("Editing method").click()
            user.find("YOLO · Run all three").click()
            user.find("Original text").type("The team prepared a report.")
            user.find("Transform text").click()
            await user.should_see("Reported gateway: $0.07", retries=20)
            await user.should_see("3 provider requests · includes all candidates")
            exported = json.loads(next(iter(user.find(ui.code).elements)).content)
            assert exported["cost_summary"]["request_count"] == 3
            assert exported["cost_summary"]["gateway_cost_usd"]["total"] == 0.07

    asyncio.run(exercise())


@pytest.mark.parametrize("generation_status", ["generated", "unavailable"])
def test_repair_costs_accumulate_even_when_no_new_draft_is_generated(generation_status):
    source = "Only managers can export reports."
    candidate = "All employees can export reports."
    finding = {
        "severity": "material",
        "category": "scope",
        "source_quote": "Only managers",
        "candidate_quote": "All employees",
        "explanation": "The candidate broadens export access.",
    }

    class CostService(FixtureService):
        def rewrite_text(self, text, **kwargs):
            return {
                "text": text,
                "status": "unchanged_no_qualified_improvement",
                "scores": {"method": "single-section-native", "detectors": {}},
                "cost_summary": summarize_responses([{"usage": {"cost": 0.01}}]),
                "candidates": [
                    {
                        "model": "fixture/editor",
                        "formatted_text": candidate,
                        "acceptable": False,
                        "mechanical": {"passed": True},
                        "reviews": [{"status": "reject", "issues": [finding]}],
                    }
                ],
            }

        def repair_candidate(self, original, draft, findings):
            assert (original, draft, findings) == (source, candidate, [finding])
            return {
                "text": original,
                "attempted_text": source,
                "generation_status": generation_status,
                "review_status": "accepted",
                "review": {"status": "accept", "issues": []},
                "cost_summary": summarize_responses([{"usage": {"cost": 0.02}}]),
            }

    async def exercise():
        async with user_simulation(lambda: _build_page(CostService())) as user:
            await user.open("/")
            user.find("Editing method").click()
            user.find("LLM technical rewrite (experimental)").click()
            user.find("Original text").type(source)
            user.find("Transform text").click()
            await user.should_see("Reported gateway: $0.01", retries=20)
            user.find("Review alternatives").click()
            user.find("fixture/editor").click()
            user.find("Repair cited issues").click()
            await user.should_see("Reported gateway: $0.03", retries=20)
            await user.should_see("2 provider requests · includes all candidates")
            if generation_status == "generated":
                await user.should_see("Repair draft 1")
            else:
                await user.should_see("Repair could not produce a new draft")
                await user.should_not_see("Repair draft 1")
            exported = json.loads(next(iter(user.find(ui.code).elements)).content)
            assert exported["cost_summary"]["gateway_cost_usd"]["total"] == 0.03
            assert next(iter(user.find("Accepted text").elements)).value == source

    asyncio.run(exercise())
