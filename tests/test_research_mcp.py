# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: William Droz <william.droz@idiap.ch>
# SPDX-License-Identifier: GPL-3.0-only

import asyncio
import threading

import pytest
from fastmcp import Client

from fakehuman import research_mcp, research_rules
from fakehuman.detectors import fakespot_clean_text


class RejectingGuard:
    def metadata(self):
        return {"type": "fixture"}

    def check_pairs(self, pairs):
        return [{"passed": False, "reason": "fixture semantic refusal"} for _ in pairs]


class FixtureDetector:
    def __init__(self, preprocessing="none"):
        self.preprocessing = preprocessing
        self.last_input_report = []

    def score(self, texts):
        self.last_input_report = [
            {
                "input_index": index,
                "input_characters": len(text),
                "prepared_characters": len(fakespot_clean_text(text))
                if self.preprocessing == "fakespot_clean_text"
                else len(text),
                "preprocessing_changed": self.preprocessing == "fakespot_clean_text"
                and fakespot_clean_text(text) != text,
                "tokens": 8,
                "max_tokens": 512,
                "truncated": False,
            }
            for index, text in enumerate(texts)
        ]
        return [0.5 for _ in texts]

    def metadata(self):
        return {
            "name": "fixture",
            "model_id": "fixture/model",
            "revision": "fixture-revision",
            "preprocessing": self.preprocessing,
            "score_definition": "softmax(logits)[1]",
        }


def test_research_mcp_returns_original_on_guard_refusal_and_has_string_contract():
    original = "The team will utilize this checklist."
    manifest = {
        "schema_version": 2,
        "rules": ["simplify_verbs"],
        "seed": 17,
        "source_hashes": research_rules.source_hashes(),
    }
    server = research_mcp.make_server(
        manifest,
        guard=RejectingGuard(),
        detectors={"first": FixtureDetector(), "second": FixtureDetector()},
    )

    async def exercise():
        async with Client(server) as client:
            tools = {tool.name: tool for tool in await client.list_tools()}
            schema = tools["humanify"].input_schema
            assert schema["properties"]["text"]["type"] == "string"

            transformed = await client.call_tool("humanify", {"text": original})
            assert isinstance(transformed.data, str)
            assert transformed.data == original

            details = await client.call_tool(
                "inspect_transformation", {"text": original}
            )
            assert details.data["refused"] is True
            assert details.data["text"] == original
            assert details.data["attempted_text"] == "The team will use this checklist."
            assert details.data["changes"][0]["rule"] == "simplify_verbs"
            assert details.data["changes"][0]["edits"][0]["before"] == "utilize"
            assert details.data["changes"][0]["edits"][0]["after"] == "use"
            assert details.data["refusal_reasons"] == [
                "The meaning screen refused the candidate: fixture semantic refusal."
            ]
            assert details.data["meaning"] == {
                "passed": False,
                "reason": "fixture semantic refusal",
            }

            invalid = await client.call_tool_mcp("humanify", {"text": 123})
            assert invalid.is_error

    asyncio.run(exercise())


def test_detailed_scores_expose_input_processing_and_model_provenance():
    from fakehuman.cli import load_manifest

    server = research_mcp.make_server(
        load_manifest(),
        guard=RejectingGuard(),
        detectors={
            "cleaned": FixtureDetector("fakespot_clean_text"),
            "unchanged": FixtureDetector(),
        },
    )
    text = "A **brief** report."

    async def exercise():
        async with Client(server) as client:
            scores = await client.call_tool("score_text", {"text": text})
            assert scores.data == {"cleaned": 0.5, "unchanged": 0.5}
            details = (
                await client.call_tool("score_text", {"text": text, "detailed": True})
            ).data
            assert details["scores"] == scores.data
            assert (
                "not calibrated authorship probabilities" in details["interpretation"]
            )
            cleaned = details["detectors"]["cleaned"]
            assert cleaned["model"]["revision"] == "fixture-revision"
            assert cleaned["model"]["preprocessing"] == "fakespot_clean_text"
            assert cleaned["input"] == {
                "input_index": 0,
                "input_characters": len(text),
                "prepared_characters": len("A brief report."),
                "preprocessing_changed": True,
                "tokens": 8,
                "max_tokens": 512,
                "truncated": False,
                "prepared_text": "A brief report.",
            }
            assert details["detectors"]["unchanged"]["input"]["prepared_text"] == text
            assert not details["detectors"]["unchanged"]["input"][
                "preprocessing_changed"
            ]
            invalid = await client.call_tool_mcp(
                "score_text", {"text": text, "detailed": "yes"}
            )
            assert invalid.is_error

    asyncio.run(exercise())


def test_preview_reports_mechanical_refusals_even_if_meaning_passes(monkeypatch):
    class AcceptingGuard:
        def check_pairs(self, pairs):
            return [{"passed": True, "reason": "test fixture"} for _ in pairs]

    monkeypatch.setitem(
        research_rules.RULES,
        "fixture_numeric_damage",
        lambda text, rng: text.replace("12", "13"),
    )
    server = research_mcp.make_server(
        {
            "schema_version": 2,
            "rules": ["fixture_numeric_damage"],
            "seed": 17,
            "source_hashes": research_rules.source_hashes(),
        },
        guard=AcceptingGuard(),
        detectors={"first": FixtureDetector()},
    )

    async def exercise():
        async with Client(server) as client:
            details = (
                await client.call_tool(
                    "inspect_transformation", {"text": "The component is 12 mm long."}
                )
            ).data
            assert details["refused"]
            assert details["text"] == "The component is 12 mm long."
            assert details["attempted_text"] == "The component is 13 mm long."
            assert details["quality_issues"] == [
                "numeric values or their order changed"
            ]
            assert details["refusal_reasons"] == [
                "Mechanical preservation check failed: numeric values or their order changed."
            ]
            assert details["changes"][0]["edits"][0]["before"] == "12"

    asyncio.run(exercise())


@pytest.mark.parametrize("second_detailed", [False, True])
def test_concurrent_score_calls_keep_their_own_input_reports(second_detailed):
    from fakehuman.cli import load_manifest

    first_started = threading.Event()
    second_started = threading.Event()
    release_first = threading.Event()
    release_second = threading.Event()

    class BlockingDetector(FixtureDetector):
        def score(self, texts):
            scores = super().score(texts)
            started, release = (
                (first_started, release_first)
                if texts == ["first"]
                else (second_started, release_second)
            )
            started.set()
            if not release.wait(timeout=5):
                raise TimeoutError("The test did not release the detector")
            return scores

    server = research_mcp.make_server(
        load_manifest(),
        guard=RejectingGuard(),
        detectors={"fixture": BlockingDetector()},
    )

    async def exercise():
        async with asyncio.timeout(10), Client(server) as client:
            first = asyncio.create_task(
                client.call_tool("score_text", {"text": "first", "detailed": True})
            )
            second = None
            try:
                assert await asyncio.to_thread(first_started.wait, 2)
                second = asyncio.create_task(
                    client.call_tool(
                        "score_text",
                        {"text": "much longer second", "detailed": second_detailed},
                    )
                )
                # With no scoring lock, the second inference now overwrites
                # the adapter report while the first inference is paused.
                await asyncio.to_thread(second_started.wait, 0.25)
                release_first.set()
                first_result = (await first).data
                release_second.set()
                second_result = (await second).data
            finally:
                release_first.set()
                release_second.set()
                await asyncio.gather(
                    *[task for task in (first, second) if task is not None],
                    return_exceptions=True,
                )
            first_report = first_result["detectors"]["fixture"]["input"]
            assert first_report["input_characters"] == len("first")
            assert first_report["prepared_text"] == "first"
            if second_detailed:
                second_report = second_result["detectors"]["fixture"]["input"]
                assert second_report["input_characters"] == len("much longer second")
                assert second_report["prepared_text"] == "much longer second"
            else:
                assert second_result == {"fixture": 0.5}

    asyncio.run(exercise())
