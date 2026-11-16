# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: William Droz <william.droz@idiap.ch>
# SPDX-License-Identifier: GPL-3.0-only

"""Opt-in CLI/MCP checks using the real pinned, prefetched checkpoints.

Prefetch with ``uv run --locked --extra cpu fakehuman download-models``, then run
``FAKEHUMAN_RUN_MODEL_TESTS=1 uv run --locked --extra cpu pytest -m integration``.
These checks always run offline; the default test suite skips them without
loading weights or contacting Hugging Face.
"""

import asyncio
import math
import os
import sys

import pytest
from fastmcp import Client
from fastmcp.client.transports import StdioTransport

from fakehuman.detectors import DEFAULT_DETECTOR_NAMES, DETECTOR_SPECS
from fakehuman.meaning import CONTRADICTION_THRESHOLD, ENTAILMENT_THRESHOLD

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("FAKEHUMAN_RUN_MODEL_TESTS") != "1",
        reason="prefetch models and set FAKEHUMAN_RUN_MODEL_TESTS=1 to opt in",
    ),
]


def test_real_cli_models_and_window_scoring_over_stdio(tmp_path):
    original = (
        "Therefore, the team prepared a brief report due to the delay. "
        "However, they did not change the 12 mm component."
    )
    expected = (
        "Thus, the team prepared a short report because of the delay. "
        "Nevertheless, they did not change the 12 mm component."
    )
    markdown = "# **Result**\nSee [paper](https://example.com) &amp; data."

    async def exercise():
        transport = StdioTransport(
            command=sys.executable,
            args=["-m", "fakehuman", "--offline"],
            cwd=str(tmp_path),
            env={
                "FASTMCP_CHECK_FOR_UPDATES": "off",
                "FASTMCP_SHOW_SERVER_BANNER": "false",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "HF_HUB_DISABLE_TELEMETRY": "1",
            },
            keep_alive=False,
        )
        async with asyncio.timeout(180), Client(transport) as client:
            tools = {tool.name for tool in await client.list_tools()}
            assert {"humanify", "inspect_transformation", "score_text"} <= tools

            details = (
                await client.call_tool("inspect_transformation", {"text": original})
            ).data
            assert details["attempted_text"] == expected
            meaning = details["meaning"]
            assert meaning["scope"] == "full_pair"
            for name in ("min_entailment", "max_contradiction"):
                assert math.isfinite(meaning[name])
                assert 0.0 <= meaning[name] <= 1.0
            assert meaning["passed"] is (
                meaning["min_entailment"] >= ENTAILMENT_THRESHOLD
                and meaning["max_contradiction"] <= CONTRADICTION_THRESHOLD
            )
            assert details["refused"] is bool(
                details["quality_issues"] or not meaning["passed"]
            )
            assert details["text"] == (original if details["refused"] else expected)
            transformed = await client.call_tool("humanify", {"text": original})
            assert transformed.data == details["text"]
            assert transformed.content[0].text == details["text"]

            scores = (await client.call_tool("score_text", {"text": markdown})).data
            assert set(scores) == set(DEFAULT_DETECTOR_NAMES)
            for score in scores.values():
                assert isinstance(score, float)
                assert math.isfinite(score)
                assert 0.0 <= score <= 1.0

            scored = (
                await client.call_tool(
                    "score_text", {"text": markdown, "detailed": True}
                )
            ).data
            assert scored["scores"] == scores
            assert set(scored["detectors"]) == set(DEFAULT_DETECTOR_NAMES)
            assert scored["interpretation"]
            for name, report in scored["detectors"].items():
                model = report["model"]
                assert model["model_id"] == DETECTOR_SPECS[name].model_id
                assert model["revision"] == DETECTOR_SPECS[name].revision
                assert model["loaded"] is True
                assert model["trust_remote_code"] is False
                inputs = report["input"]
                assert 0 < inputs["tokens"] <= inputs["max_tokens"] == 512
                assert inputs["truncated"] is False
                assert isinstance(inputs["preprocessing_changed"], bool)
            fakespot = scored["detectors"]["fakespot_roberta"]["input"]
            assert fakespot["prepared_text"] == "Result See paper & data."
            assert fakespot["preprocessing_changed"] is True
            desklib = scored["detectors"]["desklib_deberta"]["input"]
            assert desklib["prepared_text"] == markdown
            assert desklib["preprocessing_changed"] is False

            long_text = "word " * 600 + "The end of the passage."
            windowed = (
                await client.call_tool(
                    "score_text", {"text": long_text, "detailed": True}
                )
            ).data
            for name, details in windowed["detectors"].items():
                assert details["input"]["tokens"] > 512
                assert details["input"]["truncated"] is False
                windows = details["window_scoring"]
                assert windows["token_coverage_complete"] is True
                assert len(windows["windows"]) > 1
                assert all(
                    window["token_count"] <= 512 for window in windows["windows"]
                )
                assert windowed["scores"][name] == windows["mean"]

            again = await client.call_tool("score_text", {"text": markdown})
            assert again.data == scores

    asyncio.run(exercise())
