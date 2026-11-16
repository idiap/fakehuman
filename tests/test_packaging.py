# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: William Droz <william.droz@idiap.ch>
# SPDX-License-Identifier: GPL-3.0-only

"""Verify portable resources and the CLI's actual stdio string contract."""

import asyncio
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from fastmcp import Client
from fastmcp.client.transports import StdioTransport

import fakehuman


def test_bundled_pipeline_loads_from_relocated_package_and_unrelated_cwd(tmp_path):
    relocated = tmp_path / "relocated install"
    shutil.copytree(
        Path(fakehuman.__file__).parent,
        relocated / "fakehuman",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    unrelated = tmp_path / "unrelated working directory"
    unrelated.mkdir()
    env = dict(os.environ, PYTHONPATH=str(relocated))
    code = """
import json
from pathlib import Path
import fakehuman
from fakehuman.cli import load_manifest
from fakehuman.research_rules import validate_manifest
from fakehuman.rule_catalog import catalog_rule_names
from fakehuman.rule_impacts import recorded_rule_impacts
manifest = load_manifest()
validate_manifest(manifest)
print(json.dumps({"package": str(Path(fakehuman.__file__).parent),
                  "rules": manifest["rules"],
                  "catalog": catalog_rule_names(manifest["rules"]),
                  "impacts": recorded_rule_impacts()}))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=unrelated,
        env=env,
        text=True,
        capture_output=True,
        timeout=45,
        check=True,
    )
    report = json.loads(result.stdout)
    assert Path(report["package"]) == relocated / "fakehuman"
    assert report["rules"] == [
        "lex_therefore_thus",
        "lex_initial_however_nevertheless",
        "lex_due_to",
        "lex_a_brief_short",
    ]
    assert len(report["catalog"]) == 135
    assert report["catalog"][:4] == report["rules"]
    assert set(report["impacts"]) == set(report["catalog"])
    assert len(report["impacts"]) == 135
    assert all(record["n"] == 60 for record in report["impacts"].values())
    assert all(record["human"]["n"] == 60 for record in report["impacts"].values())
    assert all(record["matching_example"] for record in report["impacts"].values())


def test_cli_default_pipeline_serves_real_stdio_tools_from_unrelated_cwd(tmp_path):
    original = (
        "Therefore, the team prepared a brief report due to the delay. "
        "However, they did not change the 12 mm component."
    )
    expected = (
        "Thus, the team prepared a short report because of the delay. "
        "Nevertheless, they did not change the 12 mm component."
    )
    fixture = Path(__file__).with_name("stdio_fixture.py").resolve()

    async def exercise():
        transport = StdioTransport(
            command=sys.executable,
            args=[str(fixture)],
            cwd=str(tmp_path),
            env={
                "FASTMCP_CHECK_FOR_UPDATES": "off",
                "FASTMCP_SHOW_SERVER_BANNER": "false",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
            },
            keep_alive=False,
        )
        async with asyncio.timeout(45), Client(transport) as client:
            tools = {tool.name: tool for tool in await client.list_tools()}
            assert set(tools) == {
                "humanify",
                "inspect_transformation",
                "score_text",
                "rewrite_text",
                "repair_candidate",
            }
            assert (
                tools["humanify"].input_schema["properties"]["text"]["type"] == "string"
            )
            response = await client.call_tool("humanify", {"text": original})
            assert response.data == expected
            assert response.content[0].text == expected
            details = await client.call_tool(
                "inspect_transformation", {"text": original}
            )
            assert details.data["text"] == expected
            assert details.data["attempted_text"] == expected
            assert details.data["refused"] is False
            assert details.data["quality_issues"] == []
            assert details.data["meaning"] == {"passed": True, "reason": "test fixture"}
            assert details.data["refusal_reasons"] == []
            assert [change["rule"] for change in details.data["changes"]] == [
                "lex_therefore_thus",
                "lex_initial_however_nevertheless",
                "lex_due_to",
                "lex_a_brief_short",
            ]
            scores = await client.call_tool("score_text", {"text": original})
            assert scores.data == {"first": 0.25, "second": 0.75}
            detailed_scores = (
                await client.call_tool(
                    "score_text", {"text": original, "detailed": True}
                )
            ).data
            assert detailed_scores["scores"] == scores.data
            assert (
                detailed_scores["detectors"]["first"]["input"]["prepared_text"]
                == original
            )
            assert (
                detailed_scores["detectors"]["first"]["model"]["revision"]
                == "fixture-revision"
            )
            invalid = await client.call_tool_mcp("humanify", {"text": 123})
            assert invalid.is_error
            empty = await client.call_tool_mcp("humanify", {"text": " "})
            assert empty.is_error

    asyncio.run(exercise())
