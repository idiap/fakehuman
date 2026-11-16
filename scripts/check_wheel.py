# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: William Droz <william.droz@idiap.ch>
# SPDX-License-Identifier: GPL-3.0-only

"""Install a built wheel in a fresh environment and exercise it outside the repo."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PROBE = r"""
import asyncio
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import sys
import threading

from fastmcp import Client
from fastmcp.client.transports import StdioTransport
import torch

import fakehuman
from fakehuman.cli import load_manifest
from fakehuman.research_mcp import make_server
from fakehuman.process import _source_hashes
from fakehuman.technical_pipeline import source_hashes, snapshot_sources
from fakehuman.adaptive_transport import DeadlineClient

environment = Path(sys.argv[1])
assert Path(fakehuman.__file__).is_relative_to(environment), fakehuman.__file__
assert torch.version.cuda is None, torch.version.cuda
manifest = load_manifest()
assert len(manifest["rules"]) == 4
make_server(manifest)
assert "llm_client.py" in _source_hashes()
snapshot_sources(Path.cwd() / "source-snapshot", source_hashes())

class ProviderFixture(BaseHTTPRequestHandler):
    def do_POST(self):
        assert self.path == "/v1/chat/completions"
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        assert body["model"] == "fixture/editor"
        payload = json.dumps({"choices": [{"message": {"content": "Packaged transport OK."}, "finish_reason": "stop"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass

provider = ThreadingHTTPServer(("127.0.0.1", 0), ProviderFixture)
threading.Thread(target=provider.serve_forever, daemon=True).start()
os.environ["LLM_BASE_URL"] = f"http://127.0.0.1:{provider.server_port}/v1"
os.environ["LLM_API_KEY"] = "local-wheel-fixture"
try:
    reply = DeadlineClient(timeout_seconds=10).call({"model": "fixture/editor", "messages": []})
    assert reply["raw_response"] == "Packaged transport OK.", reply
finally:
    provider.shutdown()
    provider.server_close()

async def exercise():
    executable = environment / ("Scripts/fakehuman.exe" if sys.platform == "win32" else "bin/fakehuman")
    transport = StdioTransport(
        command=str(executable), args=[], keep_alive=False,
        env={
            "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
            "FASTMCP_CHECK_FOR_UPDATES": "off", "FASTMCP_SHOW_SERVER_BANNER": "false",
        },
    )
    async with asyncio.timeout(45), Client(transport) as client:
        tools = {tool.name for tool in await client.list_tools()}
        assert {"humanify", "inspect_transformation", "score_text", "rewrite_text"} <= tools, tools
        original = "The component is 12 mm long."
        result = await client.call_tool("humanify", {"text": original})
        assert result.data == original, result
        details = await client.call_tool("inspect_transformation", {"text": original})
        assert details.data["text"] == original and not details.data["refused"], details

asyncio.run(exercise())
print("Installed wheel resources, manifest hashes, CLI and stdio tools: OK")
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path, help="path to the built fakehuman wheel")
    wheel = parser.parse_args().wheel.resolve()
    if not wheel.is_file() or wheel.suffix != ".whl":
        parser.error(f"not a wheel file: {wheel}")
    with tempfile.TemporaryDirectory(prefix="fakehuman-wheel-") as temporary:
        directory = Path(temporary)
        environment = directory / "venv"
        env = dict(os.environ, UV_PROJECT_ENVIRONMENT=str(environment))
        env.setdefault("UV_LINK_MODE", "copy")
        env.pop("PYTHONPATH", None)
        env.pop("VIRTUAL_ENV", None)
        # Resolve dependencies from the project lock without installing its source.
        subprocess.run(
            [
                "uv",
                "sync",
                "--locked",
                "--extra",
                "cpu",
                "--no-dev",
                "--no-install-project",
                "--python",
                sys.executable,
            ],
            cwd=ROOT,
            env=env,
            check=True,
            timeout=300,
        )
        python = environment / (
            "Scripts/python.exe" if os.name == "nt" else "bin/python"
        )
        subprocess.run(
            ["uv", "pip", "install", "--python", str(python), "--no-deps", str(wheel)],
            cwd=directory,
            env=env,
            check=True,
            timeout=60,
        )
        subprocess.run(
            ["uv", "pip", "check", "--python", str(python)],
            cwd=directory,
            env=env,
            check=True,
            timeout=30,
        )
        env.update(
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            FASTMCP_CHECK_FOR_UPDATES="off",
            FASTMCP_SHOW_SERVER_BANNER="false",
        )
        subprocess.run(
            [str(python), "-m", "fakehuman", "--help"],
            cwd=directory,
            env=env,
            check=True,
            timeout=15,
            stdout=subprocess.DEVNULL,
        )
        subprocess.run(
            [str(python), "-c", PROBE, str(environment)],
            cwd=directory,
            env=env,
            check=True,
            timeout=60,
        )


if __name__ == "__main__":
    main()
