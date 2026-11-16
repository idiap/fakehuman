# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: William Droz <william.droz@idiap.ch>
# SPDX-License-Identifier: GPL-3.0-only

"""Keep the browser UI optional and the existing MCP entry point unchanged."""

import os
import subprocess
import sys
import types

import pytest

from fakehuman import cli


def test_ui_help_does_not_import_ui_or_models():
    code = """
import sys
from fakehuman.cli import main
sys.argv = ['fakehuman', 'ui', '--help']
try:
    main()
except SystemExit as exc:
    assert exc.code == 0
assert 'fakehuman.ui' not in sys.modules
assert 'nicegui' not in sys.modules
assert 'torch' not in sys.modules
assert 'transformers' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert "--host" in result.stdout
    assert "--port" in result.stdout


def test_default_cli_still_serves_stdio_without_importing_ui():
    code = """
import importlib.abc
import sys
import types

class NoUI(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in {'fakehuman.ui', 'nicegui'}:
            raise AssertionError(f'Default MCP command imported {fullname}')

sys.meta_path.insert(0, NoUI())
calls = []
torch = types.ModuleType('torch')
torch.set_num_threads = lambda count: calls.append(('threads', count))
sys.modules['torch'] = torch
server_module = types.ModuleType('fakehuman.research_mcp')
server = types.SimpleNamespace(run=lambda **kw: calls.append(('run', kw)))
server_module.make_server = lambda manifest: server
sys.modules['fakehuman.research_mcp'] = server_module
from fakehuman.cli import main
sys.argv = ['fakehuman']
main()
assert ('threads', 4) in calls
assert ('run', {'transport': 'stdio'}) in calls
assert 'fakehuman.ui' not in sys.modules
assert 'nicegui' not in sys.modules
"""
    subprocess.run(
        [sys.executable, "-c", code], check=True, capture_output=True, timeout=15
    )


def test_missing_ui_extra_has_installation_instructions_without_traceback():
    code = """
import sys
sys.modules['nicegui'] = None
from fakehuman.cli import main
sys.argv = ['fakehuman', 'ui']
main()
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 1
    assert "NiceGUI" in result.stderr
    assert "uv sync --extra cpu --extra ui" in result.stderr
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize(
    ("arguments", "host", "port", "offline"),
    [
        (["ui"], "127.0.0.1", 8080, False),
        (["--offline", "ui"], "127.0.0.1", 8080, True),
        (
            ["ui", "--offline", "--host", "0.0.0.0", "--port", "9090"],
            "0.0.0.0",
            9090,
            True,
        ),
    ],
)
def test_cli_launches_ui_with_local_defaults_and_honors_options(
    monkeypatch, tmp_path, arguments, host, port, offline
):
    calls = []
    manifest = {"fixture": True}
    module = types.ModuleType("fakehuman.ui")

    def run_ui(selected_manifest, **options):
        import os

        calls.append((selected_manifest, options))
        if offline:
            assert os.environ["HF_HUB_OFFLINE"] == "1"
            assert os.environ["TRANSFORMERS_OFFLINE"] == "1"

    module.run_ui = run_ui
    monkeypatch.setitem(sys.modules, "fakehuman.ui", module)
    monkeypatch.setattr(cli, "load_manifest", lambda path: manifest)
    monkeypatch.setattr(sys, "argv", ["fakehuman", *arguments])
    monkeypatch.chdir(tmp_path)
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "FASTMCP_CHECK_FOR_UPDATES"):
        monkeypatch.delenv(name, raising=False)
    cli.main()
    assert calls == [(manifest, {"host": host, "port": port})]


def test_cli_loads_dotenv_before_launching_the_workbench(tmp_path):
    (tmp_path / ".env").write_text(
        "FAKEHUMAN_EDITOR_MODEL=fixture/file-editor\n"
        "FAKEHUMAN_REVIEWER_MODEL=fixture/file-reviewer\n",
        encoding="utf-8",
    )
    env = dict(os.environ, FAKEHUMAN_REVIEWER_MODEL="fixture/exported-reviewer")
    env.pop("FAKEHUMAN_EDITOR_MODEL", None)
    env.pop("PYTHON_DOTENV_DISABLED", None)
    code = """
import os
import sys
import types

def run_ui(manifest, **options):
    assert os.environ["FAKEHUMAN_EDITOR_MODEL"] == "fixture/file-editor"
    assert os.environ["FAKEHUMAN_REVIEWER_MODEL"] == "fixture/exported-reviewer"

sys.modules["fakehuman.ui"] = types.SimpleNamespace(run_ui=run_ui)
from fakehuman.cli import main
sys.argv = ["fakehuman", "ui"]
main()
"""
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
        timeout=15,
    )
