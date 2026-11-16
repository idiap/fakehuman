# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: William Droz <william.droz@idiap.ch>
# SPDX-License-Identifier: GPL-3.0-only

"""Choose local storage for resumable LLM rewriting runs."""

import os
import uuid
from pathlib import Path


def new_run_directory() -> Path:
    """Return a fresh run path without creating it."""
    state = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local/state"))
    root = Path(os.environ.get("FAKEHUMAN_RUNS_DIR", state / "fakehuman/runs"))
    return root.expanduser().resolve() / uuid.uuid4().hex
