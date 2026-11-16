# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: William Droz <william.droz@idiap.ch>
# SPDX-License-Identifier: GPL-3.0-only

"""Bound provider calls by elapsed time, including reads and internal retries.

The child loads the configured credentials itself. A timeout kills and reaps
the local client process; provider completion and billing remain unknown.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from .llm_client import LLMModelCatalogError, validate_configuration


class DeadlineClient:
    def __init__(
        self, *, timeout_seconds: float = 240, root: Path | None = None
    ) -> None:
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be positive and finite")
        # Kept as a compatibility keyword; installed packages need no repository.
        self.timeout_seconds = timeout_seconds

    def call(self, body: dict) -> dict:
        return self._run(body)

    def models(self) -> dict[str, dict]:
        """Read the provider catalog under the same hard wall-clock deadline."""
        value = self._run({}, models=True)
        if value.get("finish_reason") == "transport_error":
            raise LLMModelCatalogError(
                "Unable to read the configured provider model catalog"
            )
        if any(
            not isinstance(name, str) or not isinstance(row, dict)
            for name, row in value.items()
        ):
            raise LLMModelCatalogError("Invalid provider model catalog")
        return value

    def _run(self, body: dict, *, models: bool = False) -> dict:
        validate_configuration()
        started = time.perf_counter()
        requested_at = datetime.now(timezone.utc).isoformat()
        try:
            result = subprocess.run(
                [sys.executable, "-m", "fakehuman.llm_client"]
                + (["--models"] if models else []),
                input=json.dumps(body),
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
            )
            if result.returncode:
                reason = "client_process_failure"
            else:
                value = json.loads(result.stdout)
                if not isinstance(value, dict):
                    raise ValueError("Invalid provider response")
                return value
        except subprocess.TimeoutExpired:
            # subprocess.run kills and reaps its child before raising.
            reason = "wall_clock_deadline"
        except (OSError, ValueError):
            reason = "client_protocol_failure"
        # Never expose child stderr, partial stdout, credentials, or exception
        # text. Do not retry a call with unknown provider completion status.
        return {
            "raw_response": "",
            "finish_reason": "transport_error",
            "usage": None,
            "transport_attempts": None,
            "transport_errors": [{"type": reason, "http_status": None}],
            "latency_seconds": time.perf_counter() - started,
            "requested_at": requested_at,
            "deadline_seconds": self.timeout_seconds,
            "provider_completion_unknown": True,
        }
