# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: William Droz <william.droz@idiap.ch>
# SPDX-License-Identifier: GPL-3.0-only

"""Minimal OpenAI-compatible transport for explicitly requested LLM rewrites.

Credentials come from the process environment and never enter request payloads,
checkpoint metadata, exception messages, or child-process arguments.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

DEFAULT_EDITOR_MODEL = "openai/gpt-6-luna"
DEFAULT_REVIEWER_MODEL = "deepseek/deepseek-v4.1-flash"
_MAX_RESPONSE_BYTES = 10 * 1024 * 1024


class LLMConfigurationError(ValueError):
    """A safe configuration error that contains no credential values."""


class LLMModelCatalogError(RuntimeError):
    """A safe error reading or validating the configured provider's catalog."""


@dataclass(frozen=True)
class LLMConfiguration:
    base_url: str
    api_key: str = field(repr=False)


def validate_configuration() -> LLMConfiguration:
    """Validate environment settings without loading models or making requests."""
    if os.environ.get("FAKEHUMAN_OFFLINE", "").strip().lower() in {"1", "true", "yes"}:
        raise LLMConfigurationError("LLM rewriting is unavailable in offline mode")
    base_url = os.environ.get("LLM_BASE_URL", "").strip().rstrip("/")
    api_key = os.environ.get("LLM_API_KEY", "").strip()
    if not base_url or not api_key:
        raise LLMConfigurationError(
            "Set LLM_BASE_URL and LLM_API_KEY for LLM rewriting"
        )
    try:
        parsed = urlsplit(base_url)
        valid = (
            parsed.scheme in {"http", "https"}
            and bool(parsed.hostname)
            and parsed.username is None
            and parsed.password is None
            and not parsed.query
            and not parsed.fragment
        )
        parsed.port  # Validate the optional port without including its value in errors.
    except ValueError:
        valid = False
    if not valid or any(char.isspace() or ord(char) < 32 for char in base_url):
        raise LLMConfigurationError(
            "LLM_BASE_URL must be an HTTP(S) API base URL without credentials, query, or fragment"
        )
    if any(ord(char) < 32 or ord(char) == 127 for char in api_key):
        raise LLMConfigurationError(
            "LLM_API_KEY contains unsupported control characters"
        )
    return LLMConfiguration(base_url, api_key)


def resolve_model_names(
    editor_model: str | None = None, reviewer_model: str | None = None
) -> tuple[str, str]:
    """Resolve explicit model names before environment values and defaults."""
    values = (
        editor_model
        if editor_model is not None
        else os.environ.get("FAKEHUMAN_EDITOR_MODEL", DEFAULT_EDITOR_MODEL),
        reviewer_model
        if reviewer_model is not None
        else os.environ.get("FAKEHUMAN_REVIEWER_MODEL", DEFAULT_REVIEWER_MODEL),
    )
    if any(
        not isinstance(value, str)
        or not value.strip()
        or len(value) > 200
        or any(char.isspace() or ord(char) < 32 for char in value)
        for value in values
    ):
        raise LLMConfigurationError(
            "Editor and reviewer model names must be nonempty identifiers without whitespace"
        )
    return values


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Avoid forwarding the Authorization header to a redirected endpoint.
        return None


class Client:
    """Issue one provider request per call; ambiguous failures are never retried."""

    def __init__(self, *, timeout_seconds: float = 180) -> None:
        if (
            not isinstance(timeout_seconds, (int, float))
            or isinstance(timeout_seconds, bool)
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be positive and finite")
        self.configuration = validate_configuration()
        self.timeout_seconds = timeout_seconds
        self._opener = build_opener(_NoRedirect())

    def _request(self, suffix: str, body: dict | None = None) -> dict:
        request = Request(
            self.configuration.base_url + suffix,
            data=None
            if body is None
            else json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8"),
            headers={
                "Authorization": "Bearer " + self.configuration.api_key,
                "Content-Type": "application/json",
            },
        )
        with self._opener.open(request, timeout=self.timeout_seconds) as response:
            raw = response.read(_MAX_RESPONSE_BYTES + 1)
        if len(raw) > _MAX_RESPONSE_BYTES:
            raise ValueError("Provider response exceeds the supported size")
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("Provider response must be an object")
        return data

    def models(self) -> dict[str, dict]:
        """Return complete model metadata keyed by ID, without printing headers."""
        try:
            rows = self._request("/models")["data"]
            if not isinstance(rows, list) or any(
                not isinstance(row, dict) or not isinstance(row.get("id"), str)
                for row in rows
            ):
                raise ValueError("Invalid provider model catalog")
            return {row["id"]: row for row in rows}
        except (HTTPError, URLError, OSError, ValueError, KeyError, TypeError):
            raise LLMModelCatalogError(
                "Unable to read the configured provider model catalog"
            ) from None

    def call(self, body: dict) -> dict:
        started = time.perf_counter()
        requested_at = datetime.now(timezone.utc).isoformat()
        try:
            data = self._request("/chat/completions", body)
            choice = data["choices"][0]
            raw_response = choice["message"].get("content") or ""
            if not isinstance(raw_response, str):
                raise ValueError("Provider message content must be text")
            return {
                "raw_response": raw_response,
                "finish_reason": choice.get("finish_reason"),
                "response_model": data.get("model"),
                "response_id": data.get("id"),
                "usage": data.get("usage"),
                "transport_attempts": 1,
                "transport_errors": [],
                "latency_seconds": time.perf_counter() - started,
                "requested_at": requested_at,
            }
        except HTTPError as exc:
            reason, status = "http_error", exc.code
        except (URLError, OSError):
            reason, status = "connection_error", None
        except (ValueError, KeyError, IndexError, TypeError):
            reason, status = "provider_protocol_failure", None
        return {
            "raw_response": "",
            "finish_reason": "transport_error",
            "usage": None,
            "transport_attempts": 1,
            "transport_errors": [{"type": reason, "http_status": status}],
            "latency_seconds": time.perf_counter() - started,
            "requested_at": requested_at,
            "provider_completion_unknown": True,
        }


def main() -> None:
    """Private stdin/stdout protocol used by the hard-deadline subprocess."""
    try:
        body = json.load(sys.stdin)
        if not isinstance(body, dict):
            raise ValueError("Expected an object")
        if sys.argv[1:] == ["--models"]:
            result = Client().models()
        elif not sys.argv[1:]:
            result = Client().call(body)
        else:
            raise ValueError("Unknown child operation")
    except Exception:
        # The parent deliberately discards stderr; do not produce a traceback.
        raise SystemExit(1) from None
    json.dump(result, sys.stdout, ensure_ascii=False, allow_nan=False)


if __name__ == "__main__":
    main()
