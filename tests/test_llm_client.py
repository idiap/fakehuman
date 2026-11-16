# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: William Droz <william.droz@idiap.ch>
# SPDX-License-Identifier: GPL-3.0-only

"""Provider configuration, sanitized failures, and the real subprocess protocol."""

import io
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError, URLError

import pytest

from fakehuman import llm_client
from fakehuman.adaptive_transport import DeadlineClient


@pytest.fixture(autouse=True)
def configured_provider(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "https://provider.example/v1/")
    monkeypatch.setenv("LLM_API_KEY", "PRIVATE-TEST-KEY")
    monkeypatch.delenv("FAKEHUMAN_OFFLINE", raising=False)
    monkeypatch.delenv("FAKEHUMAN_EDITOR_MODEL", raising=False)
    monkeypatch.delenv("FAKEHUMAN_REVIEWER_MODEL", raising=False)


def test_configuration_normalizes_base_and_hides_credentials_in_repr():
    configuration = llm_client.validate_configuration()
    assert configuration.base_url == "https://provider.example/v1"
    assert configuration.api_key == "PRIVATE-TEST-KEY"
    assert "PRIVATE-TEST-KEY" not in repr(configuration)


@pytest.mark.parametrize("missing", ["LLM_BASE_URL", "LLM_API_KEY"])
def test_missing_settings_fail_without_exposing_remaining_values(monkeypatch, missing):
    monkeypatch.delenv(missing)
    with pytest.raises(llm_client.LLMConfigurationError) as error:
        llm_client.validate_configuration()
    assert "PRIVATE-TEST-KEY" not in str(error.value)
    assert "provider.example" not in str(error.value)


@pytest.mark.parametrize("value", ["1", "true", "YES"])
def test_offline_disallows_provider_use_before_request(monkeypatch, value):
    monkeypatch.setenv("FAKEHUMAN_OFFLINE", value)
    with pytest.raises(llm_client.LLMConfigurationError, match="offline"):
        llm_client.Client()


@pytest.mark.parametrize(
    "base",
    [
        "file:///private/path",
        "https://user:PRIVATE-PASSWORD@provider.example/v1",
        "https://provider.example/v1?token=PRIVATE-PASSWORD",
        "https://provider.example/v1#PRIVATE-PASSWORD",
        "https://provider.example:invalid/v1",
        "https://provider.example/v1\nPRIVATE-PASSWORD",
    ],
)
def test_invalid_urls_have_fixed_errors_without_values(monkeypatch, base):
    monkeypatch.setenv("LLM_BASE_URL", base)
    with pytest.raises(llm_client.LLMConfigurationError) as error:
        llm_client.validate_configuration()
    assert "PRIVATE-PASSWORD" not in str(error.value)
    assert "provider.example" not in str(error.value)


def test_control_characters_in_key_are_rejected_without_leaking_value(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "PRIVATE\nKEY")
    with pytest.raises(llm_client.LLMConfigurationError) as error:
        llm_client.validate_configuration()
    assert "PRIVATE" not in str(error.value)


def test_model_resolution_uses_explicit_then_environment_then_defaults(monkeypatch):
    assert llm_client.resolve_model_names() == (
        llm_client.DEFAULT_EDITOR_MODEL,
        llm_client.DEFAULT_REVIEWER_MODEL,
    )
    monkeypatch.setenv("FAKEHUMAN_EDITOR_MODEL", "provider/editor")
    monkeypatch.setenv("FAKEHUMAN_REVIEWER_MODEL", "provider/reviewer")
    assert llm_client.resolve_model_names() == ("provider/editor", "provider/reviewer")
    assert llm_client.resolve_model_names("override/editor", "override/reviewer") == (
        "override/editor",
        "override/reviewer",
    )


@pytest.mark.parametrize("name", ["", "has spaces", "model\nvalue", 1])
def test_invalid_model_names_are_rejected(name):
    with pytest.raises(llm_client.LLMConfigurationError):
        llm_client.resolve_model_names(name, "reviewer")


class Opener:
    def __init__(self, value=None, *, error=None):
        self.value, self.error = value, error
        self.requests = []

    def open(self, request, *, timeout):
        self.requests.append((request, timeout))
        if self.error:
            raise self.error
        return io.BytesIO(json.dumps(self.value).encode())


def test_chat_call_preserves_response_and_sends_key_only_in_header(monkeypatch):
    opener = Opener(
        {
            "choices": [
                {"message": {"content": "Edited prose."}, "finish_reason": "stop"}
            ],
            "model": "provider/editor",
            "id": "response-1",
            "usage": {"total_tokens": 14},
        }
    )
    monkeypatch.setattr(llm_client, "build_opener", lambda *_: opener)
    result = llm_client.Client(timeout_seconds=12).call(
        {"model": "provider/editor", "messages": [{"role": "user", "content": "Café."}]}
    )
    assert result["raw_response"] == "Edited prose."
    assert result["response_id"] == "response-1"
    assert result["usage"] == {"total_tokens": 14}
    request, timeout = opener.requests[0]
    assert request.full_url == "https://provider.example/v1/chat/completions"
    assert request.get_header("Authorization") == "Bearer PRIVATE-TEST-KEY"
    assert json.loads(request.data)["messages"][0]["content"] == "Café."
    assert timeout == 12
    assert "PRIVATE-TEST-KEY" not in request.data.decode()
    assert "PRIVATE-TEST-KEY" not in json.dumps(result)


@pytest.mark.parametrize(
    ("error", "reason", "status"),
    [
        (HTTPError("PRIVATE-URL", 401, "PRIVATE-ERROR", None, None), "http_error", 401),
        (HTTPError("PRIVATE-URL", 429, "PRIVATE-ERROR", None, None), "http_error", 429),
        (URLError("PRIVATE-ERROR"), "connection_error", None),
        (ValueError("PRIVATE-ERROR"), "provider_protocol_failure", None),
    ],
)
def test_provider_failures_are_sanitized_and_never_retried(
    monkeypatch, error, reason, status
):
    opener = Opener(error=error)
    monkeypatch.setattr(llm_client, "build_opener", lambda *_: opener)
    result = llm_client.Client().call({"model": "provider/editor"})
    assert len(opener.requests) == 1
    assert result["finish_reason"] == "transport_error"
    assert result["transport_errors"] == [{"type": reason, "http_status": status}]
    assert result["provider_completion_unknown"] is True
    assert "PRIVATE" not in json.dumps(result)


@pytest.mark.parametrize(
    "value",
    [[], {}, {"choices": []}, {"choices": [{"message": {"content": ["PRIVATE"]}}]}],
)
def test_malformed_provider_responses_fail_closed(monkeypatch, value):
    opener = Opener(value)
    monkeypatch.setattr(llm_client, "build_opener", lambda *_: opener)
    result = llm_client.Client().call({"model": "editor"})
    assert result["finish_reason"] == "transport_error"
    assert "PRIVATE" not in json.dumps(result)


def test_model_catalog_keeps_complete_provider_metadata(monkeypatch):
    row = {
        "id": "provider/editor",
        "pricing": {"input": 1},
        "supported_parameters": ["response_format"],
        "context_length": 32000,
    }
    opener = Opener({"data": [row]})
    monkeypatch.setattr(llm_client, "build_opener", lambda *_: opener)
    assert llm_client.Client().models() == {"provider/editor": row}
    assert opener.requests[0][0].full_url.endswith("/v1/models")


def test_catalog_errors_omit_provider_exception_details(monkeypatch):
    opener = Opener(error=URLError("PRIVATE-ERROR"))
    monkeypatch.setattr(llm_client, "build_opener", lambda *_: opener)
    with pytest.raises(RuntimeError) as error:
        llm_client.Client().models()
    assert "PRIVATE" not in str(error.value)


def test_real_deadline_client_reaches_local_endpoint_outside_checkout(
    monkeypatch, tmp_path
):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append((self.path, self.headers["Authorization"], None))
            data = json.dumps(
                {"data": [{"id": "local/editor", "context_length": 8192}]}
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((self.path, self.headers["Authorization"], request))
            result = {
                "choices": [
                    {
                        "message": {"content": "Local test response."},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"total_tokens": 5},
            }
            data = json.dumps(result).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("LLM_BASE_URL", f"http://127.0.0.1:{server.server_port}/v1")
    monkeypatch.chdir(tmp_path)
    try:
        result = DeadlineClient(timeout_seconds=5).call({"model": "local/editor"})
        catalog = DeadlineClient(timeout_seconds=5).models()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
    assert result["raw_response"] == "Local test response."
    assert requests == [
        ("/v1/chat/completions", "Bearer PRIVATE-TEST-KEY", {"model": "local/editor"}),
        ("/v1/models", "Bearer PRIVATE-TEST-KEY", None),
    ]
    assert catalog == {"local/editor": {"id": "local/editor", "context_length": 8192}}
    assert "PRIVATE-TEST-KEY" not in json.dumps(result)
