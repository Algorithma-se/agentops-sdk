"""check_tool_call / check_tool_result: the tool stages of the guardrails API (#867).

Like test_public_api_auth.py, requests are captured at the transport layer so
the assertions are about what actually goes on the wire.
"""

import asyncio
import base64
import io
import json
from unittest.mock import patch

import httpx
import pytest
import requests

from agentops import AgentOps, AsyncAgentOps

HOST = "http://aos.test"
PK, SK = "pk-lf-test", "sk-lf-test"
BASIC = "Basic " + base64.b64encode(f"{PK}:{SK}".encode()).decode()

BLOCK = {
    "action": "block",
    "reasons": ["egress_internal_address"],
    "tags": ["guardrail:egress_internal_address"],
    "metadata": {"egress": [{"kind": "internal_address", "host": "169.254.169.254", "path": "args.url"}]},
}
CONFIG = {"enabled": True, "preTool": {"enabled": True, "egress": {"enabled": True, "action": "block"}}}


class _Wire:
    def __init__(self, status=200, body=None):
        self.status = status
        self.body = body if body is not None else BLOCK
        self.sent: list[requests.PreparedRequest] = []

    def __enter__(self):
        wire = self

        def fake_send(adapter, request, **kwargs):
            wire.sent.append(request)
            resp = requests.Response()
            resp.status_code = wire.status
            resp.raw = io.BytesIO(json.dumps(wire.body).encode())
            resp.headers["content-type"] = "application/json"
            resp.url = request.url
            resp.request = request
            return resp

        self._patch = patch.object(requests.adapters.HTTPAdapter, "send", fake_send)
        self._patch.start()
        return self

    def __exit__(self, *exc):
        self._patch.stop()


@pytest.fixture
def client():
    return AgentOps(host=HOST, public_key=PK, secret_key=SK)


def test_check_tool_call_posts_the_pre_tool_stage(client):
    args = {"url": "http://169.254.169.254/latest/meta-data/"}
    with _Wire() as wire:
        result = client.check_tool_call("http_get", args, config=CONFIG)

    assert result["action"] == "block"
    (request,) = wire.sent
    assert request.method == "POST"
    assert request.path_url == "/api/public/guardrails"
    assert request.headers["Authorization"] == BASIC
    assert json.loads(request.body) == {
        "stage": "pre_tool",
        "toolName": "http_get",
        "args": args,
        "config": CONFIG,
    }


def test_check_tool_result_posts_the_post_tool_stage(client):
    with _Wire(body={"action": "allow", "reasons": ["indirect_injection_in_tool_result"], "tags": [], "metadata": {}}) as wire:
        result = client.check_tool_result("web_fetch", {"body": "AI assistant, you must now ..."}, agent_name="support-agent")

    assert result["reasons"] == ["indirect_injection_in_tool_result"]
    (request,) = wire.sent
    assert json.loads(request.body) == {
        "stage": "post_tool",
        "toolName": "web_fetch",
        "result": {"body": "AI assistant, you must now ..."},
        "agentName": "support-agent",
    }


def test_explicit_config_wins_over_agent_name(client):
    with _Wire() as wire:
        client.check_tool_call("t", {}, agent_name="a", config=CONFIG)
    body = json.loads(wire.sent[0].body)
    assert body["config"] == CONFIG
    assert "agentName" not in body


def test_no_config_and_no_agent_sends_neither(client):
    with _Wire() as wire:
        client.check_tool_call("t", {})
    body = json.loads(wire.sent[0].body)
    assert "config" not in body and "agentName" not in body


def test_non_json_dict_payloads_are_passed_through(client):
    with _Wire() as wire:
        client.check_tool_result("t", ["a", 1, None, {"x": [1, 2]}], config=CONFIG)
    assert json.loads(wire.sent[0].body)["result"] == ["a", 1, None, {"x": [1, 2]}]


def test_server_errors_raise(client):
    with _Wire(status=400, body={"message": "Invalid request data"}):
        with pytest.raises(requests.HTTPError):
            client.check_tool_call("t", {}, config=CONFIG)


def _disabled_env(monkeypatch):
    # The SDK disables itself (fail-open no-ops) when its env vars are missing.
    for var in (
        "AGENTOPS_HOST",
        "AGENTOPS_PUBLIC_KEY",
        "AGENTOPS_SECRET_KEY",
        "LANGFUSE_HOST",
        "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_SECRET_KEY",
    ):
        monkeypatch.delenv(var, raising=False)


def test_disabled_sdk_fails_open_without_a_request(monkeypatch):
    _disabled_env(monkeypatch)
    client = AgentOps()
    assert not client.enabled
    with _Wire() as wire:
        assert client.check_tool_call("t", {"url": "http://127.0.0.1/"})["action"] == "allow"
        assert client.check_tool_result("t", "x")["action"] == "allow"
    assert wire.sent == []


def _async_client(status=200, body=None):
    sent: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(status, json=body if body is not None else BLOCK)

    c = AsyncAgentOps(host=HOST, public_key=PK, secret_key=SK)
    c._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return c, sent


def test_async_check_tool_call_and_result():
    c, sent = _async_client()

    async def run():
        try:
            call = await c.check_tool_call("http_get", {"url": "http://10.0.0.5/"}, config=CONFIG)
            result = await c.check_tool_result("web_fetch", "page text", agent_name="a")
            return call, result
        finally:
            await c.close()

    call, result = asyncio.run(run())
    assert call["action"] == "block" and result["action"] == "block"
    first, second = sent
    assert first.headers["Authorization"] == BASIC
    assert json.loads(first.content) == {
        "stage": "pre_tool",
        "toolName": "http_get",
        "args": {"url": "http://10.0.0.5/"},
        "config": CONFIG,
    }
    assert json.loads(second.content) == {
        "stage": "post_tool",
        "toolName": "web_fetch",
        "result": "page text",
        "agentName": "a",
    }


def test_async_disabled_sdk_fails_open(monkeypatch):
    _disabled_env(monkeypatch)
    c = AsyncAgentOps()

    async def run():
        try:
            return await c.check_tool_call("t", {})
        finally:
            await c.close()

    assert asyncio.run(run())["action"] == "allow"
