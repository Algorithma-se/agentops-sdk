"""Every SDK call to the AOS public REST API authenticates with Basic pk:sk.

The server resolves ``Bearer <token>`` as a *public* key with score-only
access, so ``Bearer <secret key>`` (what 0.4.0 sent for ``check_guardrails``
and ``get_file``) is a 401 on every route. These tests capture the request
at the transport layer, so they assert the header that actually goes on the
wire:

- ``requests``: ``HTTPAdapter.send`` is patched and sees the prepared request.
- ``httpx``: the async client gets an ``httpx.MockTransport``.
"""

import asyncio
import base64
import io
import json
from unittest.mock import patch

import httpx
import pytest
import requests

from agentops import AgentOps, AsyncAgentOps, Guardrails
from agentops._env import basic_auth_header

HOST = "http://aos.test"
PK, SK = "pk-lf-test", "sk-lf-test"
BASIC = "Basic " + base64.b64encode(f"{PK}:{SK}".encode()).decode()

GUARDRAIL_RESPONSE = {"action": "allow", "reasons": [], "tags": [], "metadata": {}}
FILE_RESPONSE = {"name": "cfg", "version": 1, "fileName": "cfg.json", "url": "https://signed.example/cfg"}


def test_basic_auth_header_encodes_pk_colon_sk():
    assert basic_auth_header(PK, SK) == {"Authorization": BASIC}


# --------------------------------------------------------------------------
# requests (sync client, Guardrails.from_agent)
# --------------------------------------------------------------------------


class _Wire:
    """Records requests sent through requests' HTTPAdapter."""

    def __init__(self, routes):
        self.routes = routes
        self.sent: list[requests.PreparedRequest] = []

    def send(self, adapter, request, **kwargs):
        self.sent.append(request)
        path = request.path_url.split("?")[0]
        status, body = self.routes[path]
        resp = requests.Response()
        resp.status_code = status
        # `raw` (not `_content`) so streamed downloads can read it too.
        resp.raw = io.BytesIO(json.dumps(body).encode())
        resp.headers["content-type"] = "application/json"
        resp.url = request.url
        resp.request = request
        return resp

    def __enter__(self):
        wire = self

        def fake_send(adapter, request, **kwargs):
            return wire.send(adapter, request, **kwargs)

        self._patch = patch.object(requests.adapters.HTTPAdapter, "send", fake_send)
        self._patch.start()
        return self

    def __exit__(self, *exc):
        self._patch.stop()


@pytest.fixture
def sync_client():
    return AgentOps(host=HOST, public_key=PK, secret_key=SK)


def test_sync_check_guardrails_sends_basic_auth(sync_client):
    with _Wire({"/api/public/guardrails": (200, GUARDRAIL_RESPONSE)}) as wire:
        result = sync_client.check_guardrails("hello", agent_name="support-agent")

    assert result["action"] == "allow"
    (request,) = wire.sent
    assert request.method == "POST"
    assert request.headers["Authorization"] == BASIC
    assert json.loads(request.body) == {"stage": "pre_input", "text": "hello", "agentName": "support-agent"}


def test_sync_get_file_sends_basic_auth(sync_client):
    with _Wire({"/api/public/files": (200, FILE_RESPONSE)}) as wire:
        data = sync_client.get_file("cfg", version=1)

    assert data["fileName"] == "cfg.json"
    (request,) = wire.sent
    assert request.method == "GET"
    assert request.headers["Authorization"] == BASIC


def test_sync_get_file_download_does_not_leak_credentials(sync_client, tmp_path):
    routes = {"/api/public/files": (200, FILE_RESPONSE), "/cfg": (200, {"k": "v"})}
    with _Wire(routes) as wire:
        sync_client.get_file("cfg", download=True, download_dir=str(tmp_path))

    api_call, download = wire.sent
    assert api_call.headers["Authorization"] == BASIC
    # The signed storage URL must not receive the project keys.
    assert "Authorization" not in download.headers


def test_from_agent_sends_basic_auth():
    agent = {"id": "agent-1", "name": "support-agent", "guardrailConfig": {"enabled": True}}
    routes = {"/api/public/agents": (200, {"data": [agent], "meta": {}})}
    client = AgentOps(host=HOST, public_key=PK, secret_key=SK)
    with _Wire(routes) as wire:
        # Only the guardrail-config lookup; the registry policy fetch has its
        # own coverage in test_client_policy.py / test_guardrails_policy.py.
        Guardrails.from_agent(client, "support-agent", load_policy=False)

    (request,) = wire.sent
    assert request.headers["Authorization"] == BASIC


# --------------------------------------------------------------------------
# httpx (async client)
# --------------------------------------------------------------------------


def _async_client(routes):
    sent: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        status, body = routes[request.url.path]
        return httpx.Response(status, json=body)

    client = AsyncAgentOps(host=HOST, public_key=PK, secret_key=SK)
    client._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return client, sent


def test_async_check_guardrails_sends_basic_auth():
    client, sent = _async_client({"/api/public/guardrails": (200, GUARDRAIL_RESPONSE)})

    async def run():
        try:
            return await client.check_guardrails("hello", stage="post_output", use_llm=True)
        finally:
            await client.close()

    result = asyncio.run(run())
    assert result["action"] == "allow"
    (request,) = sent
    assert request.method == "POST"
    assert request.headers["Authorization"] == BASIC
    assert json.loads(request.content) == {"stage": "post_output", "text": "hello", "useLlm": True}


def test_async_get_file_sends_basic_auth():
    client, sent = _async_client({"/api/public/files": (200, FILE_RESPONSE)})

    async def run():
        try:
            return await client.get_file("cfg")
        finally:
            await client.close()

    data = asyncio.run(run())
    assert data["fileName"] == "cfg.json"
    (request,) = sent
    assert request.method == "GET"
    assert request.headers["Authorization"] == BASIC
