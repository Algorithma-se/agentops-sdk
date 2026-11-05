"""Tests for ``AgentOps.get_policy`` / ``AsyncAgentOps.get_policy`` against the
``GET /api/public/policies/{agentName}?label=`` contract (parsing, 404 →
``PolicyNotFound``, caching with stale-on-error, fallback)."""

from __future__ import annotations

import asyncio
import base64

import httpx
import pytest

from _http_fakes import FakeResponse
from agentops import AgentOps, AgentPolicy, AsyncAgentOps, PolicyNotFound
from agentops.policies import PolicyFetchError

HOST = "https://control.example.test/"
EXPECTED_AUTH = "Basic " + base64.b64encode(b"pk-test:sk-test").decode()

MANIFEST = """
apiVersion: governance.toolkit/v1
name: support-agent-policy
version: "1.0"
default_action: allow
rules:
  - name: deny-drop
    stage: pre_tool
    condition: "tool.value == 'drop_table'"
    action: deny
"""

POLICY_BODY = {
    "agentName": "support-agent",
    "agentEntryId": "agt_123",
    "version": 3,
    "labels": ["production", "latest"],
    "format": "agt-yaml",
    "manifest": MANIFEST,
    "rego": None,
    "source": "registry",
    "updatedAt": "2026-09-16T10:00:00.000Z",
}


class _DummyLangfuse:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def flush(self):
        pass


@pytest.fixture
def no_langfuse_network(monkeypatch):
    import langfuse

    monkeypatch.setattr(langfuse, "Langfuse", _DummyLangfuse)
    monkeypatch.delenv("AGENTOPS_CLOUD_RUN_INVOKER_AUTH", raising=False)


@pytest.fixture
def client(no_langfuse_network, fake_http):
    return AgentOps(host=HOST, public_key="pk-test", secret_key="sk-test")


class TestGetPolicy:
    def test_parses_the_contract_response(self, client, fake_http):
        fake_http.queue(FakeResponse(200, POLICY_BODY))
        policy = client.get_policy("support-agent")

        assert isinstance(policy, AgentPolicy)
        assert policy.agent_name == "support-agent"
        assert policy.version == 3
        assert policy.labels == ["production", "latest"]
        assert policy.manifest == MANIFEST
        assert policy.rego is None
        assert policy.source == "registry"
        assert policy.updated_at == "2026-09-16T10:00:00.000Z"
        assert policy.agent_entry_id == "agt_123"
        assert policy.to_agt_policy_spec() == MANIFEST

        call = fake_http.gets[0]
        assert call.url == "https://control.example.test/api/public/policies/support-agent"
        assert call.params == {"label": "production"}
        assert call.headers == {"Authorization": EXPECTED_AUTH}
        assert call.timeout == 10.0

    def test_label_and_agent_name_are_encoded(self, client, fake_http):
        fake_http.queue(FakeResponse(200, {**POLICY_BODY, "agentName": "team/agent"}))
        client.get_policy("team/agent", label="staging")
        call = fake_http.gets[0]
        assert call.url.endswith("/api/public/policies/team%2Fagent")
        assert call.params == {"label": "staging"}

    def test_404_raises_policy_not_found(self, client, fake_http):
        fake_http.queue(FakeResponse(404, {"error": "not_found", "message": "no policy for support-agent"}))
        with pytest.raises(PolicyNotFound) as info:
            client.get_policy("support-agent")
        assert info.value.agent_name == "support-agent"
        assert info.value.label == "production"
        assert "no policy for support-agent" in str(info.value)

    def test_404_is_not_masked_by_fallback(self, client, fake_http):
        fake_http.queue(FakeResponse(404, {"error": "not_found", "message": "gone"}))
        with pytest.raises(PolicyNotFound):
            client.get_policy("support-agent", fallback="default_action: allow")

    def test_other_errors_raise_or_use_fallback(self, client, fake_http):
        fake_http.queue(FakeResponse(500, text="internal"))
        with pytest.raises(PolicyFetchError) as info:
            client.get_policy("support-agent")
        assert info.value.status_code == 500

        fake_http.queue(ConnectionError("down"))
        policy = client.get_policy("support-agent", fallback="default_action: allow")
        assert policy.source == "fallback" and policy.version == 0
        assert policy.to_agt_policy_spec() == "default_action: allow"

    def test_malformed_body_is_a_fetch_error(self, client, fake_http):
        fake_http.queue(FakeResponse(200, {"agentName": "support-agent"}))  # no manifest
        with pytest.raises(PolicyFetchError, match="manifest"):
            client.get_policy("support-agent")

    def test_caches_for_ttl_and_refetches_when_expired(self, client, fake_http):
        fake_http.queue(FakeResponse(200, POLICY_BODY))
        first = client.get_policy("support-agent")
        second = client.get_policy("support-agent")
        assert first is second and len(fake_http.gets) == 1

        # A different label is a different cache key.
        fake_http.queue(FakeResponse(200, {**POLICY_BODY, "version": 9}))
        assert client.get_policy("support-agent", label="staging").version == 9
        assert client.get_policy("support-agent").version == 3  # production still cached

        # ttl 0 always refetches.
        fake_http.queue(FakeResponse(200, {**POLICY_BODY, "version": 4}))
        refreshed = client.get_policy("support-agent", cache_ttl_seconds=0)
        assert refreshed.version == 4 and len(fake_http.gets) == 3
        assert client.get_policy("support-agent") is refreshed  # refetch replaced the cache entry

    def test_serves_stale_copy_when_refresh_fails(self, client, fake_http, caplog):
        fake_http.queue(FakeResponse(200, POLICY_BODY), FakeResponse(503))
        good = client.get_policy("support-agent")
        with caplog.at_level("WARNING", logger="agentops.policies"):
            stale = client.get_policy("support-agent", cache_ttl_seconds=0)
        assert stale is good
        assert "serving cached version 3" in caplog.text

    def test_clear_policy_cache(self, client, fake_http):
        fake_http.queue(FakeResponse(200, POLICY_BODY), FakeResponse(200, POLICY_BODY))
        client.get_policy("support-agent")
        client.clear_policy_cache()
        client.get_policy("support-agent")
        assert len(fake_http.gets) == 2

    def test_disabled_client_returns_none_or_fallback(self, monkeypatch, fake_http, caplog):
        for var in ("AGENTOPS_HOST", "AGENTOPS_BASE_URL", "AGENTOPS_PUBLIC_KEY", "AGENTOPS_SECRET_KEY"):
            monkeypatch.delenv(var, raising=False)
        client = AgentOps()
        assert client.enabled is False
        assert client.get_policy("support-agent") is None
        assert client.get_policy("support-agent", fallback="default_action: allow").source == "fallback"
        assert fake_http.calls == []

    def test_empty_manifest_cannot_become_a_policy_spec(self):
        with pytest.raises(ValueError):
            AgentPolicy(agent_name="a", version=1, manifest="   ").to_agt_policy_spec()


class TestAsyncGetPolicy:
    def _client(self, handler, no_langfuse_network):
        client = AsyncAgentOps(host=HOST, public_key="pk-test", secret_key="sk-test")
        client._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        return client

    def test_parses_and_caches(self, no_langfuse_network):
        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return httpx.Response(200, json=POLICY_BODY)

        async def run():
            client = self._client(handler, no_langfuse_network)
            first = await client.get_policy("support-agent")
            second = await client.get_policy("support-agent")
            await client.close()
            return first, second

        first, second = asyncio.run(run())
        assert first.version == 3 and first is second
        assert len(seen) == 1
        assert seen[0].url.path == "/api/public/policies/support-agent"
        assert seen[0].url.params["label"] == "production"
        assert seen[0].headers["Authorization"] == EXPECTED_AUTH

    def test_404_raises_policy_not_found(self, no_langfuse_network):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(404, json={"error": "not_found", "message": "nope"})

        async def run():
            client = self._client(handler, no_langfuse_network)
            try:
                await client.get_policy("support-agent")
            finally:
                await client.close()

        with pytest.raises(PolicyNotFound):
            asyncio.run(run())

    def test_fallback_on_transport_failure(self, no_langfuse_network):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("down", request=request)

        async def run():
            client = self._client(handler, no_langfuse_network)
            try:
                return await client.get_policy("support-agent", fallback="default_action: allow")
            finally:
                await client.close()

        assert asyncio.run(run()).source == "fallback"
