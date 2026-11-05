"""Tests for tool-call policy on ``Guardrails`` (``pre_tool_call``, ``govern``,
registry policy loading in ``from_agent``). Uses the fake
``agentmesh.governance`` from ``test_governance.py``; no AGT needed.

Separate from ``test_guardrails.py`` because that module skips entirely when
the optional ``pii`` extra (ff3) is not installed.
"""

from __future__ import annotations

import json
import sys
import types

import pytest

from _http_fakes import FakeResponse
from agentops import governance
from agentops.governance import Decision, PolicyDenied
from agentops.governance_events import GovernanceEventShipper
from agentops.guardrails import GuardrailResult, Guardrails
from agentops.policies import AgentPolicy, PolicyNotFound
from test_governance import POLICY_NAME, RULE_NAME, _make_fake_agt

AGENT_NAME = "support-agent"
# The fake PolicyEngine parses JSON text as its "YAML".
POLICY_TEXT = json.dumps({"name": POLICY_NAME, "deny_tools": ["delete_file"], "agents": ["*"]})
CREDS = ("https://control.example.test", "pk-test", "sk-test")


@pytest.fixture(autouse=True)
def _no_registry_guardrail_config(monkeypatch):
    """``from_agent`` also looks up the agent's text-guardrail config over
    ``requests.get`` (covered in ``test_guardrails_from_agent.py``). Answer
    it with a 404 so these tests stay offline and only exercise the policy."""
    import requests

    monkeypatch.setattr(requests, "get", lambda *a, **k: FakeResponse(404, {}))


@pytest.fixture
def fake_agt(monkeypatch):
    gov = _make_fake_agt()
    pkg = types.ModuleType("agentmesh")
    pkg.governance = gov
    monkeypatch.setitem(sys.modules, "agentmesh", pkg)
    monkeypatch.setitem(sys.modules, "agentmesh.governance", gov)
    governance._reset_for_tests()
    yield gov
    governance._reset_for_tests()


@pytest.fixture
def guard(fake_agt):
    return Guardrails(agent_name=AGENT_NAME, tool_policy=POLICY_TEXT, tool_policy_version=3)


class FakeClient:
    _host = CREDS[0]
    _public_key = CREDS[1]
    _secret_key = CREDS[2]

    def __init__(self, policy=None, error=None):
        self._policy = policy
        self._error = error
        self.calls: list[tuple] = []

    def get_policy(self, agent_name, label="production", **kwargs):
        self.calls.append((agent_name, label))
        if self._error is not None:
            raise self._error
        return self._policy


def registry_policy(version=3) -> AgentPolicy:
    return AgentPolicy(
        agent_name=AGENT_NAME,
        version=version,
        labels=["production"],
        manifest=POLICY_TEXT,
        source="registry",
        updated_at="2026-09-16T10:00:00Z",
    )


class TestGuardrailResult:
    def test_action_property_and_reasons_default(self):
        assert GuardrailResult().action == "allow"
        assert GuardrailResult(blocked=True).action == "block"
        assert GuardrailResult(transformed_text="x").action == "transform"
        assert GuardrailResult().reasons == []
        assert GuardrailResult().decision is None

    def test_text_checks_still_work(self):
        result = Guardrails(detect_pii=True).check_input("mail me at jane@example.com")
        assert result.action == "transform" and "[REDACTED_EMAIL]" in result.transformed_text


class TestPreToolCall:
    def test_allow(self, guard, fake_agt):
        result = guard.pre_tool_call("read_file", {"path": "/x"})
        assert result.action == "allow"
        assert result.blocked is False and result.passed
        assert result.reasons == []
        assert result.flags == ["policy_allow"]
        decision = result.metadata["decision"]
        assert isinstance(decision, Decision) and decision.allowed
        assert result.decision is decision
        assert result.metadata["policy_version"] == 3
        # agent_id is the registry agent name, context is AGT's shape.
        agent_did, context, stage = fake_agt.PolicyEngine.instances[-1].evaluations[-1]
        assert agent_did == AGENT_NAME and stage == "pre_tool"
        assert context == {
            "path": {"value": "/x"},
            "action": {"type": "read_file"},
            "tool": {"value": "read_file"},
        }

    def test_deny_blocks_with_policy_denied_reason(self, guard):
        result = guard.pre_tool_call("delete_file", {"path": "/x"})
        assert result.action == "block" and result.blocked
        assert result.reasons == [f"policy_denied:{RULE_NAME}"]
        assert result.flags == ["policy_deny"]
        assert result.metadata["rule"] == RULE_NAME
        assert result.metadata["policy_name"] == POLICY_NAME
        assert result.metadata["tool_name"] == "delete_file"
        assert result.decision.action == "deny"
        assert "blocked delete_file" in result.reason

    def test_require_approval_blocks_but_warn_and_log_do_not(self, guard, monkeypatch):
        def fake_evaluate(policy, agent_id, tool_name, args, kwargs=None, **kw):
            return Decision(
                allowed=False,
                action=tool_name,
                agent_id=agent_id,
                tool_name=tool_name,
                stage="pre_tool",
                rule="r",
                policy_name=POLICY_NAME,
            )

        monkeypatch.setattr(governance, "evaluate", fake_evaluate)
        assert guard.pre_tool_call("require_approval").action == "block"
        assert guard.pre_tool_call("require_approval").reasons == ["policy_denied:r"]
        warned = guard.pre_tool_call("warn")
        assert warned.action == "allow" and warned.reasons == ["policy_warn:r"]
        assert guard.pre_tool_call("log").action == "allow"

    def test_ships_decision_with_registry_version(self, guard, fake_http):
        shipper = GovernanceEventShipper(CREDS, flush_interval_s=30.0)
        guard._shipper = shipper
        guard.pre_tool_call("delete_file", {"path": "/x"}, session_id="s-9")
        shipper.shutdown(timeout=5)
        event = fake_http.posts[0].json["events"][0]
        assert event["decision"] == "deny"
        assert event["resource"] == "delete_file"
        assert event["policy_version"] == "3"
        assert event["agent_id"] == AGENT_NAME
        assert event["session_id"] == "s-9"

    def test_without_policy_raises_runtime_error_naming_the_extra(self, fake_agt):
        guard = Guardrails(agent_name=AGENT_NAME)
        assert guard.has_tool_policy is False
        with pytest.raises(RuntimeError, match=r"agentops-sdk\[governance\]") as info:
            guard.pre_tool_call("delete_file", {})
        assert "No tool-call policy is loaded for agent 'support-agent'" in str(info.value)
        with pytest.raises(RuntimeError, match=r"agentops-sdk\[governance\]"):
            guard.govern(lambda: None)

    def test_without_the_extra_raises_runtime_error(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "agentmesh", None)
        monkeypatch.setitem(sys.modules, "agentmesh.governance", None)
        governance._reset_for_tests()
        guard = Guardrails(agent_name=AGENT_NAME, tool_policy=POLICY_TEXT)
        with pytest.raises(RuntimeError, match=r'pip install "agentops-sdk\[governance\]"'):
            guard.pre_tool_call("delete_file", {})
        with pytest.raises(RuntimeError, match=r"agentops-sdk\[governance\]"):
            guard.govern(lambda: None)


class TestGovern:
    def test_decorator_forms(self, guard):
        @guard.govern
        def execute(action: str, tool: str, path: str) -> str:
            return f"{tool}:{path}"

        assert execute(action="read", tool="read_file", path="/x") == "read_file:/x"
        with pytest.raises(PolicyDenied) as info:
            execute(action="delete", tool="delete_file", path="/x")
        assert info.value.rule == RULE_NAME and info.value.agent_id == AGENT_NAME
        assert execute.__agentops_governed__ is True

        @guard.govern(tool_name="fs", raise_on_deny=False)
        def other(tool: str) -> str:
            return "ok"

        assert other(tool="read_file") == "ok"
        denied = other(tool="delete_file")
        assert isinstance(denied, Decision) and denied.denied

        wrapped = guard.govern(lambda tool: "ran", raise_on_deny=False)
        assert wrapped(tool="read_file") == "ran"

    def test_govern_ships_events(self, guard, fake_http):
        shipper = GovernanceEventShipper(CREDS, flush_interval_s=30.0)
        guard._shipper = shipper
        execute = guard.govern(lambda tool: "ok", raise_on_deny=False, tool_name="execute")
        execute(tool="read_file")
        execute(tool="delete_file")
        shipper.shutdown(timeout=5)
        events = [e for c in fake_http.posts for e in c.json["events"]]
        assert [e["decision"] for e in events] == ["allow", "deny"]
        assert all(e["resource"] == "execute" and e["policy_version"] == "3" for e in events)


GUARDRAIL_CONFIG = {
    "enabled": True,
    "preInput": {"detectPii": True, "piiAction": "block", "detectJailbreak": True},
    "postOutput": {"detectSecrets": True},
}


class TestFromAgent:
    def test_loads_text_config_and_registry_policy(self, fake_agt, fake_http, monkeypatch):
        import requests

        gets: list[tuple[str, dict]] = []
        listing = {"data": [{"id": "agent-1", "name": AGENT_NAME, "guardrailConfig": GUARDRAIL_CONFIG}], "meta": {}}

        def fake_get(url, **kwargs):
            gets.append((url, kwargs))
            return FakeResponse(200, listing)

        monkeypatch.setattr(requests, "get", fake_get)
        client = FakeClient(policy=registry_policy(version=5))

        guard = Guardrails.from_agent(client, AGENT_NAME)

        assert guard.detect_pii and guard.pii_action == "block" and guard.detect_secrets
        assert guard.agent_name == AGENT_NAME
        assert guard.has_tool_policy and guard.tool_policy == POLICY_TEXT
        assert guard.policy.version == 5 and guard.agent_id == AGENT_NAME
        assert client.calls == [(AGENT_NAME, "production")]
        ((url, kwargs),) = gets
        assert url.endswith("/api/public/agents")
        assert kwargs["params"]["name"] == AGENT_NAME

        assert guard.pre_tool_call("delete_file", {"path": "/x"}).action == "block"
        assert guard.pre_tool_call("read_file", {"path": "/x"}).action == "allow"

    def test_accepts_agent_name_first(self, fake_agt, fake_http):
        client = FakeClient(policy=registry_policy())
        guard = Guardrails.from_agent(AGENT_NAME, client, policy_label="staging")
        assert guard.agent_name == AGENT_NAME and guard.has_tool_policy
        assert client.calls == [(AGENT_NAME, "staging")]

    def test_policy_not_found_is_swallowed(self, fake_agt, fake_http):
        fake_http.queue(FakeResponse(404, {"error": "not_found"}))
        client = FakeClient(error=PolicyNotFound(AGENT_NAME, "production"))
        guard = Guardrails.from_agent(client, AGENT_NAME)
        assert guard.agent_name == AGENT_NAME
        assert guard.has_tool_policy is False
        assert guard.detect_pii is False  # registry 404 → no-op text pipeline
        with pytest.raises(RuntimeError, match=r"agentops-sdk\[governance\]"):
            guard.pre_tool_call("delete_file", {})

    def test_other_policy_errors_are_logged_not_raised(self, fake_agt, fake_http, caplog):
        client = FakeClient(error=ConnectionError("down"))
        with caplog.at_level("WARNING", logger="agentops.guardrails"):
            guard = Guardrails.from_agent(client, AGENT_NAME)
        assert guard.has_tool_policy is False
        assert "could not load tool-call policy" in caplog.text

    def test_load_policy_false_and_clients_without_get_policy(self, fake_agt, fake_http):
        client = FakeClient(policy=registry_policy())
        assert Guardrails.from_agent(client, AGENT_NAME, load_policy=False).has_tool_policy is False
        assert client.calls == []

        class Legacy:
            _host = CREDS[0]
            _secret_key = CREDS[2]

        assert Guardrails.from_agent(Legacy(), AGENT_NAME).has_tool_policy is False

    def test_requires_a_string_agent_name(self):
        with pytest.raises(TypeError):
            Guardrails.from_agent(FakeClient(), None)

    def test_shipper_passthrough(self, fake_agt, fake_http):
        shipper = GovernanceEventShipper(CREDS, flush_interval_s=30.0)
        guard = Guardrails.from_agent(FakeClient(policy=registry_policy(version=8)), AGENT_NAME, shipper=shipper)
        guard.pre_tool_call("delete_file", {})
        shipper.shutdown(timeout=5)
        event = fake_http.posts[0].json["events"][0]
        assert event["policy_version"] == "8" and event["decision"] == "deny"


class TestSetToolPolicy:
    def test_accepts_agent_policy_or_raw_spec(self, fake_agt):
        guard = Guardrails()
        guard.set_tool_policy(registry_policy(version=2))
        assert guard.tool_policy == POLICY_TEXT and guard._agt_policy_version == 2
        guard.set_tool_policy(POLICY_TEXT, version="v9")
        assert guard.policy is None and guard._agt_policy_version == "v9"
        assert guard.agent_id == "*"  # no agent name → AGT's wildcard agent
