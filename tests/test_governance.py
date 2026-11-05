"""Tests for ``agentops.governance`` — the AGT wrapper that makes policy
decisions land in AOS Control as ``agt.policy.evaluate`` spans.

None of these tests need the Agent Governance Toolkit installed: a minimal
fake ``agentmesh.governance`` is injected into ``sys.modules``. The fake's
``enable_otel`` / ``trace_policy_evaluation`` are line-for-line ports of AGT
4.1.0's ``otel_observability.py`` (recorded in
``packages/shared/src/server/otel/__fixtures__/agt/README.md``) so the span
shape asserted here is the one the real toolkit produces.

Span-shape tests use the real OpenTelemetry SDK's in-memory exporter and are
skipped when ``opentelemetry-sdk`` is not installed. A final class runs the
same flow against the *real* AGT when it happens to be importable.
"""

from __future__ import annotations

import asyncio
import base64
import json
import sys
import types
from pathlib import Path

import pytest

import agentops
from agentops import governance
from agentops.governance import (
    ARGS_MAX_BYTES,
    Decision,
    PolicyDenied,
    build_context,
    enable_agt_export,
    governed,
    pre_tool_call,
    serialize_args,
)

try:  # OpenTelemetry is optional for the SDK; the span tests need it.
    from opentelemetry import trace as otel_trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    HAVE_OTEL_SDK = True
except ImportError:  # pragma: no cover - exercised only without the extra
    HAVE_OTEL_SDK = False

needs_otel = pytest.mark.skipif(not HAVE_OTEL_SDK, reason="opentelemetry-sdk not installed")

AGENT_ID = "did:mesh:test-agent"
POLICY_NAME = "unit-test-policy"
RULE_NAME = "deny-destructive-tool-calls"


# --------------------------------------------------------------------------- #
# Fake agentmesh.governance (AGT 4.1.0 surface the wrapper touches)
# --------------------------------------------------------------------------- #
def _make_fake_agt() -> types.ModuleType:
    gov = types.ModuleType("agentmesh.governance")

    class Policy:
        """Stand-in for agentmesh.governance.policy.Policy.

        ``deny_tools`` lists ``tool.value`` values that a single pre_tool
        rule named ``RULE_NAME`` denies; anything else gets ``default_action``.
        """

        def __init__(self, name, deny_tools=(), default_action="allow", agents=None, agent=None):
            self.name = name
            self.deny_tools = list(deny_tools)
            self.default_action = default_action
            self.agents = list(agents or [])
            self.agent = agent

        def applies_to(self, agent_did):
            return "*" in self.agents or agent_did in self.agents or self.agent == agent_did

    class PolicyDecision:
        def __init__(self, allowed, action, matched_rule=None, policy_name=None, reason=None):
            self.allowed = allowed
            self.action = action
            self.matched_rule = matched_rule
            self.policy_name = policy_name
            self.reason = reason

    class GovernanceDenied(Exception):
        def __init__(self, decision):
            self.decision = decision
            super().__init__(
                f"Action denied by policy rule '{decision.matched_rule}': {decision.reason}"
            )

    class PolicyEngine:
        instances: list["PolicyEngine"] = []

        def __init__(self, conflict_strategy="priority_first_match"):
            self.conflict_strategy = conflict_strategy
            self._policies = {}
            self.evaluations = []
            self.loaded_via = None
            PolicyEngine.instances.append(self)

        def load_policy(self, policy):
            self._policies[policy.name] = policy
            self.loaded_via = self.loaded_via or "object"

        def load_yaml(self, yaml_content):
            # The fake accepts JSON text so the tests don't need PyYAML.
            data = json.loads(yaml_content)
            policy = Policy(**data)
            self.load_policy(policy)
            self.loaded_via = "yaml"
            return policy

        def load_yaml_file(self, file_path):
            policy = self.load_yaml(Path(file_path).read_text())
            self.loaded_via = "file"
            return policy

        def evaluate(self, agent_did, context, stage="pre_tool"):
            self.evaluations.append((agent_did, context, stage))
            applicable = [p for p in self._policies.values() if p.applies_to(agent_did)]
            for policy in applicable:
                tool = (context.get("tool") or {}).get("value")
                if stage == "pre_tool" and tool in policy.deny_tools:
                    return PolicyDecision(
                        allowed=False,
                        action="deny",
                        matched_rule=RULE_NAME,
                        policy_name=policy.name,
                        reason=f"Policy '{policy.name}' blocked {tool}.",
                    )
            default = applicable[0].default_action if applicable else "deny"
            return PolicyDecision(allowed=(default == "allow"), action=default)

    def govern(fn, *, policy, agent_id="*", **_kwargs):  # pragma: no cover - not used by wrapper
        raise AssertionError("agentops.governance must not call AGT's govern() (double evaluation)")

    # ---- ports of agentmesh/governance/otel_observability.py (AGT 4.1.0) ----
    gov._tracer = None
    gov._initialized = False
    gov.enable_otel_calls = []

    def enable_otel(service_name="agt-governance", endpoint=None):
        gov.enable_otel_calls.append({"service_name": service_name, "endpoint": endpoint})
        if gov._initialized:
            return
        try:
            from opentelemetry import trace
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider as _TP
        except ImportError:
            gov._initialized = True
            return
        provider = _TP(resource=Resource.create({"service.name": service_name}))
        trace.set_tracer_provider(provider)  # first-wins, exactly like AGT
        gov._tracer = trace.get_tracer("agentmesh.governance", "1.0.0")
        gov._initialized = True

    from contextlib import contextmanager

    @contextmanager
    def trace_policy_evaluation(agent_id="", stage="pre_tool", context=None):
        result = {}
        if gov._tracer:
            from opentelemetry import trace

            with gov._tracer.start_as_current_span("agt.policy.evaluate") as span:
                span.set_attribute("agt.agent.id", agent_id)
                span.set_attribute("agt.policy.stage", stage)
                try:
                    yield result
                finally:
                    action = result.get("action", "unknown")
                    rule = result.get("rule", "")
                    span.set_attribute("agt.policy.action", action)
                    span.set_attribute("agt.policy.rule", rule or "none")
                    if result.get("policy_name"):
                        span.set_attribute("agt.policy.name", result["policy_name"])
                    if not result.get("allowed", True):
                        span.set_status(trace.Status(trace.StatusCode.OK, f"Denied by {rule}"))
        else:
            yield result

    gov.Policy = Policy
    gov.PolicyDecision = PolicyDecision
    gov.GovernanceDenied = GovernanceDenied
    gov.PolicyEngine = PolicyEngine
    gov.govern = govern
    gov.enable_otel = enable_otel
    gov.trace_policy_evaluation = trace_policy_evaluation
    return gov


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
def policy(fake_agt):
    return fake_agt.Policy(POLICY_NAME, deny_tools=["delete_file"], agents=["*"])


@pytest.fixture(scope="session")
def _memory_provider():
    """One global provider for the whole session: OTel's set_tracer_provider
    is first-wins, so AGT's enable_otel() (fake or real) binds to this one,
    exactly as it does when an application already configured tracing."""
    if not HAVE_OTEL_SDK:
        pytest.skip("opentelemetry-sdk not installed")
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    otel_trace.set_tracer_provider(provider)
    return exporter


@pytest.fixture
def spans(_memory_provider, fake_agt):
    _memory_provider.clear()
    fake_agt.enable_otel(service_name="test-agent")
    return _memory_provider


def _only_span(exporter):
    finished = exporter.get_finished_spans()
    assert len(finished) == 1, [s.name for s in finished]
    return finished[0]


# --------------------------------------------------------------------------- #
# Import / packaging
# --------------------------------------------------------------------------- #
class TestImportWithoutAgt:
    def test_import_agentops_does_not_import_agentmesh(self, monkeypatch):
        monkeypatch.delitem(sys.modules, "agentmesh", raising=False)
        monkeypatch.delitem(sys.modules, "agentmesh.governance", raising=False)
        import importlib

        importlib.reload(agentops)
        assert "agentmesh" not in sys.modules
        # Lazy exports resolve without AGT present.
        assert agentops.governed is governance.governed
        assert agentops.pre_tool_call is governance.pre_tool_call
        assert agentops.enable_agt_export is governance.enable_agt_export
        assert agentops.PolicyDenied is governance.PolicyDenied
        assert agentops.Decision is governance.Decision
        assert "governed" in agentops.__all__
        with pytest.raises(AttributeError):
            agentops.not_a_real_name

    def test_using_the_helper_without_agt_points_at_the_extra(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "agentmesh", None)
        monkeypatch.setitem(sys.modules, "agentmesh.governance", None)
        governance._reset_for_tests()
        with pytest.raises(ImportError, match=r"agentops-sdk\[governance\]"):
            pre_tool_call("policy.yaml", AGENT_ID, "read_file", {"path": "/x"})


# --------------------------------------------------------------------------- #
# Context shape (must match AGT's GovernedCallable._build_context)
# --------------------------------------------------------------------------- #
class TestBuildContext:
    def test_matches_agt_shape_for_govern_style_kwargs(self):
        ctx = build_context(
            (),
            {"action": "delete", "tool": "delete_file", "path": "/data/x.csv", "meta": {"k": 1}},
            tool_name="execute_tool",
        )
        assert ctx == {
            "action": {"type": "delete"},
            "tool": {"value": "delete_file"},
            "path": {"value": "/data/x.csv"},
            "meta": {"k": 1},  # dicts pass through untouched, like AGT
        }

    def test_first_positional_arg_is_action_type_and_dict_action_passes_through(self):
        assert build_context(("read", "x"), {})["action"] == {"type": "read"}
        assert build_context((), {"action": {"type": "read", "scope": "s"}})["action"] == {
            "type": "read",
            "scope": "s",
        }

    def test_tool_name_fills_action_and_tool_only_when_absent(self):
        ctx = build_context((), {"path": "/x"}, tool_name="delete_file")
        assert ctx == {
            "path": {"value": "/x"},
            "action": {"type": "delete_file"},
            "tool": {"value": "delete_file"},
        }
        ctx = build_context((), {"tool": "other", "action": "read"}, tool_name="delete_file")
        assert ctx["tool"] == {"value": "other"}
        assert ctx["action"] == {"type": "read"}


class TestSerializeArgs:
    def test_json_with_pii_redacted(self):
        text = serialize_args((), {"to": "jane@example.com", "n": 1})
        data = json.loads(text)
        assert data == {"to": "[REDACTED_EMAIL]", "n": 1}

    def test_positional_args_kept_under_underscore_args(self):
        assert json.loads(serialize_args(("delete",), {"path": "/x"})) == {
            "_args": ["delete"],
            "path": "/x",
        }

    def test_truncated_to_budget(self):
        text = serialize_args((), {"blob": "x" * (ARGS_MAX_BYTES * 3)})
        assert len(text.encode("utf-8")) <= ARGS_MAX_BYTES
        assert text.endswith("[truncated]")

    def test_redaction_can_be_disabled_and_unserializable_values_repr(self):
        class Opaque:
            def __repr__(self):
                return "<opaque>"

        text = serialize_args((), {"to": "jane@example.com", "o": Opaque()}, redact=False)
        assert json.loads(text) == {"to": "jane@example.com", "o": "<opaque>"}


# --------------------------------------------------------------------------- #
# pre_tool_call
# --------------------------------------------------------------------------- #
class TestPreToolCall:
    def test_allow_returns_decision_with_agt_context(self, fake_agt, policy):
        decision = pre_tool_call(policy, AGENT_ID, "read_file", {"path": "/data/x.csv"})
        assert isinstance(decision, Decision)
        assert decision.allowed and not decision.denied
        assert decision.action == "allow"
        assert decision.rule is None and decision.policy_name is None
        assert decision.tool_name == "read_file"
        assert decision.stage == "pre_tool"
        engine = fake_agt.PolicyEngine.instances[-1]
        agent, ctx, stage = engine.evaluations[-1]
        assert (agent, stage) == (AGENT_ID, "pre_tool")
        assert ctx == {
            "path": {"value": "/data/x.csv"},
            "action": {"type": "read_file"},
            "tool": {"value": "read_file"},
        }

    def test_deny_returns_decision_by_default(self, policy):
        decision = pre_tool_call(policy, AGENT_ID, "delete_file", {"path": "/x"})
        assert decision.denied
        assert (decision.action, decision.rule, decision.policy_name) == ("deny", RULE_NAME, POLICY_NAME)
        assert "blocked" in decision.reason
        assert decision.raw.matched_rule == RULE_NAME

    def test_deny_raises_when_asked(self, fake_agt, policy):
        with pytest.raises(PolicyDenied) as info:
            pre_tool_call(policy, AGENT_ID, "delete_file", {"path": "/x"}, raise_on_deny=True)
        exc = info.value
        assert (exc.rule, exc.policy_name, exc.agent_id, exc.tool_name) == (
            RULE_NAME,
            POLICY_NAME,
            AGENT_ID,
            "delete_file",
        )
        assert isinstance(exc.__cause__, fake_agt.GovernanceDenied)
        assert exc.governance_denied is exc.__cause__
        assert exc.__cause__.decision is exc.decision.raw
        assert "delete_file" in str(exc) and RULE_NAME in str(exc)

    def test_stage_is_forwarded_to_engine(self, fake_agt, policy):
        decision = pre_tool_call(policy, AGENT_ID, "delete_file", {}, stage="post_tool")
        assert decision.allowed  # the fake's rule only fires at pre_tool, like AGT stage filtering
        assert fake_agt.PolicyEngine.instances[-1].evaluations[-1][2] == "post_tool"

    def test_engine_is_cached_per_policy_and_strategy(self, fake_agt, policy):
        pre_tool_call(policy, AGENT_ID, "read_file", {})
        pre_tool_call(policy, AGENT_ID, "read_file", {})
        assert len(fake_agt.PolicyEngine.instances) == 1
        assert fake_agt.PolicyEngine.instances[0].conflict_strategy == "deny_overrides"
        pre_tool_call(policy, AGENT_ID, "read_file", {}, conflict_strategy="priority_first_match")
        assert len(fake_agt.PolicyEngine.instances) == 2
        governance.clear_engine_cache()
        pre_tool_call(policy, AGENT_ID, "read_file", {})
        assert len(fake_agt.PolicyEngine.instances) == 3

    def test_policy_from_file_path_string_and_inline_text(self, fake_agt, tmp_path):
        spec = json.dumps({"name": POLICY_NAME, "deny_tools": ["delete_file"]})
        path = tmp_path / "policy.yaml"
        path.write_text(spec)

        assert pre_tool_call(str(path), AGENT_ID, "delete_file", {}).denied
        assert fake_agt.PolicyEngine.instances[-1].loaded_via == "file"
        assert pre_tool_call(path, AGENT_ID, "delete_file", {}).denied  # pathlib.Path
        assert fake_agt.PolicyEngine.instances[-1].loaded_via == "file"
        assert pre_tool_call(spec, AGENT_ID, "delete_file", {}).denied  # inline text
        assert fake_agt.PolicyEngine.instances[-1].loaded_via == "yaml"
        # Untargeted policies default to every agent, as govern() does.
        assert fake_agt.PolicyEngine.instances[-1]._policies[POLICY_NAME].agents == ["*"]

    def test_unsupported_policy_type(self, fake_agt):
        with pytest.raises(TypeError, match="policy must be"):
            pre_tool_call(42, AGENT_ID, "read_file", {})


# --------------------------------------------------------------------------- #
# governed decorator
# --------------------------------------------------------------------------- #
class TestGoverned:
    def test_decorator_form_allows_and_calls_fn_once(self, fake_agt, policy):
        calls = []

        @governed(policy=policy, agent_id=AGENT_ID)
        def execute_tool(action: str, tool: str, path: str) -> str:
            calls.append((action, tool, path))
            return f"{tool}({path})"

        assert execute_tool.__name__ == "execute_tool"
        assert execute_tool.tool_name == "execute_tool"
        assert execute_tool(action="read", tool="read_file", path="/x") == "read_file(/x)"
        assert calls == [("read", "read_file", "/x")]
        engine = fake_agt.PolicyEngine.instances[-1]
        assert len(engine.evaluations) == 1  # evaluated once, not via govern()
        assert engine.evaluations[0][1] == {
            "action": {"type": "read"},
            "tool": {"value": "read_file"},
            "path": {"value": "/x"},
        }

    def test_wrapper_form_denies_with_policy_denied_and_skips_fn(self, fake_agt, policy):
        calls = []

        def execute_tool(action, tool, path):
            calls.append(tool)

        wrapped = governed(execute_tool, policy=policy, agent_id=AGENT_ID, tool_name="fs")
        with pytest.raises(PolicyDenied) as info:
            wrapped(action="delete", tool="delete_file", path="/x")
        assert calls == []
        assert info.value.tool_name == "fs"
        assert info.value.rule == RULE_NAME
        assert isinstance(info.value.__cause__, fake_agt.GovernanceDenied)

    def test_raise_on_deny_false_returns_decision(self, policy):
        @governed(policy=policy, agent_id=AGENT_ID, raise_on_deny=False)
        def execute_tool(action, tool, path):
            return "ran"

        result = execute_tool(action="delete", tool="delete_file", path="/x")
        assert isinstance(result, Decision) and result.denied
        assert execute_tool(action="read", tool="read_file", path="/x") == "ran"

    def test_dry_run_evaluate_does_not_call_fn(self, policy):
        calls = []

        @governed(policy=policy, agent_id=AGENT_ID)
        def execute_tool(action, tool, path):
            calls.append(1)

        decision = execute_tool.evaluate(action="delete", tool="delete_file", path="/x")
        assert decision.denied and calls == []

    def test_async_function(self, policy):
        @governed(policy=policy, agent_id=AGENT_ID)
        async def execute_tool(action, tool, path):
            return "async-ran"

        assert asyncio.run(execute_tool(action="read", tool="read_file", path="/x")) == "async-ran"
        with pytest.raises(PolicyDenied):
            asyncio.run(execute_tool(action="delete", tool="delete_file", path="/x"))


# --------------------------------------------------------------------------- #
# Span shape (real OpenTelemetry SDK, in-memory exporter)
# --------------------------------------------------------------------------- #
@needs_otel
class TestSpans:
    def test_deny_span_attributes(self, spans, policy):
        @governed(policy=policy, agent_id=AGENT_ID, session_id="sess-42")
        def execute_tool(action, tool, path):
            return "never"

        with pytest.raises(PolicyDenied):
            execute_tool(action="delete", tool="delete_file", path="/data/report.csv")

        span = _only_span(spans)
        assert span.name == "agt.policy.evaluate"
        attrs = dict(span.attributes)
        assert attrs["agt.agent.id"] == AGENT_ID
        assert attrs["agt.policy.stage"] == "pre_tool"
        assert attrs["agt.policy.action"] == "deny"
        assert attrs["agt.policy.rule"] == RULE_NAME
        assert attrs["agt.policy.name"] == POLICY_NAME
        assert attrs["agt.tool.name"] == "execute_tool"
        assert json.loads(attrs["agt.tool.args"]) == {
            "action": "delete",
            "tool": "delete_file",
            "path": "/data/report.csv",
        }
        assert attrs["session.id"] == "sess-42"
        # AGT marks denials Status(OK, ...) — the SDK keeps the OK code and
        # drops the description; mapping must key on agt.policy.action.
        assert span.status.status_code.name == "OK"

    def test_allow_span_has_rule_none_and_no_policy_name(self, spans, policy):
        decision = pre_tool_call(policy, AGENT_ID, "read_file", {"path": "/x"})
        assert decision.allowed
        span = _only_span(spans)
        attrs = dict(span.attributes)
        assert attrs["agt.policy.action"] == "allow"
        assert attrs["agt.policy.rule"] == "none"
        assert "agt.policy.name" not in attrs
        assert attrs["agt.tool.name"] == "read_file"
        assert json.loads(attrs["agt.tool.args"]) == {"path": "/x"}
        assert "session.id" not in attrs
        assert span.status.status_code.name == "UNSET"

    def test_args_are_redacted_and_truncated_and_can_be_omitted(self, spans, policy):
        # Keys are sorted in the JSON, so the padding key must sort after "to"
        # or the email would be truncated away before we can check redaction.
        pre_tool_call(policy, AGENT_ID, "send_email", {"to": "jane@example.com", "zz_body": "x" * 10_000})
        attrs = dict(_only_span(spans).attributes)
        args = attrs["agt.tool.args"]
        assert "[REDACTED_EMAIL]" in args and "jane@example.com" not in args
        assert len(args.encode("utf-8")) <= ARGS_MAX_BYTES
        assert args.endswith("[truncated]")

        spans.clear()
        pre_tool_call(policy, AGENT_ID, "send_email", {"to": "jane@example.com"}, record_args=False)
        assert "agt.tool.args" not in dict(_only_span(spans).attributes)

    def test_session_id_from_env_and_callable(self, spans, policy, monkeypatch):
        monkeypatch.setenv("AGENTOPS_SESSION_ID", "env-session")
        pre_tool_call(policy, AGENT_ID, "read_file", {})
        assert dict(_only_span(spans).attributes)["session.id"] == "env-session"

        spans.clear()
        pre_tool_call(policy, AGENT_ID, "read_file", {}, session_id=lambda: "dyn-session")
        assert dict(_only_span(spans).attributes)["session.id"] == "dyn-session"

    def test_stage_and_custom_tool_name_on_span(self, spans, policy):
        pre_tool_call(policy, AGENT_ID, "delete_file", {}, stage="post_tool")
        attrs = dict(_only_span(spans).attributes)
        assert attrs["agt.policy.stage"] == "post_tool"
        assert attrs["agt.tool.name"] == "delete_file"

    def test_no_span_and_no_foreign_annotation_when_agt_tracer_disabled(self, _memory_provider, fake_agt, policy):
        # enable_otel() NOT called: AGT's trace_policy_evaluation yields
        # without a span. An application span that happens to be active
        # must not pick up agt.tool.* attributes.
        _memory_provider.clear()
        tracer = otel_trace.get_tracer("app")
        with tracer.start_as_current_span("app.step"):
            pre_tool_call(policy, AGENT_ID, "delete_file", {"path": "/x"})
        finished = _memory_provider.get_finished_spans()
        assert [s.name for s in finished] == ["app.step"]
        assert "agt.tool.name" not in dict(finished[0].attributes)


# --------------------------------------------------------------------------- #
# enable_agt_export
# --------------------------------------------------------------------------- #
def _processors(provider):
    return list(provider._active_span_processor._span_processors)


def _exporter_of(processor):
    return getattr(processor, "span_exporter", None) or processor._exporter


@needs_otel
class TestEnableAgtExport:
    @pytest.fixture(autouse=True)
    def _needs_otlp_exporter(self):
        pytest.importorskip("opentelemetry.exporter.otlp.proto.http.trace_exporter")

    def test_builds_endpoint_and_basic_header_and_is_idempotent(self, fake_agt):
        provider = TracerProvider(shutdown_on_exit=False)
        returned = enable_agt_export(
            "support-agent",
            host="https://control.example.test/",
            public_key="pk-test",
            secret_key="sk-test",
            provider=provider,
        )
        assert returned is provider
        assert fake_agt.enable_otel_calls == [{"service_name": "support-agent", "endpoint": None}]

        procs = _processors(provider)
        assert len(procs) == 1
        assert type(procs[0]).__name__ == "BatchSpanProcessor"
        exporter = _exporter_of(procs[0])
        assert exporter._endpoint == "https://control.example.test/api/public/otel/v1/traces"
        expected = "Basic " + base64.b64encode(b"pk-test:sk-test").decode()
        assert exporter._headers["Authorization"] == expected

        enable_agt_export(
            "support-agent",
            host="https://control.example.test",
            public_key="pk-test",
            secret_key="sk-test",
            provider=provider,
        )
        assert len(_processors(provider)) == 1

    def test_resolves_credentials_from_env(self, fake_agt, monkeypatch):
        monkeypatch.setenv("AGENTOPS_HOST", "https://env.example.test")
        monkeypatch.setenv("AGENTOPS_PUBLIC_KEY", "pk-env")
        monkeypatch.setenv("AGENTOPS_SECRET_KEY", "sk-env")
        provider = TracerProvider(shutdown_on_exit=False)
        enable_agt_export(provider=provider)
        exporter = _exporter_of(_processors(provider)[0])
        assert exporter._endpoint == "https://env.example.test/api/public/otel/v1/traces"
        assert exporter._headers["Authorization"] == "Basic " + base64.b64encode(b"pk-env:sk-env").decode()
        assert fake_agt.enable_otel_calls[-1]["service_name"] == governance.DEFAULT_SERVICE_NAME

    def test_missing_credentials_warns_and_attaches_nothing(self, fake_agt, monkeypatch, caplog):
        for var in ("AGENTOPS_HOST", "AGENTOPS_BASE_URL", "AGENTOPS_PUBLIC_KEY", "AGENTOPS_SECRET_KEY"):
            monkeypatch.delenv(var, raising=False)
        provider = TracerProvider(shutdown_on_exit=False)
        with caplog.at_level("WARNING", logger="agentops.governance"):
            assert enable_agt_export(provider=provider) is provider
        assert _processors(provider) == []
        assert "AGENTOPS_HOST" in caplog.text and "AGENTOPS_SECRET_KEY" in caplog.text
        assert fake_agt.enable_otel_calls  # AGT tracer still switched on

    def test_attaches_to_existing_global_provider(self, _memory_provider, fake_agt):
        global_provider = otel_trace.get_tracer_provider()
        before = _processors(global_provider)
        returned = enable_agt_export(
            host="https://control.example.test",
            public_key="pk",
            secret_key="sk",
        )
        assert returned is global_provider
        added = [p for p in _processors(global_provider) if p not in before]
        assert len(added) == 1
        exporter = _exporter_of(added[0])
        try:
            assert exporter._endpoint == "https://control.example.test/api/public/otel/v1/traces"
        finally:
            # Detach again so later tests don't try to POST to the fake host.
            global_provider._active_span_processor._span_processors = tuple(before)
            added[0].shutdown()


# --------------------------------------------------------------------------- #
# Real AGT (only when agent-governance-toolkit-core is importable)
# --------------------------------------------------------------------------- #
REAL_POLICY_YAML = f"""
apiVersion: governance.toolkit/v1
name: {POLICY_NAME}
version: "1.0"
agents: ["*"]
default_action: allow
rules:
  - name: {RULE_NAME}
    stage: pre_tool
    condition: "action.type in ['drop', 'delete', 'truncate'] or tool.value == 'delete_file'"
    action: deny
    priority: 100
"""


@needs_otel
class TestRealAgt:
    @pytest.fixture
    def real_agt(self, _memory_provider):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            gov = pytest.importorskip("agentmesh.governance")
        governance._reset_for_tests()
        gov.enable_otel(service_name="real-agt-test")
        _memory_provider.clear()
        yield gov
        governance._reset_for_tests()

    def test_end_to_end_against_agt_4_1(self, real_agt, _memory_provider):
        @governed(policy=REAL_POLICY_YAML, agent_id=AGENT_ID, session_id="s1")
        def execute_tool(action: str, tool: str, path: str) -> str:
            return f"{tool}({path}) executed"

        assert execute_tool(action="read", tool="read_file", path="/x") == "read_file(/x) executed"
        with pytest.raises(PolicyDenied) as info:
            execute_tool(action="delete", tool="delete_file", path="/x")
        assert (info.value.rule, info.value.policy_name) == (RULE_NAME, POLICY_NAME)
        assert isinstance(info.value.__cause__, real_agt.GovernanceDenied)

        allow_span, deny_span = _memory_provider.get_finished_spans()
        allow, deny = dict(allow_span.attributes), dict(deny_span.attributes)
        assert allow_span.name == deny_span.name == "agt.policy.evaluate"
        assert allow["agt.policy.action"] == "allow" and allow["agt.policy.rule"] == "none"
        assert "agt.policy.name" not in allow
        assert deny["agt.policy.action"] == "deny"
        assert deny["agt.policy.rule"] == RULE_NAME
        assert deny["agt.policy.name"] == POLICY_NAME
        assert deny["agt.tool.name"] == "execute_tool"
        assert json.loads(deny["agt.tool.args"])["tool"] == "delete_file"
        assert deny["session.id"] == "s1"

    def test_pre_tool_call_context_matches_govern_context(self, real_agt):
        # Same policy, same decision whether evaluated through AGT's govern()
        # or through pre_tool_call() with only the tool's argument dict.
        governed_by_agt = real_agt.govern(
            lambda **kw: "ok", policy=REAL_POLICY_YAML, agent_id=AGENT_ID, audit=False
        )
        with pytest.raises(real_agt.GovernanceDenied):
            governed_by_agt(action="delete", tool="delete_file", path="/x")
        assert governed_by_agt(action="read", tool="read_file", path="/x") == "ok"

        assert pre_tool_call(REAL_POLICY_YAML, AGENT_ID, "delete_file", {"path": "/x"}).denied
        assert pre_tool_call(REAL_POLICY_YAML, AGENT_ID, "read_file", {"path": "/x"}).allowed
