"""Tests for ``agentops.governance_events`` — the batching shipper for
``POST /api/public/governance-events`` and the Decision / AGT AuditEntry
converters — plus its hook-up in ``agentops.governance``.

HTTP goes through the ``fake_http`` fixture (``conftest.py``), which swaps
the shared ``requests.Session`` used by ``agentops._http``. The AGT surface
is the fake from ``test_governance.py``; the audit-entry fixture is the real
AGT 4.1.0 capture in ``packages/shared/src/server/otel/__fixtures__/agt``.
"""

from __future__ import annotations

import base64
import json
import sys
import time
import types
from datetime import datetime, timezone
from pathlib import Path

import pytest

from _http_fakes import FakeResponse
from agentops import governance
from agentops.governance import Decision, enable_agt_export, pre_tool_call, set_default_shipper
from agentops.governance_events import (
    ATTRIBUTES_MAX_BYTES,
    EVENT_ID_MAX_LEN,
    AgentOpsAuditSink,
    GovernanceEvent,
    GovernanceEventShipper,
    audit_entry_to_event,
    decision_to_event,
    ship_audit_log,
)
from test_governance import AGENT_ID, POLICY_NAME, RULE_NAME, _make_fake_agt

try:
    from opentelemetry import trace as otel_trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    HAVE_OTEL_SDK = True
except ImportError:  # pragma: no cover
    HAVE_OTEL_SDK = False

needs_otel = pytest.mark.skipif(not HAVE_OTEL_SDK, reason="opentelemetry-sdk not installed")

HOST = "https://control.example.test/"
CREDS = (HOST, "pk-test", "sk-test")
EXPECTED_URL = "https://control.example.test/api/public/governance-events"
EXPECTED_AUTH = "Basic " + base64.b64encode(b"pk-test:sk-test").decode()

FIXTURE_DIR = Path(__file__).resolve().parents[3] / "packages/shared/src/server/otel/__fixtures__/agt"
AUDIT_FIXTURE = FIXTURE_DIR / "governance-event.json"


def _shipper(**kw) -> GovernanceEventShipper:
    kw.setdefault("flush_interval_s", 0.05)
    kw.setdefault("backoff_s", 0.01)
    return GovernanceEventShipper(CREDS, **kw)


def _decision(action="deny", tool="delete_file", rule=RULE_NAME, allowed=None, reason="nope") -> Decision:
    return Decision(
        allowed=(action == "allow") if allowed is None else allowed,
        action=action,
        agent_id=AGENT_ID,
        tool_name=tool,
        stage="pre_tool",
        rule=rule,
        policy_name=POLICY_NAME,
        reason=reason,
        context={"action": {"type": "delete"}, "tool": {"value": tool}, "path": {"value": "/x"}},
    )


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


# --------------------------------------------------------------------------- #
# GovernanceEvent
# --------------------------------------------------------------------------- #
class TestGovernanceEvent:
    def test_to_wire_has_required_keys_and_omits_none(self):
        event = GovernanceEvent(kind="policy.evaluation", decision="deny", policy_version=3)
        wire = event.to_wire()
        assert set(wire) == {"event_id", "occurred_at", "kind", "decision", "policy_version", "source"}
        assert wire["policy_version"] == "3"
        assert wire["occurred_at"].endswith("Z")
        assert len(wire["event_id"]) == 36  # uuid4 by default

    def test_attributes_are_kept_under_16kb(self):
        event = GovernanceEvent(
            kind="k",
            attributes={"small": 1, "huge": "x" * (ATTRIBUTES_MAX_BYTES * 2), "mid": "y" * 100},
        )
        attrs = event.to_wire()["attributes"]
        assert len(json.dumps(attrs).encode()) <= ATTRIBUTES_MAX_BYTES
        assert attrs["small"] == 1 and attrs["mid"] == "y" * 100
        assert attrs["_dropped"] == ["huge"]

    def test_event_id_is_clipped_deterministically(self):
        long_id = "e" * 300
        a = GovernanceEvent(kind="k", event_id=long_id).to_wire()["event_id"]
        b = GovernanceEvent(kind="k", event_id=long_id).to_wire()["event_id"]
        assert a == b and len(a) <= EVENT_ID_MAX_LEN
        short = GovernanceEvent(kind="k", event_id="audit_1").to_wire()["event_id"]
        assert short == "audit_1"


class TestDecisionToEvent:
    def test_maps_decision_fields(self):
        event = decision_to_event(
            _decision(), trace_id="a" * 32, span_id="b" * 16, session_id="s1", policy_version=7
        )
        assert event.kind == "policy.evaluation"
        assert event.decision == "deny"
        assert event.severity == "warning"
        assert event.rule == RULE_NAME
        assert event.policy_name == POLICY_NAME
        assert event.policy_version == 7
        assert event.resource == "delete_file"
        assert event.action == "delete"
        assert event.agent_id == AGENT_ID
        assert (event.trace_id, event.span_id, event.session_id) == ("a" * 32, "b" * 16, "s1")
        assert event.attributes == {"stage": "pre_tool", "allowed": False, "reason": "nope"}
        assert event.source == "agentops-sdk-python"
        assert len(event.event_id) == 36

    @pytest.mark.parametrize(
        "action,severity",
        [("allow", "info"), ("warn", "info"), ("log", "info"), ("deny", "warning"), ("require_approval", "warning")],
    )
    def test_severity(self, action, severity):
        assert decision_to_event(_decision(action=action)).severity == severity

    def test_every_call_gets_a_fresh_event_id(self):
        ids = {decision_to_event(_decision()).event_id for _ in range(5)}
        assert len(ids) == 5


@pytest.mark.skipif(not AUDIT_FIXTURE.exists(), reason="AGT audit fixture not present")
class TestAuditEntryToEvent:
    def test_real_agt_4_1_audit_entry(self):
        entry = json.loads(AUDIT_FIXTURE.read_text())
        event = audit_entry_to_event(entry)

        assert event.event_id == "audit_4aa6d19f98db44be"
        assert event.occurred_at == "2026-09-16T11:13:34.207503Z"
        assert event.kind == "policy.evaluation"
        assert event.agent_id == "did:mesh:aos-control-spike-agent"
        assert event.action == "delete"
        assert event.resource is None
        assert event.decision == "deny"
        assert event.severity == "warning"
        assert event.rule == "deny-destructive-tool-calls"  # from data.rule, matched_rule is null
        assert event.previous_hash == entry["previous_hash"]
        assert event.hash == entry["entry_hash"]
        assert event.trace_id is None and event.session_id is None
        assert event.policy_version is None
        assert event.source == "agt-audit-log"
        assert event.attributes["event_type"] == "policy_evaluation"
        assert event.attributes["reason"].startswith("Destructive actions")
        assert event.attributes["evaluation_ms"] == 0.046
        assert "rule" not in event.attributes
        for absent in ("issued_at", "approver_did", "sandbox_id", "entry_hash", "previous_hash"):
            assert absent not in event.attributes

        wire = event.to_wire()
        assert wire["previous_hash"] == entry["previous_hash"]
        assert wire["hash"] == entry["entry_hash"]
        assert "trace_id" not in wire

    def test_accepts_pydantic_like_models_and_datetimes(self):
        raw = json.loads(AUDIT_FIXTURE.read_text())
        raw["timestamp"] = datetime(2026, 9, 16, 11, 13, 34, tzinfo=timezone.utc)
        raw["sandbox_id"] = "sbx-1"

        class Model:
            def model_dump(self, mode="python"):
                return dict(raw)

        event = audit_entry_to_event(Model())
        assert event.occurred_at == "2026-09-16T11:13:34Z"
        assert event.attributes["sandbox_id"] == "sbx-1"


# --------------------------------------------------------------------------- #
# Shipper
# --------------------------------------------------------------------------- #
class TestShipper:
    def test_posts_batch_with_basic_auth_to_contract_url(self, fake_http):
        shipper = _shipper()
        events = [GovernanceEvent(kind="policy.evaluation", event_id=f"evt-{i}", decision="allow") for i in range(3)]
        for e in events:
            assert shipper.emit(e) is True
        assert shipper.flush(timeout=5) is True
        shipper.shutdown()

        assert len(fake_http.posts) == 1
        call = fake_http.posts[0]
        assert call.url == EXPECTED_URL
        assert call.headers["Authorization"] == EXPECTED_AUTH
        assert call.timeout == shipper.timeout_s
        body = call.json
        assert list(body) == ["events"]
        assert [e["event_id"] for e in body["events"]] == ["evt-0", "evt-1", "evt-2"]
        assert all(set(e) >= {"event_id", "occurred_at", "kind"} for e in body["events"])
        assert shipper.stats["sent"] == 3 and shipper.stats["requests"] == 1

    def test_idempotent_event_id_survives_a_resend(self, fake_http):
        fake_http.queue(FakeResponse(503), FakeResponse(200, {"accepted": 0, "duplicates": 1, "rejected": []}))
        shipper = _shipper()
        shipper.emit(GovernanceEvent(kind="k", event_id="same-id"))
        assert shipper.flush(timeout=5)
        shipper.shutdown()
        ids = [c.json["events"][0]["event_id"] for c in fake_http.posts]
        assert ids == ["same-id", "same-id"]
        assert shipper.stats["duplicates"] == 1

    def test_splits_into_batches_of_batch_size(self, fake_http):
        shipper = _shipper(batch_size=100, flush_interval_s=5.0)
        for i in range(250):
            shipper.emit(GovernanceEvent(kind="k", event_id=str(i)))
        assert shipper.flush(timeout=5)
        shipper.shutdown()
        sizes = [len(c.json["events"]) for c in fake_http.posts]
        assert sum(sizes) == 250 and max(sizes) <= 100 and len(sizes) >= 3
        sent_ids = [e["event_id"] for c in fake_http.posts for e in c.json["events"]]
        assert sent_ids == [str(i) for i in range(250)]  # order preserved

    def test_batch_size_is_capped_at_the_contract_maximum(self):
        assert _shipper(batch_size=10_000).batch_size == 500

    def test_retries_5xx_and_connection_errors_then_succeeds(self, fake_http):
        fake_http.queue(FakeResponse(500, text="boom"), ConnectionError("reset"), FakeResponse(429))
        shipper = _shipper(max_retries=5)
        shipper.emit(GovernanceEvent(kind="k", event_id="x"))
        assert shipper.flush(timeout=5)
        shipper.shutdown()
        assert len(fake_http.posts) == 4
        assert shipper.stats["sent"] == 1 and shipper.stats["dropped"] == 0

    def test_does_not_retry_4xx(self, fake_http, caplog):
        fake_http.queue(FakeResponse(400, {"error": "bad_request"}))
        shipper = _shipper()
        shipper.emit(GovernanceEvent(kind="k", event_id="x"))
        with caplog.at_level("WARNING", logger="agentops.governance_events"):
            assert shipper.flush(timeout=5)
        shipper.shutdown()
        assert len(fake_http.posts) == 1
        assert shipper.stats["dropped"] == 1 and shipper.stats["sent"] == 0
        assert "not retrying" in caplog.text

    def test_gives_up_after_max_retries_without_raising(self, fake_http, caplog):
        fake_http.queue(*[FakeResponse(503)] * 10)
        shipper = _shipper(max_retries=2)
        shipper.emit(GovernanceEvent(kind="k", event_id="x"))
        with caplog.at_level("WARNING", logger="agentops.governance_events"):
            assert shipper.flush(timeout=5)
        shipper.shutdown()
        assert len(fake_http.posts) == 3  # 1 + 2 retries
        assert shipper.stats["dropped"] == 1 and shipper.stats["failed_batches"] == 1
        assert "giving up" in caplog.text

    def test_transport_exceptions_never_reach_the_caller(self, fake_http):
        class Weird(Exception):
            pass

        fake_http.queue(*[Weird("bug")] * 10)
        shipper = _shipper(max_retries=1)
        assert shipper.emit(GovernanceEvent(kind="k")) is True
        assert shipper.flush(timeout=5)
        shipper.shutdown()
        assert shipper.stats["dropped"] == 1

    def test_rejected_entries_are_counted_and_logged(self, fake_http, caplog):
        fake_http.queue(
            FakeResponse(200, {"accepted": 1, "duplicates": 0, "rejected": [{"index": 1, "error": "kind: Required"}]})
        )
        shipper = _shipper()
        shipper.emit(GovernanceEvent(kind="k", event_id="a"))
        shipper.emit(GovernanceEvent(kind="k", event_id="b"))
        with caplog.at_level("WARNING", logger="agentops.governance_events"):
            assert shipper.flush(timeout=5)
        shipper.shutdown()
        assert shipper.stats == {
            **shipper.stats,
            "accepted": 1,
            "rejected": 1,
            "sent": 2,
        }
        assert "rejected 1 of 2" in caplog.text

    def test_drops_when_queue_is_full(self, fake_http, caplog):
        shipper = _shipper(max_queue=2, batch_size=100, flush_interval_s=30.0)
        with caplog.at_level("WARNING", logger="agentops.governance_events"):
            assert shipper.emit(GovernanceEvent(kind="k", event_id="1"))
            assert shipper.emit(GovernanceEvent(kind="k", event_id="2"))
            assert shipper.emit(GovernanceEvent(kind="k", event_id="3")) is False
        assert shipper.stats["dropped"] == 1
        assert "queue full" in caplog.text
        shipper.shutdown(timeout=5)
        assert [e["event_id"] for c in fake_http.posts for e in c.json["events"]] == ["1", "2"]

    def test_shutdown_drains_the_queue(self, fake_http):
        shipper = _shipper(flush_interval_s=30.0)
        for i in range(5):
            shipper.emit(GovernanceEvent(kind="k", event_id=str(i)))
        shipper.shutdown(timeout=5)
        assert len(fake_http.posts) == 1 and len(fake_http.posts[0].json["events"]) == 5
        assert shipper.emit(GovernanceEvent(kind="k")) is False  # stopped

    def test_flush_interval_sends_partial_batches(self, fake_http):
        shipper = _shipper(batch_size=100, flush_interval_s=0.05)
        shipper.emit(GovernanceEvent(kind="k", event_id="1"))
        deadline = time.monotonic() + 3
        while not fake_http.posts and time.monotonic() < deadline:
            time.sleep(0.01)
        shipper.shutdown()
        assert len(fake_http.posts) == 1

    def test_missing_credentials_disable_shipping_with_a_warning(self, fake_http, monkeypatch, caplog):
        for var in ("AGENTOPS_HOST", "AGENTOPS_BASE_URL", "AGENTOPS_PUBLIC_KEY", "AGENTOPS_SECRET_KEY"):
            monkeypatch.delenv(var, raising=False)
        with caplog.at_level("WARNING", logger="agentops.governance_events"):
            shipper = GovernanceEventShipper()
        assert shipper.enabled is False
        assert "AGENTOPS_HOST" in caplog.text and "AGENTOPS_SECRET_KEY" in caplog.text
        assert shipper.emit(GovernanceEvent(kind="k")) is False
        assert shipper.flush() is True
        assert fake_http.posts == []

    def test_credentials_from_client_dict_and_env(self, fake_http, monkeypatch):
        class Client:
            _host = HOST
            _public_key = "pk-test"
            _secret_key = "sk-test"

        assert GovernanceEventShipper(Client()).url == EXPECTED_URL
        assert GovernanceEventShipper({"host": HOST, "public_key": "pk", "secret_key": "sk"}).enabled
        monkeypatch.setenv("AGENTOPS_HOST", "https://env.example.test")
        monkeypatch.setenv("AGENTOPS_PUBLIC_KEY", "pk-env")
        monkeypatch.setenv("AGENTOPS_SECRET_KEY", "sk-env")
        env_shipper = GovernanceEventShipper()
        assert env_shipper.url == "https://env.example.test/api/public/governance-events"
        assert env_shipper._headers["Authorization"] == "Basic " + base64.b64encode(b"pk-env:sk-env").decode()

    def test_emit_accepts_plain_dicts(self, fake_http):
        shipper = _shipper()
        assert shipper.emit({"kind": "custom.event", "event_id": "d1", "attributes": {"a": 1}})
        assert shipper.flush(timeout=5)
        shipper.shutdown()
        sent = fake_http.posts[0].json["events"][0]
        assert sent["kind"] == "custom.event" and sent["attributes"] == {"a": 1}


# --------------------------------------------------------------------------- #
# AuditLog export
# --------------------------------------------------------------------------- #
class _FakeChain:
    def __init__(self):
        self._entries: list[dict] = []


class FakeAuditLog:
    """Mimics AGT 4.1.0's AuditLog storage (``_chain._entries``)."""

    def __init__(self):
        self._chain = _FakeChain()

    def log(self, entry_id: str, **overrides) -> dict:
        base = json.loads(AUDIT_FIXTURE.read_text()) if AUDIT_FIXTURE.exists() else {
            "event_type": "policy_evaluation",
            "agent_did": AGENT_ID,
            "action": "delete",
            "data": {"rule": RULE_NAME},
            "outcome": "deny",
            "policy_decision": "deny",
            "timestamp": "2026-09-16T11:13:34.207503Z",
        }
        base.update(entry_id=entry_id, **overrides)
        self._chain._entries.append(base)
        return base


class TestShipAuditLog:
    def test_ships_only_new_entries_across_calls(self, fake_http):
        shipper = _shipper(flush_interval_s=30.0)
        log = FakeAuditLog()
        log.log("audit_1")
        log.log("audit_2")
        assert ship_audit_log(log, shipper) == 2
        assert ship_audit_log(log, shipper) == 0
        log.log("audit_3")
        assert ship_audit_log(log, shipper) == 1
        shipper.shutdown(timeout=5)
        sent = [e["event_id"] for c in fake_http.posts for e in c.json["events"]]
        assert sent == ["audit_1", "audit_2", "audit_3"]
        assert all(e["source"] == "agt-audit-log" for c in fake_http.posts for e in c.json["events"])

    def test_after_entry_id_overrides_the_cursor(self, fake_http):
        shipper = _shipper(flush_interval_s=30.0)
        log = FakeAuditLog()
        for i in range(4):
            log.log(f"audit_{i}")
        assert ship_audit_log(log, shipper, after_entry_id="audit_1") == 2
        shipper.shutdown(timeout=5)
        sent = [e["event_id"] for c in fake_http.posts for e in c.json["events"]]
        assert sent == ["audit_2", "audit_3"]

    def test_accepts_entries_attribute(self, fake_http):
        class Other:
            entries = [{"entry_id": "e1", "event_type": "approval", "outcome": "success"}]

        shipper = _shipper(flush_interval_s=30.0)
        assert ship_audit_log(Other(), shipper) == 1
        shipper.shutdown(timeout=5)
        event = fake_http.posts[0].json["events"][0]
        assert event["kind"] == "approval" and "decision" not in event

    def test_rejects_non_audit_logs(self):
        with pytest.raises(TypeError):
            ship_audit_log(object(), _shipper())

    def test_sink_forwards_each_written_entry(self, fake_http):
        shipper = _shipper(flush_interval_s=30.0)
        sink = AgentOpsAuditSink(shipper)
        log = FakeAuditLog()
        sink.write(log.log("audit_a"))
        sink.write_batch([log.log("audit_b"), log.log("audit_c")])
        assert sink.verify_integrity() == (True, None)
        sink.close()  # flushes
        shipper.shutdown()
        sent = [e["event_id"] for c in fake_http.posts for e in c.json["events"]]
        assert sent == ["audit_a", "audit_b", "audit_c"]


# --------------------------------------------------------------------------- #
# agentops.governance integration
# --------------------------------------------------------------------------- #
class TestGovernanceIntegration:
    def test_pre_tool_call_emits_a_decision_event(self, fake_http, policy):
        shipper = _shipper(flush_interval_s=30.0)
        decision = pre_tool_call(
            policy,
            AGENT_ID,
            "delete_file",
            {"path": "/x"},
            shipper=shipper,
            policy_version=7,
            session_id="sess-1",
        )
        assert decision.denied
        shipper.shutdown(timeout=5)
        events = fake_http.posts[0].json["events"]
        assert len(events) == 1
        event = events[0]
        assert event["kind"] == "policy.evaluation"
        assert event["decision"] == "deny"
        assert event["severity"] == "warning"
        assert event["rule"] == RULE_NAME
        assert event["policy_name"] == POLICY_NAME
        assert event["policy_version"] == "7"
        assert event["resource"] == "delete_file"
        assert event["action"] == "delete_file"  # action.type defaulted to the tool name
        assert event["agent_id"] == AGENT_ID
        assert event["session_id"] == "sess-1"
        assert event["attributes"]["stage"] == "pre_tool"
        assert event["attributes"]["allowed"] is False

    def test_default_shipper_is_used_when_none_is_passed(self, fake_http, policy):
        shipper = _shipper(flush_interval_s=30.0)
        set_default_shipper(shipper)
        try:
            pre_tool_call(policy, AGENT_ID, "read_file", {"path": "/x"})
            pre_tool_call(policy, AGENT_ID, "delete_file", {"path": "/x"})
        finally:
            set_default_shipper(None)
        shipper.shutdown(timeout=5)
        decisions = [e["decision"] for c in fake_http.posts for e in c.json["events"]]
        assert decisions == ["allow", "deny"]

    def test_no_shipper_means_no_http(self, fake_http, policy):
        pre_tool_call(policy, AGENT_ID, "delete_file", {"path": "/x"})
        assert fake_http.posts == []

    def test_shipping_failures_never_break_the_decision(self, fake_http, policy):
        class Broken:
            def emit(self, event):
                raise RuntimeError("shipper exploded")

        decision = pre_tool_call(policy, AGENT_ID, "delete_file", {"path": "/x"}, shipper=Broken())
        assert decision.denied

    @needs_otel
    def test_event_carries_the_ids_of_the_agt_span(self, fake_http, fake_agt, policy):
        provider = otel_trace.get_tracer_provider()
        if not hasattr(provider, "add_span_processor"):
            provider = TracerProvider()
            otel_trace.set_tracer_provider(provider)
            provider = otel_trace.get_tracer_provider()
        exporter = InMemorySpanExporter()
        processor = SimpleSpanProcessor(exporter)
        provider.add_span_processor(processor)
        try:
            fake_agt.enable_otel(service_name="events-test")
            shipper = _shipper(flush_interval_s=30.0)
            pre_tool_call(policy, AGENT_ID, "delete_file", {"path": "/x"}, shipper=shipper)
            shipper.shutdown(timeout=5)
        finally:
            processor.shutdown()

        spans = [s for s in exporter.get_finished_spans() if s.name == governance.SPAN_NAME]
        assert len(spans) == 1
        ctx = spans[0].get_span_context()
        event = fake_http.posts[0].json["events"][0]
        assert event["trace_id"] == format(ctx.trace_id, "032x")
        assert event["span_id"] == format(ctx.span_id, "016x")

    @needs_otel
    def test_enable_agt_export_installs_a_default_shipper(self, fake_agt):
        provider = TracerProvider(shutdown_on_exit=False)
        enable_agt_export(
            "support-agent", host=HOST, public_key="pk-test", secret_key="sk-test", provider=provider
        )
        shipper = governance.get_default_shipper()
        assert isinstance(shipper, GovernanceEventShipper)
        assert shipper.url == EXPECTED_URL
        assert shipper._headers["Authorization"] == EXPECTED_AUTH
        # Idempotent: a second call keeps the same shipper.
        enable_agt_export(
            "support-agent", host=HOST, public_key="pk-test", secret_key="sk-test", provider=provider
        )
        assert governance.get_default_shipper() is shipper

    @needs_otel
    def test_enable_agt_export_can_skip_or_take_a_custom_shipper(self, fake_agt, monkeypatch):
        provider = TracerProvider(shutdown_on_exit=False)
        enable_agt_export(host=HOST, public_key="pk", secret_key="sk", provider=provider, ship_events=False)
        assert governance.get_default_shipper() is None
        mine = _shipper()
        enable_agt_export(host=HOST, public_key="pk", secret_key="sk", provider=provider, shipper=mine)
        assert governance.get_default_shipper() is mine
        governance._reset_for_tests()
        # Missing credentials: no shipper either (warn-and-continue).
        for var in ("AGENTOPS_HOST", "AGENTOPS_BASE_URL", "AGENTOPS_PUBLIC_KEY", "AGENTOPS_SECRET_KEY"):
            monkeypatch.delenv(var, raising=False)
        enable_agt_export(provider=provider)
        assert governance.get_default_shipper() is None


# --------------------------------------------------------------------------- #
# Real AGT (only when agent-governance-toolkit-core is importable)
# --------------------------------------------------------------------------- #
class TestRealAgtAuditLog:
    @pytest.fixture
    def real_agt(self):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            gov = pytest.importorskip("agentmesh.governance")
        governance._reset_for_tests()
        yield gov
        governance._reset_for_tests()

    def test_ships_real_audit_log_entries_with_hash_chain(self, real_agt, fake_http):
        audit_log = real_agt.AuditLog()
        first = audit_log.log(
            "policy_evaluation",
            AGENT_ID,
            "delete",
            data={"rule": RULE_NAME, "reason": "not permitted", "evaluation_ms": 0.05},
            outcome="deny",
            policy_decision="deny",
        )
        second = audit_log.log("policy_evaluation", AGENT_ID, "read", outcome="allow", policy_decision="allow")

        shipper = _shipper(flush_interval_s=30.0)
        assert ship_audit_log(audit_log, shipper) == 2
        assert ship_audit_log(audit_log, shipper) == 0
        shipper.shutdown(timeout=5)

        events = [e for c in fake_http.posts for e in c.json["events"]]
        assert [e["event_id"] for e in events] == [first.entry_id, second.entry_id]
        assert events[0]["decision"] == "deny" and events[0]["rule"] == RULE_NAME
        assert events[0]["hash"] == first.entry_hash
        assert events[1]["previous_hash"] == first.entry_hash  # chain preserved
        assert events[1]["hash"] == second.entry_hash
        assert events[0]["kind"] == "policy.evaluation"
        assert events[0]["attributes"]["reason"] == "not permitted"
        assert events[0]["agent_id"] == AGENT_ID

    def test_sink_receives_entries_written_by_agt(self, real_agt, fake_http):
        shipper = _shipper(flush_interval_s=30.0)
        audit_log = real_agt.AuditLog(sink=AgentOpsAuditSink(shipper))
        entry = audit_log.log("policy_evaluation", AGENT_ID, "delete", outcome="deny", policy_decision="deny")
        shipper.shutdown(timeout=5)
        sent = fake_http.posts[0].json["events"][0]
        assert sent["event_id"] == entry.entry_id and sent["hash"] == entry.entry_hash
