"""AGT spike: run Microsoft's Agent Governance Toolkit against a local OTLP
capture and dump the emitted spans + audit record as JSON fixtures.

What this proves (AGT 4.1.0):

1. ``agentmesh.governance.enable_otel()`` installs a global ``TracerProvider``
   but attaches NO span processor/exporter; its ``endpoint`` argument is
   accepted and ignored. Nothing leaves the process until the caller adds an
   exporter to the provider.
2. ``agentmesh.governance.govern()`` (and ``PolicyEngine.evaluate``) emit no
   spans at all. Spans only appear when the caller wraps evaluation in the
   public ``trace_policy_evaluation`` context manager.
3. The span emitted is named ``agt.policy.evaluate`` with attributes
   ``agt.agent.id``, ``agt.policy.stage``, ``agt.policy.action``,
   ``agt.policy.rule`` and (only when a rule matched) ``agt.policy.name``.

Usage::

    python -m venv .venv && . .venv/bin/activate
    pip install -r requirements.txt
    python agt_spike.py --out ./fixtures

Fixtures written to ``--out``: policy-evaluate-allow.json,
policy-evaluate-deny.json, approval-request.json, trust-verify.json,
governance-event.json, governance-event.cloudevent.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

from google.protobuf.json_format import MessageToDict
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
    ExportTraceServiceRequest,
)
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

# --- The real AGT imports (verified against agent-governance-toolkit==4.1.0) ---
from agentmesh.governance import (
    GovernanceDenied,
    enable_otel,
    govern,
    trace_approval,
    trace_policy_evaluation,
    trace_trust_verification,
)

HERE = Path(__file__).resolve().parent
POLICY_FILE = HERE / "policy.yaml"
AGENT_ID = "did:mesh:aos-control-spike-agent"
SERVICE_NAME = "agt-spike-governed-agent"

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("agt_spike")


# --------------------------------------------------------------------------- #
# Local OTLP/HTTP capture: decodes protobuf ExportTraceServiceRequest bodies
# into OTLP/JSON (camelCase, hex trace/span ids) exactly like a collector.
# --------------------------------------------------------------------------- #
class _Capture(BaseHTTPRequestHandler):
    requests: list[dict[str, Any]] = []

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        req = ExportTraceServiceRequest()
        req.ParseFromString(body)
        as_json = MessageToDict(req, preserving_proto_field_name=False)
        _hexify_ids(as_json)
        _Capture.requests.append(as_json)
        self.send_response(200)
        self.send_header("Content-Type", "application/x-protobuf")
        self.end_headers()

    def log_message(self, *_args: Any) -> None:  # silence
        return


def _hexify_ids(payload: dict[str, Any]) -> None:
    """protobuf JSON encodes bytes as base64; OTLP/JSON mandates lowercase hex."""
    import base64

    for rs in payload.get("resourceSpans", []):
        for ss in rs.get("scopeSpans", []):
            for span in ss.get("spans", []):
                for key in ("traceId", "spanId", "parentSpanId"):
                    if key in span and span[key]:
                        span[key] = base64.b64decode(span[key]).hex()


def start_capture() -> tuple[HTTPServer, str]:
    server = HTTPServer(("127.0.0.1", 0), _Capture)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    endpoint = f"http://127.0.0.1:{server.server_address[1]}/v1/traces"
    return server, endpoint


def captured_spans() -> list[dict[str, Any]]:
    out = []
    for req in _Capture.requests:
        for rs in req.get("resourceSpans", []):
            for ss in rs.get("scopeSpans", []):
                out.extend(ss.get("spans", []))
    return out


# --------------------------------------------------------------------------- #
# The "agent tool" under governance
# --------------------------------------------------------------------------- #
def execute_tool(action: str, tool: str, path: str) -> str:
    return f"{tool}({path}) executed"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, help="fixture output directory")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    server, endpoint = start_capture()
    log.info("OTLP capture listening at %s", endpoint)

    # Step 1 — the documented quickstart line. NOTE: endpoint is ignored by AGT 4.1.0.
    enable_otel(service_name=SERVICE_NAME, endpoint=endpoint)

    governed = govern(
        execute_tool,
        policy=str(POLICY_FILE),
        agent_id=AGENT_ID,
        audit=True,
        conflict_strategy="deny_overrides",
    )

    # Step 2 — prove govern() emits nothing even though enable_otel() ran.
    governed(action="read", tool="read_file", path="/data/report.csv")
    try:
        governed(action="delete", tool="delete_file", path="/data/report.csv")
    except GovernanceDenied as e:
        log.info("denied as expected: %s", e)
    trace.get_tracer_provider().force_flush()  # type: ignore[attr-defined]
    log.info("spans exported after govern() with enable_otel() only: %d", len(captured_spans()))
    assert not captured_spans(), "unexpected: govern() exported spans"

    # Step 3 — attach an exporter to the provider enable_otel() installed.
    provider = trace.get_tracer_provider()
    provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))  # type: ignore[attr-defined]

    governed(action="read", tool="read_file", path="/data/report.csv")
    provider.force_flush()  # type: ignore[attr-defined]
    log.info("spans exported after govern() with exporter attached: %d", len(captured_spans()))
    assert not captured_spans(), "unexpected: govern() exported spans"

    # Step 4 — the only way to get agt.policy.evaluate spans in 4.1.0: wrap the
    # engine evaluation yourself with the public trace_policy_evaluation helper.
    engine = governed.engine
    ctx_allow = {"action": {"type": "read"}, "tool": {"value": "read_file"}, "path": {"value": "/data/report.csv"}}
    ctx_deny = {"action": {"type": "delete"}, "tool": {"value": "delete_file"}, "path": {"value": "/data/report.csv"}}

    def traced_evaluate(context: dict[str, Any]):
        with trace_policy_evaluation(agent_id=AGENT_ID, stage="pre_tool", context=context) as result:
            decision = engine.evaluate(AGENT_ID, context, stage="pre_tool")
            result["action"] = decision.action
            result["rule"] = decision.matched_rule
            result["policy_name"] = decision.policy_name
            result["allowed"] = decision.allowed
        return decision

    allow_decision = traced_evaluate(ctx_allow)
    provider.force_flush()  # type: ignore[attr-defined]
    allow_req = _Capture.requests[-1]

    deny_decision = traced_evaluate(ctx_deny)
    provider.force_flush()  # type: ignore[attr-defined]
    deny_req = _Capture.requests[-1]

    # Extra helper-emitted spans the epic maps to EVENT (not produced by govern()).
    with trace_approval(agent_id=AGENT_ID, rule_name="deny-destructive-tool-calls") as approval:
        approval["outcome"] = "rejected"
        approval["approver"] = "auto-reject"
    provider.force_flush()  # type: ignore[attr-defined]
    approval_req = _Capture.requests[-1]

    with trace_trust_verification(agent_id=AGENT_ID) as trust:
        trust["score"] = 500
        trust["tier"] = "standard"
    provider.force_flush()  # type: ignore[attr-defined]
    trust_req = _Capture.requests[-1]

    # Step 5 — the audit record govern() DID write for the denied call.
    audit = governed.audit_log
    assert audit is not None
    denied_entries = [e for e in audit.query(limit=100) if e.policy_decision == "deny"]
    assert denied_entries, "expected a denied audit entry"
    deny_entry = denied_entries[0]

    fixtures = {
        "policy-evaluate-allow.json": allow_req,
        "policy-evaluate-deny.json": deny_req,
        "approval-request.json": approval_req,
        "trust-verify.json": trust_req,
        "governance-event.json": deny_entry.model_dump(mode="json"),
        "governance-event.cloudevent.json": deny_entry.to_cloudevent(),
    }
    for name, payload in fixtures.items():
        text = json.dumps(payload, indent=2, sort_keys=False) + "\n"
        text = _scrub(text)
        (out / name).write_text(text)
        log.info("wrote %s", out / name)

    # Summary of every agt.* key observed.
    keys: dict[str, set[str]] = {}
    for req in (allow_req, deny_req, approval_req, trust_req):
        for rs in req["resourceSpans"]:
            for ss in rs["scopeSpans"]:
                for span in ss["spans"]:
                    for attr in span.get("attributes", []):
                        keys.setdefault(span["name"], set()).add(attr["key"])
    print("\nallow decision:", allow_decision.model_dump(mode="json"))
    print("deny decision: ", deny_decision.model_dump(mode="json"))
    print("\nobserved span attribute keys:")
    for span_name, attr_keys in keys.items():
        print(f"  {span_name}: {sorted(attr_keys)}")

    server.shutdown()
    return 0


def _scrub(text: str) -> str:
    """Remove absolute paths / usernames that could leak from the capture host."""
    text = re.sub(r"/(?:Users|home)/[^/\"\\s]+", "/home/user", text)
    text = re.sub(r"/private/tmp/[^\"\\s]*", "/tmp/scrubbed", text)
    user = os.environ.get("USER")
    if user:
        text = text.replace(user, "user")
    return text


if __name__ == "__main__":
    sys.exit(main())
