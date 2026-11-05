"""
Agent Governance Toolkit (AGT) + AOS Control — runnable example.

A plain-Python agent (no framework) that:

  1. loads an AGT policy from examples/agt_policy.yaml with `govern()`,
  2. evaluates every tool call at stage `pre_tool` inside AGT's
     `trace_policy_evaluation` span helper, so each decision becomes an
     `agt.policy.evaluate` span (`search_docs` is allowed,
     `delete_customer_record` is denied),
  3. ships those spans to AOS Control's OTLP/HTTP endpoint with Basic auth,
  4. records an AgentOps trace through the `agentops` SDK with the same
     `session_id`, so the SDK trace and the AGT spans appear in one session.

Prerequisites (Python >= 3.11):

    pip install "agent-governance-toolkit[full]==4.1.0" opentelemetry-exporter-otlp-proto-http
    pip install agentops-sdk
    export AGENTOPS_HOST=https://your-aos-control.example.com
    export AGENTOPS_PUBLIC_KEY=pk-lf-...
    export AGENTOPS_SECRET_KEY=sk-lf-...
    python sdk/python/examples/agt_governed_agent.py

Expected result in AOS Control: the denied call shows as a GUARDRAIL
observation (level ERROR) in the Trace Explorer and a CRITICAL alert with
stage `pre_tool` in Project -> Guardrail Alerts. See docs/guardrails.md,
section "Runtime tool-call policy with the Agent Governance Toolkit".

Verified against agent-governance-toolkit==4.1.0 (spike, 2026-09-16):
  - Import surface: `agentmesh.governance` (govern, GovernanceDenied,
    enable_otel, trace_policy_evaluation). Importing it logs
    `DeprecationWarning: agentmesh-platform is deprecated`; expected, harmless.
  - `enable_otel()` installs a TracerProvider but attaches NO exporter and
    ignores its `endpoint` argument; the exporter is added below.
  - `govern()` itself emits NO spans. Spans exist only because this file wraps
    `engine.evaluate(...)` in `trace_policy_evaluation(...)`.
  - The evaluate span carries `agt.agent.id`, `agt.policy.stage`,
    `agt.policy.action`, `agt.policy.rule` ("none" when no rule matched) and
    `agt.policy.name` (absent when no rule matched). It carries no tool name
    or arguments, so this file sets `agt.tool.name` / `agt.tool.args` itself.
  - Policy schema in agt_policy.yaml and the kwargs -> context mapping.
Verified against this repository:
  - OTLP endpoint and auth (web/src/pages/api/public/otel/v1/traces):
    POST {AGENTOPS_HOST}/api/public/otel/v1/traces,
    Authorization: Basic base64("<public key>:<secret key>").
  - AGENTOPS_* env resolution (sdk/python/src/agentops/_env.py) and
    `AgentOps().trace(...)` / `.update()` / `.flush()`.

Recommended path once merged: the `agentops[governance]` extra
(`agentops.governance.governed` / `enable_agt_export`) does steps 2-3 in one
call and stamps `agt.tool.name` / `agt.tool.args` for you. This file shows
what that helper does under the hood.
"""

from __future__ import annotations

import base64
import json
import os
import sys
import uuid
import warnings
from pathlib import Path
from typing import Any

# Expected on `import agentmesh` in 4.1.0; silence it so the run output is clean.
warnings.filterwarnings(
    "ignore", message=".*agentmesh-platform is deprecated.*", category=DeprecationWarning
)

try:
    from agentmesh.governance import (
        GovernanceDenied,
        enable_otel,
        govern,
        trace_policy_evaluation,
    )
except ImportError:
    sys.exit(
        "Agent Governance Toolkit not found (module 'agentmesh.governance').\n"
        'Install it with: pip install "agent-governance-toolkit[full]==4.1.0" '
        "opentelemetry-exporter-otlp-proto-http\n"
        "Note: 'agent-control-specification' on PyPI is a different project, not AGT."
    )

try:
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
except ImportError:
    sys.exit(
        "OTLP exporter not installed ([full] does not include one). Run: "
        "pip install opentelemetry-exporter-otlp-proto-http"
    )

# Reuse the SDK's env resolution (AGENTOPS_HOST / AGENTOPS_BASE_URL fallbacks).
try:
    from agentops._env import get_host, get_public_key, get_secret_key
except ImportError:  # SDK not installed; the OTel half still works on its own

    def get_host() -> str | None:
        return os.getenv("AGENTOPS_HOST") or os.getenv("AGENTOPS_BASE_URL")

    def get_public_key() -> str | None:
        return os.getenv("AGENTOPS_PUBLIC_KEY")

    def get_secret_key() -> str | None:
        return os.getenv("AGENTOPS_SECRET_KEY")


AGENT_ID = "support-agent"
SERVICE_NAME = "agt-governed-agent"
POLICY_PATH = Path(__file__).with_name("agt_policy.yaml")


# ----------------------------------------------------------------------
# 1. OpenTelemetry -> AOS Control
# ----------------------------------------------------------------------
def aos_otlp_settings() -> tuple[str, dict[str, str]]:
    """Return (endpoint, headers) for AOS Control's OTLP/HTTP traces endpoint."""
    host, pk, sk = get_host(), get_public_key(), get_secret_key()
    missing = [
        name
        for name, val in (
            ("AGENTOPS_HOST", host),
            ("AGENTOPS_PUBLIC_KEY", pk),
            ("AGENTOPS_SECRET_KEY", sk),
        )
        if not val
    ]
    if missing:
        sys.exit(f"Missing environment variable(s): {', '.join(missing)}")

    endpoint = host.rstrip("/") + "/api/public/otel/v1/traces"
    token = base64.b64encode(f"{pk}:{sk}".encode("utf-8")).decode("ascii")
    return endpoint, {"Authorization": f"Basic {token}"}


def configure_otel(service_name: str) -> Any:
    """Bootstrap AGT's tracer provider, then attach the AOS Control exporter.

    `enable_otel()` sets the global TracerProvider (resource service.name) but
    adds no span processor; without the `add_span_processor` call below every
    span is created and dropped.
    """
    enable_otel(service_name=service_name)
    endpoint, headers = aos_otlp_settings()
    provider = trace.get_tracer_provider()
    provider.add_span_processor(  # type: ignore[attr-defined]
        BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint, headers=headers))
    )
    return provider


# ----------------------------------------------------------------------
# 2. Fake tools behind one governed dispatcher
# ----------------------------------------------------------------------
def search_docs(query: str) -> str:
    return f"3 documents matched '{query}'."


def delete_customer_record(customer_id: str) -> str:
    return f"customer {customer_id} deleted"  # never reached under the policy


TOOLS = {"search_docs": search_docs, "delete_customer_record": delete_customer_record}


def execute_tool(action: str, tool: str, **params: Any) -> Any:
    """The callable AGT governs. `govern()` turns its kwargs into the policy
    context {"action": {"type": action}, "tool": {"value": tool}, ...}."""
    return TOOLS[tool](**params)


# ----------------------------------------------------------------------
# 3. Traced, governed tool call
# ----------------------------------------------------------------------
def governed_call(governed: Any, action: str, tool: str, **params: Any) -> Any:
    """Evaluate the policy inside an `agt.policy.evaluate` span, then run the
    governed callable.

    `govern()` re-evaluates the same policy (sub-millisecond) when called and
    writes the local audit entry; 4.1.0 has no hook to reuse the traced
    decision, so the evaluation happens twice. The `agentops[governance]`
    helper folds this into one pass.
    """
    context = {
        "action": {"type": action},
        "tool": {"value": tool},
        **{k: {"value": v} for k, v in params.items()},
    }

    with trace_policy_evaluation(agent_id=AGENT_ID, stage="pre_tool", context=context) as result:
        decision = governed.engine.evaluate(AGENT_ID, context, stage="pre_tool")
        result["action"] = decision.action
        result["rule"] = decision.matched_rule
        result["policy_name"] = decision.policy_name
        result["allowed"] = decision.allowed

        # AGT puts no tool identity on the span; add it so the Trace Explorer
        # can show which tool was evaluated and with what arguments.
        span = trace.get_current_span()
        span.set_attribute("agt.tool.name", tool)
        span.set_attribute("agt.tool.args", json.dumps(params, default=str))

    return governed(action=action, tool=tool, **params)  # raises GovernanceDenied on deny


# ----------------------------------------------------------------------
# 4. Run one governed turn
# ----------------------------------------------------------------------
def main() -> int:
    session_id = f"agt-demo-{uuid.uuid4().hex[:8]}"
    provider = configure_otel(SERVICE_NAME)

    governed = govern(
        execute_tool,
        policy=str(POLICY_PATH),
        agent_id=AGENT_ID,
        audit=True,
        conflict_strategy="deny_overrides",
    )

    # AgentOps SDK trace with the same session_id as the AGT spans.
    from agentops import AgentOps

    agentops = AgentOps()
    aos_trace = agentops.trace(
        name=SERVICE_NAME,
        session_id=session_id,
        input={"request": "Find the refund policy, then delete customer 42."},
        metadata={"agt.agent.id": AGENT_ID, "agt.policy.name": "support-agent-policy"},
    )

    tracer = trace.get_tracer(SERVICE_NAME)
    outcome: dict[str, Any] = {}

    # Parent span carrying session.id so the agt.policy.evaluate children join
    # the same session as the SDK trace.
    with tracer.start_as_current_span("agent.turn") as span:
        span.set_attribute("session.id", session_id)
        span.set_attribute("agt.agent.id", AGENT_ID)

        outcome["search_docs"] = governed_call(
            governed, action="read", tool="search_docs", query="refund policy"
        )
        print("allowed :", outcome["search_docs"])

        try:
            governed_call(governed, action="delete", tool="delete_customer_record", customer_id="42")
        except GovernanceDenied as exc:
            outcome["delete_customer_record"] = {"denied": True, "reason": str(exc)}
            print("denied  :", exc)

    aos_trace.update(output=outcome)
    agentops.flush()
    provider.force_flush()
    provider.shutdown()

    print(
        f"Done. In AOS Control open session '{session_id}': the deny is a GUARDRAIL "
        "observation (level ERROR) and a CRITICAL alert with stage pre_tool under "
        "Project -> Guardrail Alerts."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
