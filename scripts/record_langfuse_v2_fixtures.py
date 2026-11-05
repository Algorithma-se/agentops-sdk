"""Record real Langfuse-v2-SDK ingestion batches as contract fixtures.

Runs the actual ``langfuse`` v2 Python client (the one ``agentops-sdk`` 0.x
wraps) against a local capture server and writes every
``POST /api/public/ingestion`` request it sends to
``tests/fixtures/langfuse_v2_ingestion/<scenario>.json``. The web
servertest and the worker integration test replay these files, so the
server's v2 ingestion contract is checked against what the SDK really sends,
not against hand-written payloads.

Scenarios:

* ``lowlevel``: the handle-based API Base-Agent's
  ``agents-observability-agentops`` observer uses (``Langfuse().trace()``,
  ``.span()``, ``.generation()``, ``.update()``, ``.end()``, ``.event()``,
  ``.score()``), with both ``usage_details`` and the legacy ``usage`` shape.
* ``decorator``: the ``@observe`` + ``update_current_trace`` /
  ``update_current_observation`` flow Securitas uses through ``agentops``.
* ``sdk_log``: an ``sdk-log`` event, which the v2 client only emits with
  ``PYTHONTRACEMALLOC=1`` and ``LANGFUSE_DEBUG_MEMORY_REPORT_INTERVAL`` set
  (run in a subprocess so the env applies from interpreter start).

Usage (from ``sdk/python``)::

    uv venv --python 3.12 .venv
    uv pip install --python .venv/bin/python -e . "langfuse==2.60.10"
    .venv/bin/python scripts/record_langfuse_v2_fixtures.py

Set ``AGENTOPS_FIXTURE_OUT_DIR`` to write somewhere else (the drift test in
``tests/test_langfuse_v2_fixtures.py`` does), and ``--only a,b`` to record a
subset of scenarios.

Recorded values are real except for normalisation that keeps the fixtures
portable: local file paths in the ``sdk-log`` memory report are reduced to
their basename, and the host/port of the capture server is replaced with
``http://capture.invalid``.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
FIXTURE_DIR = Path(
    os.environ.get("AGENTOPS_FIXTURE_OUT_DIR")
    or HERE.parent / "tests" / "fixtures" / "langfuse_v2_ingestion"
)
PUBLIC_KEY = "pk-lf-recorder"
SECRET_KEY = "sk-lf-recorder"
KEPT_HEADERS = {
    "content-type",
    "x-langfuse-sdk-name",
    "x-langfuse-sdk-version",
    "x-langfuse-public-key",
    "x-fern-language",
    "x-fern-sdk-name",
    "x-fern-sdk-version",
}


class _Capture(BaseHTTPRequestHandler):
    requests: list[dict[str, Any]] = []
    unhandled: list[str] = []

    def log_message(self, *args: Any) -> None:  # silence stderr access log
        return

    def do_POST(self) -> None:  # noqa: N802 (stdlib naming)
        length = int(self.headers.get("content-length") or 0)
        raw = self.rfile.read(length) if length else b""
        if self.path != "/api/public/ingestion":
            type(self).unhandled.append(f"POST {self.path}")
            self.send_response(404)
            self.end_headers()
            return
        body = json.loads(raw or b"{}")
        auth = self.headers.get("authorization", "")
        type(self).requests.append(
            {
                "method": "POST",
                "path": self.path,
                "authScheme": auth.split(" ", 1)[0] if auth else None,
                "headers": {
                    k.lower(): v
                    for k, v in self.headers.items()
                    if k.lower() in KEPT_HEADERS
                },
                "body": body,
            }
        )
        # Answer like the real endpoint so the client doesn't retry.
        result = {
            "successes": [
                {"id": e.get("id"), "status": 201} for e in body.get("batch", [])
            ],
            "errors": [],
        }
        payload = json.dumps(result).encode()
        self.send_response(207)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:  # noqa: N802
        type(self).unhandled.append(f"GET {self.path}")
        self.send_response(404)
        self.end_headers()


def _serve() -> tuple[ThreadingHTTPServer, str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Capture)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


def _configure_env(host: str) -> None:
    os.environ.update(
        {
            "AGENTOPS_HOST": host,
            "AGENTOPS_PUBLIC_KEY": PUBLIC_KEY,
            "AGENTOPS_SECRET_KEY": SECRET_KEY,
            "LANGFUSE_HOST": host,
            "LANGFUSE_PUBLIC_KEY": PUBLIC_KEY,
            "LANGFUSE_SECRET_KEY": SECRET_KEY,
            "LANGFUSE_RELEASE": "contract-fixture",
            # Same batching for the @observe singleton as the explicit
            # clients below: one request per scenario.
            "LANGFUSE_FLUSH_AT": "1000",
            "LANGFUSE_FLUSH_INTERVAL": "2",
        }
    )


# --------------------------------------------------------------------------
# Scenarios. Each runs in the current process and must flush before return.
# --------------------------------------------------------------------------


def scenario_lowlevel() -> None:
    """Base-Agent observer shape: handle-based client objects."""
    from langfuse import Langfuse

    lf = Langfuse(
        host=os.environ["LANGFUSE_HOST"],
        public_key=PUBLIC_KEY,
        secret_key=SECRET_KEY,
        # One request per scenario keeps the fixture readable and the
        # replay deterministic; the client still batches internally.
        flush_at=1000,
        flush_interval=2,
        environment="contract",
    )
    trace = lf.trace(
        name="support-turn",
        session_id="session-contract-1",
        user_id="user-contract-1",
        metadata={"agent": "support-agent", "channel": "chat"},
        tags=["contract", "v2"],
        input={"message": "Where is my order?"},
    )
    span = trace.span(
        name="retrieve-order",
        input={"orderRef": "A-1001"},
        metadata={"source": "orders-db"},
    )
    span.update(metadata={"rows": 1})
    span.end(output={"status": "shipped"})

    gen = trace.generation(
        name="answer",
        model="gpt-4o-mini",
        model_parameters={"temperature": 0.2, "max_tokens": 256},
        input=[
            {"role": "system", "content": "You are a support agent."},
            {"role": "user", "content": "Where is my order?"},
        ],
        metadata={"step": 1},
        parent_observation_id=span.id,
    )
    gen.update(completion_start_time=gen_start_time())
    gen.end(
        output={"role": "assistant", "content": "It shipped yesterday."},
        usage_details={"input": 42, "output": 7, "total": 49},
    )

    # Legacy v2 `usage` argument with an OpenAI-style usage dict, as
    # integrations that forward the provider response pass it.
    legacy = trace.generation(
        name="classify",
        model="gpt-4o-mini",
        input="Where is my order?",
    )
    legacy.end(
        output="order_status",
        usage={"prompt_tokens": 11, "completion_tokens": 2, "total_tokens": 13},
        cost_details={"input": 0.0000165, "output": 0.0000012},
    )

    trace.event(
        name="guardrail-check",
        input={"stage": "pre_input"},
        output={"action": "allow"},
        level="WARNING",
        status_message="jailbreak signal below threshold",
    )
    trace.score(name="helpfulness", value=0.9, comment="resolved")
    lf.score(
        trace_id=trace.id,
        observation_id=gen.id,
        name="tone",
        value="friendly",
        data_type="CATEGORICAL",
    )
    trace.update(output={"answer": "It shipped yesterday."})
    lf.flush()


def gen_start_time():
    from datetime import datetime, timezone

    return datetime.now(timezone.utc)


def scenario_decorator() -> None:
    """Securitas shape: @observe + update_current_* through agentops."""
    import agentops
    from agentops import AgentOps, observe

    client = AgentOps()

    @observe(as_type="generation", name="llm-call")
    def llm_call(prompt: str) -> str:
        client.update_current_observation(
            model="gpt-4o-mini",
            input=[{"role": "user", "content": prompt}],
            usage_details={"input": 20, "output": 5},
            model_parameters={"temperature": 0},
            metadata={"deployment": "contract"},
        )
        return "Hello from the agent"

    @observe(name="handle-message")
    def handle(message: str) -> str:
        client.update_current_trace(
            name="securitas-agent",
            session_id="session-contract-2",
            user_id="user-contract-2",
            metadata={"conversation": "c-1"},
            tags=["contract", "decorator"],
        )
        answer = llm_call(message)
        agentops.update_current_observation(output={"answer": answer})
        return answer

    handle("Hi there")
    client.flush()


def scenario_sdk_log() -> None:
    """Only emitted with tracemalloc on; see the module docstring."""
    from langfuse import Langfuse

    lf = Langfuse(
        host=os.environ["LANGFUSE_HOST"],
        public_key=PUBLIC_KEY,
        secret_key=SECRET_KEY,
        flush_at=1000,
        flush_interval=2,
    )
    lf.trace(name="sdk-log-carrier")
    lf.flush()


SCENARIOS = {
    "lowlevel": scenario_lowlevel,
    "decorator": scenario_decorator,
    "sdk_log": scenario_sdk_log,
}


def _normalise(value: Any, host: str) -> Any:
    if isinstance(value, dict):
        return {k: _normalise(v, host) for k, v in value.items()}
    if isinstance(value, list):
        return [_normalise(v, host) for v in value]
    if isinstance(value, str):
        value = value.replace(host, "http://capture.invalid")
        if os.sep in value and ("site-packages" in value or str(Path.home()) in value):
            # tracemalloc lines look like "<path>:<line>: size=..., count=..."
            head, sep, tail = value.partition(":")
            return f"{Path(head).name}{sep}{tail}"
    return value


def _run_child(name: str) -> None:
    server, host = _serve()
    _configure_env(host)
    SCENARIOS[name]()
    import langfuse

    requests = _normalise(_Capture.requests, host)
    out = {
        "scenario": name,
        "recordedWith": {
            "langfuse": getattr(langfuse, "__version__", "unknown"),
            "agentopsSdk": _agentops_version(),
            "python": platform.python_version(),
        },
        "unhandledRequests": _Capture.unhandled,
        "requests": requests,
    }
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIXTURE_DIR / f"{name}.json"
    path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    server.shutdown()
    events = [e["type"] for r in requests for e in r["body"].get("batch", [])]
    print(f"{name}: {len(requests)} request(s), events={events} -> {path}")


def _agentops_version() -> str:
    try:
        from importlib.metadata import version

        return version("agentops-sdk")
    except Exception:
        return "unknown"


def main() -> None:
    if len(sys.argv) == 2 and sys.argv[1] in SCENARIOS:
        _run_child(sys.argv[1])
        return
    # Each scenario gets a fresh interpreter: the v2 decorator keeps a
    # process-wide singleton, and sdk-log needs PYTHONTRACEMALLOC at start.
    # `--only a,b` records a subset (used by tests/test_langfuse_v2_fixtures.py).
    names = list(SCENARIOS)
    if len(sys.argv) == 3 and sys.argv[1] == "--only":
        names = [n for n in sys.argv[2].split(",") if n in SCENARIOS]
    for name in names:
        env = dict(os.environ)
        if name == "sdk_log":
            env.update(
                {
                    "PYTHONTRACEMALLOC": "1",
                    "LANGFUSE_DEBUG_MEMORY_REPORT_INTERVAL": "1",
                    "LANGFUSE_DEBUG_MEMORY_TOP_K": "3",
                }
            )
        subprocess.run([sys.executable, __file__, name], env=env, check=True)


if __name__ == "__main__":
    main()
