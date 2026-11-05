"""Ship governance events (policy decisions, AGT audit entries) to AOS Control.

Wire contract — ``POST {host}/api/public/governance-events`` (Basic auth
``public_key:secret_key``), body ``{"events": [...]}`` with 1–500 events per
batch, idempotent on ``event_id``, answering
``{"accepted": n, "duplicates": n, "rejected": [{"index", "error"}]}``.
Each event mirrors :class:`GovernanceEvent` (all fields optional except
``event_id``, ``occurred_at`` and ``kind``; ``attributes`` ≤ 16 kB).

Three producers feed it:

* :func:`decision_to_event` — a :class:`agentops.governance.Decision` from
  ``governed`` / ``pre_tool_call``, tagged with the OTel trace/span ids of
  the ``agt.policy.evaluate`` span so the event joins the trace in AOS
  Control. ``agentops.governance`` calls this automatically when a shipper
  is configured (``enable_agt_export(ship_events=True)`` or
  :func:`agentops.governance.set_default_shipper`).
* :func:`audit_entry_to_event` — a real AGT 4.1.0 ``AuditEntry`` (the record
  ``govern()`` writes to its ``AuditLog``; rule under ``data.rule``,
  ``trace_id`` null, hash chain in ``previous_hash`` / ``entry_hash``). The
  chain fields are preserved so AOS Control can verify the chain
  independently of the agent's local copy.
* :func:`ship_audit_log` / :class:`AgentOpsAuditSink` — bulk or real-time
  export of an AGT ``AuditLog``.

The :class:`GovernanceEventShipper` is best-effort by design: a background
daemon thread batches events, retries transport errors and 5xx with
backoff, never retries 4xx, drops (and logs) when the server keeps failing or
the queue is full, and never raises into the agent. ``flush()`` runs at
interpreter exit so short-lived processes do not lose their last batch.
"""

from __future__ import annotations

import atexit
import collections
import json
import logging
import os
import threading
import time
import uuid
import weakref
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Optional, Union

from . import _http
from ._env import get_host, get_public_key, get_secret_key

_log = logging.getLogger("agentops.governance_events")

GOVERNANCE_EVENTS_PATH = "/api/public/governance-events"
MAX_BATCH_SIZE = 500
EVENT_ID_MAX_LEN = 128
ATTRIBUTES_MAX_BYTES = 16 * 1024

KIND_POLICY_EVALUATION = "policy.evaluation"
DEFAULT_SOURCE = "agentops-sdk-python"
AUDIT_LOG_SOURCE = "agt-audit-log"

DECISIONS = ("allow", "deny", "warn", "require_approval", "log")
_WARNING_DECISIONS = frozenset({"deny", "require_approval"})


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def severity_for(decision: Optional[str]) -> str:
    return "warning" if decision in _WARNING_DECISIONS else "info"


def current_span_ids() -> tuple[Optional[str], Optional[str]]:
    """``(trace_id, span_id)`` of the current OTel span as lowercase hex
    (32 / 16 chars, the OTLP JSON spelling AOS Control stores), or
    ``(None, None)`` when OpenTelemetry is not installed or no span is active.
    """
    try:
        from opentelemetry import trace  # type: ignore[import-not-found]
    except ImportError:
        return None, None
    try:
        ctx = trace.get_current_span().get_span_context()
    except Exception:  # pragma: no cover - defensive
        return None, None
    if not ctx.is_valid:
        return None, None
    return format(ctx.trace_id, "032x"), format(ctx.span_id, "016x")


# --------------------------------------------------------------------------- #
# Event
# --------------------------------------------------------------------------- #
def _fit_attributes(attributes: dict[str, Any], max_bytes: int = ATTRIBUTES_MAX_BYTES) -> dict[str, Any]:
    """Keep ``attributes`` under the server's size limit by dropping the
    largest values first; records what was dropped under ``_dropped``."""

    def encoded(value: Any) -> bytes:
        return json.dumps(value, default=str, ensure_ascii=False).encode("utf-8")

    if len(encoded(attributes)) <= max_bytes:
        return attributes
    kept = dict(attributes)
    dropped: list[str] = []
    while kept and len(encoded({**kept, "_dropped": dropped})) > max_bytes:
        biggest = max(kept, key=lambda k: len(encoded(kept[k])))
        kept.pop(biggest)
        dropped.append(biggest)
    kept["_dropped"] = dropped
    return kept


def _clip_event_id(event_id: str) -> str:
    event_id = str(event_id)
    if len(event_id) <= EVENT_ID_MAX_LEN:
        return event_id
    # Keep the id stable (so idempotency still holds) but within the limit.
    import hashlib

    digest = hashlib.sha256(event_id.encode("utf-8")).hexdigest()[:16]
    return f"{event_id[: EVENT_ID_MAX_LEN - 17]}-{digest}"


@dataclass
class GovernanceEvent:
    """One governance event as accepted by ``POST /api/public/governance-events``."""

    kind: str
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    occurred_at: str = field(default_factory=utc_now_iso)
    severity: Optional[str] = None
    agent_id: Optional[str] = None
    session_id: Optional[str] = None
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    observation_id: Optional[str] = None
    action: Optional[str] = None
    resource: Optional[str] = None
    decision: Optional[str] = None  # allow | deny | warn | require_approval | log
    policy_name: Optional[str] = None
    policy_version: Optional[Union[str, int]] = None
    rule: Optional[str] = None
    attributes: dict[str, Any] = field(default_factory=dict)
    previous_hash: Optional[str] = None
    hash: Optional[str] = None
    source: Optional[str] = DEFAULT_SOURCE

    def to_wire(self) -> dict[str, Any]:
        """JSON-ready dict: ``None`` fields omitted, ids/limits normalised."""
        wire: dict[str, Any] = {
            "event_id": _clip_event_id(self.event_id),
            "occurred_at": self.occurred_at,
            "kind": self.kind,
        }
        for name in (
            "severity",
            "agent_id",
            "session_id",
            "trace_id",
            "span_id",
            "observation_id",
            "action",
            "resource",
            "decision",
            "policy_name",
            "rule",
            "previous_hash",
            "hash",
            "source",
        ):
            value = getattr(self, name)
            if value is not None:
                wire[name] = str(value)
        if self.policy_version is not None:
            wire["policy_version"] = str(self.policy_version)
        if self.attributes:
            wire["attributes"] = _fit_attributes(dict(self.attributes))
        return wire


def decision_to_event(
    decision: Any,
    *,
    trace_id: Optional[str] = None,
    span_id: Optional[str] = None,
    session_id: Optional[str] = None,
    policy_version: Optional[Union[str, int]] = None,
    observation_id: Optional[str] = None,
    event_id: Optional[str] = None,
    occurred_at: Optional[str] = None,
) -> GovernanceEvent:
    """Convert an :class:`agentops.governance.Decision` into a wire event.

    ``kind="policy.evaluation"``, ``decision=decision.action``,
    ``resource=decision.tool_name``, ``action=context.action.type``,
    ``severity="warning"`` for deny/require_approval else ``"info"``.
    """
    context = getattr(decision, "context", None) or {}
    action_ctx = context.get("action") if isinstance(context, dict) else None
    action_type = action_ctx.get("type") if isinstance(action_ctx, dict) else None

    attributes: dict[str, Any] = {
        "stage": getattr(decision, "stage", None),
        "allowed": bool(getattr(decision, "allowed", False)),
    }
    reason = getattr(decision, "reason", None)
    if reason is not None:
        attributes["reason"] = str(reason)

    action = str(getattr(decision, "action", "") or "") or None
    return GovernanceEvent(
        kind=KIND_POLICY_EVALUATION,
        event_id=event_id or str(uuid.uuid4()),
        occurred_at=occurred_at or utc_now_iso(),
        severity=severity_for(action),
        agent_id=getattr(decision, "agent_id", None),
        session_id=session_id,
        trace_id=trace_id,
        span_id=span_id,
        observation_id=observation_id,
        action=str(action_type) if action_type is not None else None,
        resource=getattr(decision, "tool_name", None),
        decision=action,
        policy_name=getattr(decision, "policy_name", None),
        policy_version=policy_version,
        rule=getattr(decision, "rule", None),
        attributes=attributes,
        source=DEFAULT_SOURCE,
    )


_AUDIT_TOP_LEVEL_MAPPED = frozenset(
    {
        "entry_id",
        "timestamp",
        "event_type",
        "agent_did",
        "action",
        "resource",
        "data",
        "outcome",
        "policy_decision",
        "matched_rule",
        "policy_version",
        "previous_hash",
        "entry_hash",
        "hash",
        "trace_id",
        "session_id",
    }
)


def _as_json_dict(entry: Any) -> dict[str, Any]:
    if isinstance(entry, dict):
        return dict(entry)
    dump = getattr(entry, "model_dump", None)
    if callable(dump):
        try:
            return dict(dump(mode="json"))
        except TypeError:
            return dict(dump())
    raise TypeError(f"audit entry must be a dict or a pydantic model, got {type(entry).__name__}")


def _iso(value: Any) -> str:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.isoformat().replace("+00:00", "Z")
    return str(value)


def audit_entry_to_event(entry: Any) -> GovernanceEvent:
    """Convert an AGT 4.1.0 ``AuditEntry`` (model or ``model_dump(mode="json")``
    dict) into a wire event, preserving the hash chain.

    Mapping: ``entry_id``→``event_id``, ``timestamp``→``occurred_at``,
    ``event_type``→``kind`` (``policy_evaluation``→``policy.evaluation``),
    ``agent_did``→``agent_id``, ``outcome``/``policy_decision``→``decision``,
    ``data.rule``/``matched_rule``→``rule``, ``previous_hash``/``entry_hash``
    →``previous_hash``/``hash``; ``action``, ``resource``, ``trace_id``,
    ``session_id``, ``policy_version`` pass through. Everything else
    (``data.reason``, ``data.evaluation_ms``, ``sandbox_id``, …) lands in
    ``attributes`` (``None`` values skipped, ``event_type`` always kept).
    """
    raw = _as_json_dict(entry)
    data = raw.get("data") if isinstance(raw.get("data"), dict) else {}

    event_type = str(raw.get("event_type") or "audit")
    kind = event_type.replace("_", ".")
    decision = raw.get("policy_decision") or raw.get("outcome")
    decision_str = str(decision) if decision is not None else None
    rule = data.get("rule") or raw.get("matched_rule")

    attributes: dict[str, Any] = {"event_type": event_type}
    for key, value in raw.items():
        if key in _AUDIT_TOP_LEVEL_MAPPED or value is None:
            continue
        attributes[key] = value
    for key, value in data.items():
        if key == "rule" or value is None:
            continue
        attributes[f"data.{key}" if key in attributes else key] = value
    outcome = raw.get("outcome")
    if outcome is not None and raw.get("policy_decision") not in (None, outcome):
        attributes["outcome"] = outcome

    entry_id = raw.get("entry_id") or str(uuid.uuid4())
    timestamp = raw.get("timestamp")
    return GovernanceEvent(
        kind=kind,
        event_id=str(entry_id),
        occurred_at=_iso(timestamp) if timestamp is not None else utc_now_iso(),
        severity=severity_for(decision_str if decision_str in DECISIONS else None),
        agent_id=raw.get("agent_did"),
        session_id=raw.get("session_id"),
        trace_id=raw.get("trace_id"),
        action=raw.get("action"),
        resource=raw.get("resource"),
        decision=decision_str if decision_str in DECISIONS else None,
        policy_version=raw.get("policy_version"),
        rule=str(rule) if rule is not None else None,
        attributes=attributes,
        previous_hash=raw.get("previous_hash"),
        hash=raw.get("entry_hash") or raw.get("hash"),
        source=AUDIT_LOG_SOURCE,
    )


# --------------------------------------------------------------------------- #
# Shipper
# --------------------------------------------------------------------------- #
def _resolve_credentials(client_or_credentials: Any) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Accept an ``AgentOps`` / ``AsyncAgentOps`` client, a ``(host,
    public_key, secret_key)`` tuple, a dict with those keys, or ``None``
    (``AGENTOPS_*`` environment)."""
    obj = client_or_credentials
    host = public = secret = None
    if obj is None:
        pass
    elif isinstance(obj, (tuple, list)):
        if len(obj) != 3:
            raise ValueError("credentials tuple must be (host, public_key, secret_key)")
        host, public, secret = obj
    elif isinstance(obj, dict):
        host = obj.get("host")
        public = obj.get("public_key")
        secret = obj.get("secret_key")
    else:
        host = getattr(obj, "_host", None) or getattr(obj, "host", None)
        public = getattr(obj, "_public_key", None) or getattr(obj, "public_key", None)
        secret = getattr(obj, "_secret_key", None) or getattr(obj, "secret_key", None)
    host = host or get_host()
    public = public or get_public_key()
    secret = secret or get_secret_key()
    return (str(host) if host else None, str(public) if public else None, str(secret) if secret else None)


def governance_events_url(host: str) -> str:
    return f"{host.rstrip('/')}{GOVERNANCE_EVENTS_PATH}"


class _Retryable(Exception):
    pass


class GovernanceEventShipper:
    """Batching, best-effort background exporter for governance events.

    ``client_or_credentials`` is an ``AgentOps``/``AsyncAgentOps`` instance, a
    ``(host, public_key, secret_key)`` tuple, a dict, or ``None`` to read
    ``AGENTOPS_HOST`` / ``AGENTOPS_PUBLIC_KEY`` / ``AGENTOPS_SECRET_KEY``.
    Missing credentials disable the shipper (events are dropped, one
    warning logged) — the same warn-and-continue behaviour as ``agentops.init()``.

    * ``emit(event)`` enqueues and returns immediately (``False`` when dropped).
    * A daemon thread posts batches of ``batch_size`` (≤ 500) or whatever has
      accumulated after ``flush_interval_s``.
    * Transport errors, 429 and 5xx are retried ``max_retries`` times with
      exponential backoff (``backoff_s``); 4xx responses are logged and dropped.
    * ``flush(timeout)`` blocks until the queue is drained; ``shutdown()``
      drains and stops the thread and is also registered with ``atexit``.
    """

    def __init__(
        self,
        client_or_credentials: Any = None,
        *,
        batch_size: int = 100,
        flush_interval_s: float = 2.0,
        max_queue: int = 5000,
        max_retries: int = 3,
        backoff_s: float = 0.5,
        timeout_s: float = _http.DEFAULT_TIMEOUT_S,
        session: Any = None,
        source: str = DEFAULT_SOURCE,
    ) -> None:
        host, public_key, secret_key = _resolve_credentials(client_or_credentials)
        self.batch_size = max(1, min(int(batch_size), MAX_BATCH_SIZE))
        self.flush_interval_s = max(0.01, float(flush_interval_s))
        self.max_queue = max(1, int(max_queue))
        self.max_retries = max(0, int(max_retries))
        self.backoff_s = max(0.0, float(backoff_s))
        self.timeout_s = float(timeout_s)
        self.source = source
        self._session = session

        self.enabled = bool(host and public_key and secret_key)
        self.url = governance_events_url(host) if host else None
        self._headers: dict[str, str] = {}
        if self.enabled:
            self._headers = {
                "Authorization": _http.basic_auth_header(str(public_key), str(secret_key)),
                "Content-Type": "application/json",
            }
        else:
            missing = [
                name
                for name, value in (
                    ("AGENTOPS_HOST", host),
                    ("AGENTOPS_PUBLIC_KEY", public_key),
                    ("AGENTOPS_SECRET_KEY", secret_key),
                )
                if not value
            ]
            _log.warning(
                "GovernanceEventShipper disabled — missing %s. Governance events will not reach AOS Control.",
                ", ".join(missing),
            )

        self._queue: collections.deque[GovernanceEvent] = collections.deque()
        self._cv = threading.Condition()
        self._in_flight = 0
        self._flush_requested = False
        self._stopped = False
        self._thread: Optional[threading.Thread] = None
        self._atexit_registered = False
        self._last_drop_warning = 0.0
        self.stats: dict[str, int] = {
            "emitted": 0,
            "sent": 0,
            "accepted": 0,
            "duplicates": 0,
            "rejected": 0,
            "dropped": 0,
            "failed_batches": 0,
            "requests": 0,
        }

    # ----------------------------------------------------------------- queue
    def emit(self, event: Union[GovernanceEvent, dict[str, Any]]) -> bool:
        """Queue ``event`` for delivery. Never raises; returns ``False`` when
        the event was dropped (shipper disabled/stopped, queue full)."""
        try:
            if isinstance(event, dict):
                event = GovernanceEvent(**event)
            if event.source is None:
                event.source = self.source
            with self._cv:
                self.stats["emitted"] += 1
                if not self.enabled or self._stopped:
                    self.stats["dropped"] += 1
                    return False
                if len(self._queue) >= self.max_queue:
                    self.stats["dropped"] += 1
                    self._warn_drop_locked("queue full (%d events)" % self.max_queue)
                    return False
                self._queue.append(event)
                self._ensure_thread_locked()
                if len(self._queue) >= self.batch_size:
                    self._cv.notify_all()
            return True
        except Exception as e:  # pragma: no cover - must never reach the agent
            _log.warning("GovernanceEventShipper.emit failed: %s", e)
            return False

    def emit_decision(self, decision: Any, **kwargs: Any) -> bool:
        """``emit(decision_to_event(decision, **kwargs))`` (ids default to the current span)."""
        if "trace_id" not in kwargs and "span_id" not in kwargs:
            kwargs["trace_id"], kwargs["span_id"] = current_span_ids()
        return self.emit(decision_to_event(decision, **kwargs))

    @property
    def pending(self) -> int:
        with self._cv:
            return len(self._queue) + self._in_flight

    def flush(self, timeout: Optional[float] = 10.0) -> bool:
        """Block until every queued event has been posted (or dropped).
        Returns ``False`` if ``timeout`` elapsed first."""
        if not self.enabled:
            return True
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._cv:
            if self._queue and (self._thread is None or not self._thread.is_alive()) and not self._stopped:
                self._ensure_thread_locked()
            self._flush_requested = True
            self._cv.notify_all()
            while self._queue or self._in_flight:
                if self._thread is None or not self._thread.is_alive():
                    return not (self._queue or self._in_flight)
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return False
                self._cv.wait(remaining if remaining is not None else 1.0)
            return True

    def shutdown(self, timeout: Optional[float] = 5.0) -> None:
        """Drain the queue (bounded by ``timeout``) and stop the worker.
        Idempotent; also runs at interpreter exit."""
        with self._cv:
            self._stopped = True
            self._cv.notify_all()
            thread = self._thread
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout)
        if self._atexit_registered:
            try:
                atexit.unregister(self.shutdown)
            except Exception:  # pragma: no cover
                pass
            self._atexit_registered = False

    def __enter__(self) -> "GovernanceEventShipper":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.shutdown()

    # ---------------------------------------------------------------- worker
    def _ensure_thread_locked(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(
            target=self._run, name="agentops-governance-events", daemon=True
        )
        self._thread.start()
        if not self._atexit_registered:
            atexit.register(self.shutdown)
            self._atexit_registered = True

    def _warn_drop_locked(self, why: str) -> None:
        now = time.monotonic()
        if now - self._last_drop_warning > 30.0:
            self._last_drop_warning = now
            _log.warning("GovernanceEventShipper dropping events: %s", why)

    def _run(self) -> None:
        while True:
            with self._cv:
                deadline = time.monotonic() + self.flush_interval_s
                while (
                    not self._stopped
                    and not self._flush_requested
                    and len(self._queue) < self.batch_size
                ):
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    self._cv.wait(remaining)
                if not self._queue:
                    self._flush_requested = False
                    self._cv.notify_all()
                    if self._stopped:
                        return
                    continue
                batch = [self._queue.popleft() for _ in range(min(self.batch_size, len(self._queue)))]
                self._in_flight += len(batch)
            try:
                self._send_with_retry(batch)
            except Exception as e:  # pragma: no cover - belt and braces
                _log.warning("GovernanceEventShipper: unexpected failure, %d events dropped: %s", len(batch), e)
                with self._cv:
                    self.stats["dropped"] += len(batch)
                    self.stats["failed_batches"] += 1
            finally:
                with self._cv:
                    self._in_flight -= len(batch)
                    if not self._queue:
                        self._flush_requested = False
                    self._cv.notify_all()

    # ------------------------------------------------------------- transport
    def _session_obj(self) -> Any:
        return self._session if self._session is not None else _http.get_session()

    def _post(self, payload: dict[str, Any]) -> Any:
        with self._cv:
            self.stats["requests"] += 1
        return self._session_obj().post(self.url, json=payload, headers=self._headers, timeout=self.timeout_s)

    def _send_with_retry(self, batch: list[GovernanceEvent]) -> bool:
        payload = {"events": [e.to_wire() for e in batch]}
        attempt = 0
        while True:
            try:
                self._send_once(payload, len(batch))
                return True
            except _Retryable as e:
                if attempt >= self.max_retries:
                    with self._cv:
                        self.stats["dropped"] += len(batch)
                        self.stats["failed_batches"] += 1
                    _log.warning(
                        "GovernanceEventShipper: giving up on batch of %d events after %d attempts: %s",
                        len(batch),
                        attempt + 1,
                        e,
                    )
                    return False
                delay = self.backoff_s * (2**attempt)
                attempt += 1
                if delay > 0:
                    time.sleep(delay)
            except _Fatal as e:
                with self._cv:
                    self.stats["dropped"] += len(batch)
                    self.stats["failed_batches"] += 1
                _log.warning("GovernanceEventShipper: batch of %d events rejected (%s); not retrying", len(batch), e)
                return False

    def _send_once(self, payload: dict[str, Any], count: int) -> None:
        try:
            resp = self._post(payload)
        except Exception as e:  # connection errors, timeouts, transport bugs
            raise _Retryable(f"{type(e).__name__}: {e}") from e

        status = int(getattr(resp, "status_code", 0) or 0)
        if 200 <= status < 300:
            body: Any = None
            try:
                body = resp.json()
            except Exception:
                body = None
            accepted = duplicates = 0
            rejected: list[Any] = []
            if isinstance(body, dict):
                accepted = int(body.get("accepted") or 0)
                duplicates = int(body.get("duplicates") or 0)
                rejected = body.get("rejected") or []
            with self._cv:
                self.stats["sent"] += count
                self.stats["accepted"] += accepted
                self.stats["duplicates"] += duplicates
                self.stats["rejected"] += len(rejected)
            if rejected:
                _log.warning(
                    "GovernanceEventShipper: server rejected %d of %d events: %s",
                    len(rejected),
                    count,
                    json.dumps(rejected[:5], default=str)[:500],
                )
            return

        text = ""
        try:
            text = (getattr(resp, "text", "") or "")[:300]
        except Exception:  # pragma: no cover
            text = ""
        if status == 429 or status >= 500 or status == 0:
            raise _Retryable(f"HTTP {status} {text}".strip())
        raise _Fatal(f"HTTP {status} {text}".strip())


class _Fatal(Exception):
    pass


# --------------------------------------------------------------------------- #
# AGT AuditLog integration
# --------------------------------------------------------------------------- #
_AUDIT_CURSORS: "weakref.WeakKeyDictionary[Any, str]" = weakref.WeakKeyDictionary()
_AUDIT_CURSORS_BY_ID: dict[int, str] = {}


def _audit_entries(audit_log: Any) -> list[Any]:
    """All entries of an AGT ``AuditLog`` in chain order.

    AGT 4.1.0 keeps them on ``audit_log._chain._entries`` (``query()`` caps at
    ``limit`` and ``export()`` at 10 000); other implementations may expose
    ``entries`` — both are accepted.
    """
    chain = getattr(audit_log, "_chain", None)
    entries = getattr(chain, "_entries", None) if chain is not None else None
    if entries is None:
        entries = getattr(audit_log, "entries", None)
    if entries is None and hasattr(audit_log, "query"):
        entries = audit_log.query(limit=10**9)
    if entries is None:
        raise TypeError(f"{type(audit_log).__name__} does not look like an AGT AuditLog")
    return list(entries)


def _entry_id(entry: Any) -> Optional[str]:
    if isinstance(entry, dict):
        value = entry.get("entry_id")
    else:
        value = getattr(entry, "entry_id", None)
    return str(value) if value is not None else None


def _get_cursor(audit_log: Any) -> Optional[str]:
    try:
        return _AUDIT_CURSORS.get(audit_log)
    except TypeError:
        return _AUDIT_CURSORS_BY_ID.get(id(audit_log))


def _set_cursor(audit_log: Any, entry_id: str) -> None:
    try:
        _AUDIT_CURSORS[audit_log] = entry_id
    except TypeError:
        _AUDIT_CURSORS_BY_ID[id(audit_log)] = entry_id


def ship_audit_log(
    audit_log: Any,
    shipper: GovernanceEventShipper,
    *,
    after_entry_id: Optional[str] = None,
) -> int:
    """Emit every entry of an AGT ``AuditLog`` not shipped before.

    The last shipped ``entry_id`` is remembered per ``audit_log`` object, so
    calling this repeatedly (e.g. after each governed call, or on a timer)
    only sends new entries. ``after_entry_id`` overrides the remembered
    cursor. Returns the number of events emitted. Idempotency on the server
    (``event_id`` = ``entry_id``) makes an occasional re-send harmless.
    """
    cursor = after_entry_id if after_entry_id is not None else _get_cursor(audit_log)
    entries = _audit_entries(audit_log)

    start = 0
    if cursor is not None:
        for index, entry in enumerate(entries):
            if _entry_id(entry) == cursor:
                start = index + 1
                break

    shipped = 0
    last_id: Optional[str] = None
    for entry in entries[start:]:
        try:
            event = audit_entry_to_event(entry)
        except Exception as e:
            _log.warning("ship_audit_log: skipping unconvertible audit entry: %s", e)
            continue
        if shipper.emit(event):
            shipped += 1
        last_id = _entry_id(entry) or last_id
    if last_id is not None:
        _set_cursor(audit_log, last_id)
    return shipped


class AgentOpsAuditSink:
    """An AGT ``AuditSink`` that forwards each ``AuditEntry`` to AOS Control
    as it is written — ``AuditLog(sink=AgentOpsAuditSink(shipper))``.

    Real-time counterpart of :func:`ship_audit_log`; implements the
    ``write`` / ``write_batch`` / ``verify_integrity`` / ``close`` protocol of
    ``agentmesh.governance.audit_backends.AuditSink`` (AGT 4.1.0).
    """

    def __init__(self, shipper: GovernanceEventShipper):
        self.shipper = shipper

    def write(self, entry: Any) -> None:
        try:
            self.shipper.emit(audit_entry_to_event(entry))
        except Exception as e:  # never break the audit log
            _log.warning("AgentOpsAuditSink: could not ship audit entry: %s", e)

    def write_batch(self, entries: Iterable[Any]) -> None:
        for entry in entries:
            self.write(entry)

    def verify_integrity(self) -> tuple[bool, Optional[str]]:
        # The server verifies the hash chain (nightly job); locally there is
        # nothing to check beyond what AuditLog itself does.
        return True, None

    def close(self) -> None:
        self.shipper.flush()


__all__ = [
    "ATTRIBUTES_MAX_BYTES",
    "AUDIT_LOG_SOURCE",
    "DEFAULT_SOURCE",
    "EVENT_ID_MAX_LEN",
    "GOVERNANCE_EVENTS_PATH",
    "KIND_POLICY_EVALUATION",
    "MAX_BATCH_SIZE",
    "AgentOpsAuditSink",
    "GovernanceEvent",
    "GovernanceEventShipper",
    "audit_entry_to_event",
    "current_span_ids",
    "decision_to_event",
    "governance_events_url",
    "severity_for",
    "ship_audit_log",
    "utc_now_iso",
]
