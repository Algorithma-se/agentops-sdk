"""Tool-call policy enforcement with Microsoft's Agent Governance Toolkit
(AGT), with every decision exported to AOS Control as an OpenTelemetry span.

Why this module exists (verified against ``agent-governance-toolkit-core``
4.1.0, see ``packages/shared/src/server/otel/__fixtures__/agt/README.md``):

* ``agentmesh.governance.enable_otel()`` installs a global ``TracerProvider``
  but attaches **no exporter**; its ``endpoint`` argument is ignored.
* ``agentmesh.governance.govern()`` evaluates policy but emits **no spans**.
  The only way to get an ``agt.policy.evaluate`` span is to wrap the
  evaluation in AGT's public ``trace_policy_evaluation`` context manager.
* That span carries no tool name and no arguments.

So this module (a) attaches an OTLP/HTTP exporter pointed at AOS Control to
whatever provider AGT installed, and (b) evaluates the policy *once* through
AGT's own ``PolicyEngine`` inside ``trace_policy_evaluation``, adding
``agt.tool.name`` / ``agt.tool.args`` / ``session.id`` so the ingestion side
can render a ``GUARDRAIL`` observation and raise a Guardrail Alert.

Install::

    pip install "agentops-sdk[governance]"   # AGT core 4.1.0 + OTel SDK + OTLP exporter

Usage::

    from agentops.governance import enable_agt_export, governed, PolicyDenied

    enable_agt_export(service_name="support-agent")   # AGENTOPS_HOST/keys from env

    @governed(policy="policy.yaml", agent_id="did:mesh:support-agent")
    def execute_tool(action: str, tool: str, path: str) -> str:
        ...

    try:
        execute_tool(action="delete", tool="delete_file", path="/data/x.csv")
    except PolicyDenied as e:
        print(e.rule, e.policy_name)

For frameworks where you cannot wrap the callable (e.g. a tool-dispatch hook)::

    from agentops.governance import pre_tool_call

    decision = pre_tool_call(policy, agent_id, tool_name="delete_file", args=call.arguments)
    if not decision.allowed:
        return f"blocked by {decision.rule}"

``agentmesh`` is imported lazily inside functions, so importing ``agentops``
(or this module) never requires AGT to be installed.
"""

from __future__ import annotations

import functools
import inspect
import json
import logging
import os
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Union

from ._env import get_host, get_public_key, get_secret_key
from ._http import basic_auth_header
from .governance_events import (
    GovernanceEventShipper,
    current_span_ids,
    decision_to_event,
)
from .guardrails import _redact_pii

_log = logging.getLogger("agentops.governance")

# Span/attribute names. ``SPAN_NAME`` and the ``agt.policy.*`` keys are AGT's
# (``agentmesh.governance.otel_observability``); the ``agt.tool.*`` keys reuse
# AGT's ``ATTR_TOOL_NAME`` spelling (which AGT itself only uses on metrics).
SPAN_NAME = "agt.policy.evaluate"
ATTR_TOOL_NAME = "agt.tool.name"
ATTR_TOOL_ARGS = "agt.tool.args"
ATTR_SESSION_ID = "session.id"

ARGS_MAX_BYTES = 4096
_ARGS_TRUNCATED_MARKER = "…[truncated]"

OTEL_TRACES_PATH = "/api/public/otel/v1/traces"
DEFAULT_SERVICE_NAME = "agentops-governed-agent"
SESSION_ID_ENV = "AGENTOPS_SESSION_ID"

_INSTALL_HINT = 'pip install "agentops-sdk[governance]"'

PolicySpec = Union[str, "os.PathLike[str]", Any]  # path, YAML text, or agentmesh Policy


# --------------------------------------------------------------------------- #
# Governance event shipping (POST /api/public/governance-events)
# --------------------------------------------------------------------------- #
_DEFAULT_SHIPPER: Optional[GovernanceEventShipper] = None


def set_default_shipper(shipper: Optional[GovernanceEventShipper]) -> None:
    """Install the :class:`GovernanceEventShipper` used by ``governed`` /
    ``pre_tool_call`` / ``evaluate`` when they are not given one explicitly.
    ``enable_agt_export(ship_events=True)`` sets one up from the same
    credentials; pass ``None`` to stop shipping."""
    global _DEFAULT_SHIPPER
    _DEFAULT_SHIPPER = shipper


def get_default_shipper() -> Optional[GovernanceEventShipper]:
    return _DEFAULT_SHIPPER


def _ensure_default_shipper(host: str, public_key: str, secret_key: str) -> GovernanceEventShipper:
    global _DEFAULT_SHIPPER
    if _DEFAULT_SHIPPER is None:
        _DEFAULT_SHIPPER = GovernanceEventShipper((host, public_key, secret_key))
    return _DEFAULT_SHIPPER


def _ship_decision(
    decision: "Decision",
    shipper: Optional[GovernanceEventShipper],
    *,
    trace_id: Optional[str],
    span_id: Optional[str],
    session_id: Optional[str],
    policy_version: Union[int, str, None],
) -> None:
    """Best-effort: turn the decision into a wire event and queue it. Never raises."""
    target = shipper if shipper is not None else _DEFAULT_SHIPPER
    if target is None:
        return
    try:
        target.emit(
            decision_to_event(
                decision,
                trace_id=trace_id,
                span_id=span_id,
                session_id=session_id,
                policy_version=policy_version,
            )
        )
    except Exception as e:  # pragma: no cover - shipper.emit already swallows
        _log.debug("governance event not shipped: %s", e)


# --------------------------------------------------------------------------- #
# Lazy AGT access
# --------------------------------------------------------------------------- #
def _agt() -> Any:
    """Import and return ``agentmesh.governance`` lazily.

    AGT 4.1.0 emits ``DeprecationWarning: agentmesh-platform is deprecated``
    on import even when only ``agent-governance-toolkit-core`` is installed
    (there is no other import path in 4.1.0); that one warning is silenced.
    """
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="agentmesh-platform is deprecated",
                category=DeprecationWarning,
            )
            import agentmesh.governance as governance  # type: ignore[import-not-found]
    except ImportError as e:
        raise ImportError(
            "agentops.governance requires the Agent Governance Toolkit "
            f"(agentmesh.governance). Install the extra: {_INSTALL_HINT}"
        ) from e
    return governance


# --------------------------------------------------------------------------- #
# Public result types
# --------------------------------------------------------------------------- #
@dataclass
class Decision:
    """Outcome of one policy evaluation for one tool call.

    ``allowed`` follows AGT's ``PolicyDecision.allowed`` exactly. Note that
    AGT 4.1.0 sets ``allowed=False`` for every action other than ``allow``
    — including ``warn``, ``log`` and ``require_approval`` — and its own
    ``govern()`` raises on them, so this wrapper does too.
    """

    allowed: bool
    action: str  # allow | deny | warn | require_approval | log
    agent_id: str
    tool_name: str
    stage: str
    rule: Optional[str] = None
    policy_name: Optional[str] = None
    reason: Optional[str] = None
    context: dict[str, Any] = field(default_factory=dict)
    raw: Any = None  # the agentmesh PolicyDecision this was built from

    @property
    def denied(self) -> bool:
        return not self.allowed


class PolicyDenied(Exception):
    """Raised when AGT policy does not allow a governed tool call.

    Wraps AGT's ``GovernanceDenied`` (available as ``__cause__`` /
    ``.governance_denied``) and exposes the decision fields directly.
    """

    def __init__(self, decision: Decision, governance_denied: Optional[BaseException] = None):
        self.decision = decision
        self.governance_denied = governance_denied
        self.rule = decision.rule
        self.policy_name = decision.policy_name
        self.agent_id = decision.agent_id
        self.tool_name = decision.tool_name
        self.action = decision.action
        self.reason = decision.reason
        super().__init__(
            f"Tool call '{decision.tool_name}' by agent '{decision.agent_id}' "
            f"not allowed (action={decision.action}, rule="
            f"{decision.rule or 'none'}, policy={decision.policy_name or 'none'}): "
            f"{decision.reason or 'no reason given'}"
        )


# --------------------------------------------------------------------------- #
# Policy engine construction (mirrors GovernedCallable.__init__ in AGT 4.1.0)
# --------------------------------------------------------------------------- #
_ENGINES: dict[tuple[Any, ...], tuple[Any, Any]] = {}


def _engine_key(policy: PolicySpec, conflict_strategy: str) -> tuple[Any, ...]:
    if isinstance(policy, os.PathLike):
        return ("path", os.fspath(policy), conflict_strategy)
    if isinstance(policy, str):
        return ("str", policy, conflict_strategy)
    return ("obj", id(policy), conflict_strategy)


def _load_engine(policy: PolicySpec, conflict_strategy: str) -> Any:
    agt = _agt()
    engine = agt.PolicyEngine(conflict_strategy=conflict_strategy)

    if isinstance(policy, os.PathLike):
        loaded = engine.load_yaml_file(os.fspath(policy))
    elif isinstance(policy, str):
        # Same disambiguation AGT's govern() uses: an existing file path wins,
        # anything else is treated as inline YAML.
        if os.path.isfile(policy):
            loaded = engine.load_yaml_file(policy)
        else:
            loaded = engine.load_yaml(policy)
    elif isinstance(policy, agt.Policy):
        loaded = policy
        engine.load_policy(loaded)
    else:
        raise TypeError(
            "policy must be a file path, YAML string, or agentmesh.governance.Policy, "
            f"got {type(policy).__name__}"
        )

    # govern() defaults an untargeted policy to every agent; do the same so a
    # policy file that works with govern() works here.
    if not getattr(loaded, "agent", None) and not getattr(loaded, "agents", None):
        loaded.agents = ["*"]
    return engine


def get_engine(policy: PolicySpec, conflict_strategy: str = "deny_overrides") -> Any:
    """Return the (cached) AGT ``PolicyEngine`` for ``policy``.

    Engines are cached per (policy, conflict_strategy) so ``pre_tool_call``
    does not re-parse a YAML file on every tool call. ``Policy`` objects are
    cached by identity.
    """
    key = _engine_key(policy, conflict_strategy)
    hit = _ENGINES.get(key)
    if hit is not None:
        return hit[0]
    engine = _load_engine(policy, conflict_strategy)
    _ENGINES[key] = (engine, policy)  # keep the Policy alive so id() stays unique
    return engine


def clear_engine_cache() -> None:
    """Drop cached policy engines (e.g. after rewriting a policy file)."""
    _ENGINES.clear()


# --------------------------------------------------------------------------- #
# Evaluation context (port of GovernedCallable._build_context, AGT 4.1.0)
# --------------------------------------------------------------------------- #
def build_context(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    tool_name: Optional[str] = None,
) -> dict[str, Any]:
    """Build the policy context exactly as AGT's ``govern()`` would.

    * ``action=`` kwarg → ``{"action": {"type": str(value)}}`` (a dict is
      passed through); otherwise the first positional arg becomes
      ``action.type``.
    * every other kwarg → ``{"<kwarg>": {"value": value}}`` (dicts pass
      through), so a tool-name rule is written ``tool.value == 'delete_file'``.

    On top of AGT's shape, when ``tool_name`` is given and the call supplied
    no ``action`` / ``tool``, ``action.type`` and ``tool.value`` default to
    the tool name so policies written against ``tool.value`` keep working
    for frameworks that only hand us the tool's argument dict.
    """
    context: dict[str, Any] = {}

    if "action" in kwargs:
        action_val = kwargs["action"]
        context["action"] = action_val if isinstance(action_val, dict) else {"type": str(action_val)}
    elif args:
        context["action"] = {"type": str(args[0])}

    for key, val in kwargs.items():
        if key != "action":
            context[key] = val if isinstance(val, dict) else {"value": val}

    if tool_name:
        context.setdefault("action", {"type": tool_name})
        context.setdefault("tool", {"value": tool_name})

    return context


# --------------------------------------------------------------------------- #
# Span annotation helpers
# --------------------------------------------------------------------------- #
def _json_default(value: Any) -> Any:
    if isinstance(value, (set, frozenset, tuple)):
        return list(value)
    if isinstance(value, bytes):
        return f"<{len(value)} bytes>"
    if hasattr(value, "model_dump"):
        try:
            return value.model_dump(mode="json")
        except Exception:
            pass
    return repr(value)


def serialize_args(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    redact: bool = True,
    max_bytes: int = ARGS_MAX_BYTES,
) -> str:
    """JSON-encode a tool call's arguments for the ``agt.tool.args`` attribute.

    Keyword arguments are the payload; positional arguments (rare for tool
    calls, but AGT reads ``args[0]`` as the action) are kept under ``_args``.
    PII is redacted with the same patterns ``agentops.guardrails`` uses, and
    the result is truncated to ``max_bytes`` of UTF-8.
    """
    payload: dict[str, Any] = dict(kwargs)
    if args:
        payload = {"_args": list(args), **payload}
    try:
        text = json.dumps(payload, default=_json_default, ensure_ascii=False, sort_keys=True)
    except Exception:
        text = repr(payload)
    if redact:
        text, _ = _redact_pii(text)

    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    marker = _ARGS_TRUNCATED_MARKER.encode("utf-8")
    head = encoded[: max(0, max_bytes - len(marker))].decode("utf-8", errors="ignore")
    return head + _ARGS_TRUNCATED_MARKER


def _resolve_session_id(explicit: Union[str, Callable[[], Optional[str]], None]) -> Optional[str]:
    value: Any = explicit() if callable(explicit) else explicit
    if value:
        return str(value)
    env_value = os.environ.get(SESSION_ID_ENV)
    return env_value or None


def _annotate_current_span(
    *,
    tool_name: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    session_id: Union[str, Callable[[], Optional[str]], None],
    record_args: bool,
    redact_args: bool,
) -> None:
    """Add ``agt.tool.*`` and ``session.id`` to the span opened by
    ``trace_policy_evaluation`` — and only to that span. If AGT's tracer is
    not enabled, the current span belongs to the application and is left alone.
    """
    try:
        from opentelemetry import trace  # type: ignore[import-not-found]
    except ImportError:
        return

    span = trace.get_current_span()
    if not span.is_recording() or getattr(span, "name", None) != SPAN_NAME:
        return

    span.set_attribute(ATTR_TOOL_NAME, tool_name)
    if record_args:
        span.set_attribute(ATTR_TOOL_ARGS, serialize_args(args, kwargs, redact=redact_args))
    sid = _resolve_session_id(session_id)
    if sid:
        span.set_attribute(ATTR_SESSION_ID, sid)


# --------------------------------------------------------------------------- #
# Core: evaluate once, inside AGT's span
# --------------------------------------------------------------------------- #
def evaluate(
    policy: PolicySpec,
    agent_id: str,
    tool_name: str,
    args: tuple[Any, ...] = (),
    kwargs: Optional[dict[str, Any]] = None,
    *,
    stage: str = "pre_tool",
    session_id: Union[str, Callable[[], Optional[str]], None] = None,
    conflict_strategy: str = "deny_overrides",
    record_args: bool = True,
    redact_args: bool = True,
    shipper: Optional[GovernanceEventShipper] = None,
    policy_version: Union[int, str, None] = None,
    ship: bool = True,
) -> Decision:
    """Evaluate ``policy`` for one tool call and emit the ``agt.policy.evaluate`` span.

    This is what ``governed`` and ``pre_tool_call`` share: build the context
    the way AGT's ``govern()`` does, run ``PolicyEngine.evaluate`` **once**
    inside ``trace_policy_evaluation`` (populating ``action`` / ``rule`` /
    ``policy_name`` / ``allowed`` so AGT sets the ``agt.policy.*``
    attributes), and add ``agt.tool.name``, ``agt.tool.args`` and
    ``session.id`` to that span. Never calls AGT's governed callable, so the
    policy is not evaluated twice.

    When a shipper is configured (``shipper=`` or :func:`set_default_shipper`
    / ``enable_agt_export(ship_events=True)``) and ``ship`` is true, the
    decision is also queued as a ``policy.evaluation`` governance event
    carrying the trace/span ids of the span it was evaluated in (hex, OTLP
    spelling) and ``policy_version`` (the registry version from
    ``client.get_policy``), so it joins the trace in AOS Control.
    """
    agt = _agt()
    kwargs = dict(kwargs or {})
    engine = get_engine(policy, conflict_strategy)
    context = build_context(args, kwargs, tool_name=tool_name)
    resolved_session_id = _resolve_session_id(session_id)

    raw: Any = None
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    with agt.trace_policy_evaluation(agent_id=agent_id, stage=stage, context=context) as result:
        _annotate_current_span(
            tool_name=tool_name,
            args=args,
            kwargs=kwargs,
            session_id=resolved_session_id,
            record_args=record_args,
            redact_args=redact_args,
        )
        trace_id, span_id = current_span_ids()
        raw = engine.evaluate(agent_id, context, stage=stage)
        result["action"] = raw.action
        result["rule"] = raw.matched_rule
        result["policy_name"] = raw.policy_name
        result["allowed"] = raw.allowed

    decision = Decision(
        allowed=bool(raw.allowed),
        action=str(raw.action),
        agent_id=agent_id,
        tool_name=tool_name,
        stage=stage,
        rule=raw.matched_rule,
        policy_name=raw.policy_name,
        reason=getattr(raw, "reason", None),
        context=context,
        raw=raw,
    )
    if ship:
        _ship_decision(
            decision,
            shipper,
            trace_id=trace_id,
            span_id=span_id,
            session_id=resolved_session_id,
            policy_version=policy_version,
        )
    return decision


def _raise_denied(decision: Decision) -> None:
    governance_denied: Optional[BaseException] = None
    try:
        governance_denied = _agt().GovernanceDenied(decision.raw)
    except Exception:  # pragma: no cover - AGT exception construction is best-effort
        governance_denied = None
    raise PolicyDenied(decision, governance_denied) from governance_denied


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def pre_tool_call(
    policy: PolicySpec,
    agent_id: str,
    tool_name: str,
    args: Optional[dict[str, Any]],
    stage: str = "pre_tool",
    *,
    session_id: Union[str, Callable[[], Optional[str]], None] = None,
    conflict_strategy: str = "deny_overrides",
    raise_on_deny: bool = False,
    record_args: bool = True,
    redact_args: bool = True,
    shipper: Optional[GovernanceEventShipper] = None,
    policy_version: Union[int, str, None] = None,
) -> Decision:
    """Evaluate policy for a tool call you cannot wrap (dispatch hooks, MCP
    servers, framework callbacks). ``args`` is the tool's argument dict; it
    is mapped to AGT's context shape (``{"<arg>": {"value": ...}}``) with
    ``action.type`` / ``tool.value`` defaulting to ``tool_name`` when absent.

    Returns a :class:`Decision`; with ``raise_on_deny=True`` raises
    :class:`PolicyDenied` instead when the call is not allowed. The decision
    is shipped to AOS Control through ``shipper`` (or the default shipper,
    see :func:`set_default_shipper`) tagged with ``policy_version``.
    """
    decision = evaluate(
        policy,
        agent_id,
        tool_name,
        (),
        args or {},
        stage=stage,
        session_id=session_id,
        conflict_strategy=conflict_strategy,
        record_args=record_args,
        redact_args=redact_args,
        shipper=shipper,
        policy_version=policy_version,
    )
    if raise_on_deny and not decision.allowed:
        _raise_denied(decision)
    return decision


def governed(
    fn: Optional[Callable[..., Any]] = None,
    *,
    policy: PolicySpec,
    agent_id: str,
    stage: str = "pre_tool",
    tool_name: Optional[str] = None,
    raise_on_deny: bool = True,
    session_id: Union[str, Callable[[], Optional[str]], None] = None,
    conflict_strategy: str = "deny_overrides",
    record_args: bool = True,
    redact_args: bool = True,
    shipper: Optional[GovernanceEventShipper] = None,
    policy_version: Union[int, str, None] = None,
) -> Any:
    """Decorate (or wrap) a tool function with AGT policy enforcement.

    Works as ``@governed(policy=..., agent_id=...)`` and as
    ``governed(fn, policy=..., agent_id=...)``; sync and ``async`` functions
    are both supported. Each call evaluates the policy once against the
    same context AGT's ``govern()`` would build from the call's arguments,
    emits the ``agt.policy.evaluate`` span (with ``agt.tool.name`` =
    ``tool_name`` or ``fn.__name__``), then either invokes ``fn`` or, when
    not allowed, raises :class:`PolicyDenied` — or returns the
    :class:`Decision` when ``raise_on_deny=False``.

    The returned wrapper exposes ``.evaluate(*args, **kwargs) -> Decision``
    for a dry run that emits the span without calling ``fn``.
    """

    def decorate(func: Callable[..., Any]) -> Callable[..., Any]:
        name = tool_name or getattr(func, "__name__", None) or repr(func)

        def check(call_args: tuple[Any, ...], call_kwargs: dict[str, Any]) -> Decision:
            return evaluate(
                policy,
                agent_id,
                name,
                call_args,
                call_kwargs,
                stage=stage,
                session_id=session_id,
                conflict_strategy=conflict_strategy,
                record_args=record_args,
                redact_args=redact_args,
                shipper=shipper,
                policy_version=policy_version,
            )

        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*call_args: Any, **call_kwargs: Any) -> Any:
                decision = check(call_args, call_kwargs)
                if not decision.allowed:
                    if raise_on_deny:
                        _raise_denied(decision)
                    return decision
                return await func(*call_args, **call_kwargs)

            wrapper: Callable[..., Any] = async_wrapper
        else:

            @functools.wraps(func)
            def sync_wrapper(*call_args: Any, **call_kwargs: Any) -> Any:
                decision = check(call_args, call_kwargs)
                if not decision.allowed:
                    if raise_on_deny:
                        _raise_denied(decision)
                    return decision
                return func(*call_args, **call_kwargs)

            wrapper = sync_wrapper

        wrapper.evaluate = lambda *a, **kw: check(a, kw)  # type: ignore[attr-defined]
        wrapper.tool_name = name  # type: ignore[attr-defined]
        wrapper.__agentops_governed__ = True  # type: ignore[attr-defined]
        return wrapper

    if fn is None:
        return decorate
    return decorate(fn)


# --------------------------------------------------------------------------- #
# OTLP export to AOS Control
# --------------------------------------------------------------------------- #
_EXPORTS: dict[tuple[int, str, str], Any] = {}


def otlp_traces_endpoint(host: str) -> str:
    return f"{host.rstrip('/')}{OTEL_TRACES_PATH}"


def enable_agt_export(
    service_name: Optional[str] = None,
    host: Optional[str] = None,
    public_key: Optional[str] = None,
    secret_key: Optional[str] = None,
    *,
    provider: Any = None,
    ship_events: bool = True,
    shipper: Optional[GovernanceEventShipper] = None,
) -> Any:
    """Turn on AGT's OpenTelemetry tracer and export its spans to AOS Control.

    With ``ship_events=True`` (default) a :class:`GovernanceEventShipper`
    built from the same credentials becomes the default shipper (see
    :func:`set_default_shipper`), so every ``governed`` / ``pre_tool_call``
    decision is also posted to ``/api/public/governance-events``; pass
    ``shipper=`` to use your own instance instead.

    1. Calls ``agentmesh.governance.enable_otel(service_name=...)`` (AGT's
       bootstrap; idempotent inside AGT).
    2. Attaches ``BatchSpanProcessor(OTLPSpanExporter(endpoint=
       "<host>/api/public/otel/v1/traces", headers={"Authorization":
       "Basic <base64(public_key:secret_key)>"}))`` to the global
       ``TracerProvider`` — the one AGT installed, or the application's own
       if one was already set (``set_tracer_provider`` is first-wins).

    ``host`` / ``public_key`` / ``secret_key`` default to ``AGENTOPS_HOST``
    (or ``AGENTOPS_BASE_URL``), ``AGENTOPS_PUBLIC_KEY`` and
    ``AGENTOPS_SECRET_KEY``. Idempotent: calling it again with the same
    endpoint and key adds nothing. If credentials are missing, the tracer is
    still enabled, a warning is logged and no exporter is attached (the same
    warn-and-continue behaviour as ``agentops.init()``).

    ``provider`` lets you attach to a specific ``TracerProvider`` instead of
    the global one. Returns the provider the exporter was attached to; call
    ``provider.force_flush()`` before a short-lived process exits.
    """
    agt = _agt()
    try:
        from opentelemetry import trace  # type: ignore[import-not-found]
        from opentelemetry.sdk.resources import Resource  # type: ignore[import-not-found]
        from opentelemetry.sdk.trace import TracerProvider  # type: ignore[import-not-found]
        from opentelemetry.sdk.trace.export import BatchSpanProcessor  # type: ignore[import-not-found]
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (  # type: ignore[import-not-found]
            OTLPSpanExporter,
        )
    except ImportError as e:
        raise ImportError(
            "enable_agt_export requires opentelemetry-sdk and "
            f"opentelemetry-exporter-otlp-proto-http. Install the extra: {_INSTALL_HINT}"
        ) from e

    resolved_service_name = service_name or DEFAULT_SERVICE_NAME
    agt.enable_otel(service_name=resolved_service_name)

    if provider is None:
        provider = trace.get_tracer_provider()
        if not hasattr(provider, "add_span_processor"):
            # Still the API's proxy provider: nothing (not even AGT — e.g. its
            # enable_otel() already ran as a no-op) installed an SDK provider.
            provider = TracerProvider(resource=Resource.create({"service.name": resolved_service_name}))
            trace.set_tracer_provider(provider)

    resolved_host = (host or get_host() or "").strip()
    resolved_public = public_key or get_public_key()
    resolved_secret = secret_key or get_secret_key()

    missing = [
        name
        for name, value in (
            ("AGENTOPS_HOST", resolved_host),
            ("AGENTOPS_PUBLIC_KEY", resolved_public),
            ("AGENTOPS_SECRET_KEY", resolved_secret),
        )
        if not value
    ]
    if shipper is not None:
        set_default_shipper(shipper)

    if missing:
        _log.warning(
            "enable_agt_export: missing %s — AGT spans will be created but not exported to AOS Control.",
            ", ".join(missing),
        )
        return provider

    if ship_events and shipper is None:
        _ensure_default_shipper(resolved_host, str(resolved_public), str(resolved_secret))

    endpoint = otlp_traces_endpoint(resolved_host)
    key = (id(provider), endpoint, str(resolved_public))
    if key in _EXPORTS:
        return provider

    exporter = OTLPSpanExporter(
        endpoint=endpoint,
        headers={"Authorization": basic_auth_header(str(resolved_public), str(resolved_secret))},
    )
    processor = BatchSpanProcessor(exporter)
    provider.add_span_processor(processor)
    _EXPORTS[key] = processor
    _log.info("AGT spans exporting to %s (service=%s)", endpoint, resolved_service_name)
    return provider


def _reset_for_tests() -> None:
    """Forget cached engines, attached exporters and the default shipper (tests only)."""
    global _DEFAULT_SHIPPER
    _ENGINES.clear()
    _EXPORTS.clear()
    if _DEFAULT_SHIPPER is not None:
        try:
            _DEFAULT_SHIPPER.shutdown(timeout=1.0)
        except Exception:  # pragma: no cover
            pass
    _DEFAULT_SHIPPER = None


__all__ = [
    "ARGS_MAX_BYTES",
    "ATTR_SESSION_ID",
    "ATTR_TOOL_ARGS",
    "ATTR_TOOL_NAME",
    "Decision",
    "GovernanceEventShipper",
    "OTEL_TRACES_PATH",
    "PolicyDenied",
    "SPAN_NAME",
    "basic_auth_header",
    "build_context",
    "clear_engine_cache",
    "enable_agt_export",
    "evaluate",
    "get_default_shipper",
    "get_engine",
    "governed",
    "otlp_traces_endpoint",
    "pre_tool_call",
    "serialize_args",
    "set_default_shipper",
]
