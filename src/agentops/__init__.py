from .client import AgentOps, init
from .async_client import AsyncAgentOps, async_init
from .guardrails import (
    GuardrailConfigUnavailableError,
    GuardrailConfigWarning,
    GuardrailResult,
    Guardrails,
)
from .policies import AgentOpsError, AgentPolicy, PolicyFetchError, PolicyNotFound
from .talk import (
    TALK_TURN_ID_HEADER,
    echo_talk_turn_id,
    get_talk_turn_id,
    talk_turn_metadata,
)
from .tables import (
    CMS_PREFIX,
    TABLES_PREFIX,
    TableData,
    ensure_cms_name,
    ensure_tables_name,
    parse_table_from_prompt_response,
    row_as_name_map,
    table_lookup,
    table_search,
    table_similarity_named,
    table_similarity_rows,
)
from langfuse.decorators import observe, langfuse_context  # type: ignore[import-untyped]


def update_current_trace(
    *,
    name: str | None = None,
    session_id: str | None = None,
    user_id: str | None = None,
    **kwargs,
):
    """Convenience shortcut — update the active ``@observe()`` trace without a client instance."""
    kw = {**kwargs}
    if name is not None:
        kw["name"] = name
    if session_id is not None:
        kw["session_id"] = session_id
    if user_id is not None:
        kw["user_id"] = user_id
    langfuse_context.update_current_trace(**kw)


def update_current_observation(
    *,
    model: str | None = None,
    input=None,
    output=None,
    usage_details: dict | None = None,
    cost_details: dict | None = None,
    metadata: dict | None = None,
    level: str | None = None,
    status_message: str | None = None,
    completion_start_time=None,
    model_parameters: dict | None = None,
    version: str | None = None,
    **kwargs,
):
    """Convenience shortcut — update the active ``@observe()`` observation without a client instance.

    For generation observations, supply ``model`` + ``usage_details`` to
    unlock server-side cost calculation. The ingestion path only resolves a
    model definition (and therefore prices) when ``provided_model_name`` is
    populated, which happens automatically when ``model`` is set here.
    """
    kw = {**kwargs}
    if model is not None:
        kw["model"] = model
    if input is not None:
        kw["input"] = input
    if output is not None:
        kw["output"] = output
    if usage_details is not None:
        kw["usage_details"] = usage_details
    if cost_details is not None:
        kw["cost_details"] = cost_details
    if metadata is not None:
        kw["metadata"] = metadata
    if level is not None:
        kw["level"] = level
    if status_message is not None:
        kw["status_message"] = status_message
    if completion_start_time is not None:
        kw["completion_start_time"] = completion_start_time
    if model_parameters is not None:
        kw["model_parameters"] = model_parameters
    if version is not None:
        kw["version"] = version
    langfuse_context.update_current_observation(**kw)


# Tool-call policy via the Agent Governance Toolkit lives in
# ``agentops.governance`` and needs the optional ``governance`` extra. The
# names are resolved lazily so ``import agentops`` never imports ``agentmesh``
# (or even our own governance module) unless one of them is actually used.
_GOVERNANCE_EXPORTS = frozenset(
    {
        "Decision",
        "PolicyDenied",
        "enable_agt_export",
        "governed",
        "pre_tool_call",
        "set_default_shipper",
        "get_default_shipper",
    }
)
# Governance-event shipping (``agentops.governance_events``) — also lazy so
# ``import agentops`` stays light and never touches OpenTelemetry.
_GOVERNANCE_EVENT_EXPORTS = frozenset(
    {
        "AgentOpsAuditSink",
        "GovernanceEvent",
        "GovernanceEventShipper",
        "audit_entry_to_event",
        "decision_to_event",
        "ship_audit_log",
    }
)


def __getattr__(name: str):
    if name in _GOVERNANCE_EXPORTS:
        from . import governance as _governance

        return getattr(_governance, name)
    if name in _GOVERNANCE_EVENT_EXPORTS:
        from . import governance_events as _governance_events

        return getattr(_governance_events, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AgentOps",
    "AgentOpsAuditSink",
    "AgentOpsError",
    "AgentPolicy",
    "AsyncAgentOps",
    "CMS_PREFIX",
    "GuardrailConfigUnavailableError",
    "GuardrailConfigWarning",
    "Decision",
    "GovernanceEvent",
    "GovernanceEventShipper",
    "GuardrailResult",
    "Guardrails",
    "PolicyDenied",
    "PolicyFetchError",
    "PolicyNotFound",
    "audit_entry_to_event",
    "decision_to_event",
    "enable_agt_export",
    "get_default_shipper",
    "governed",
    "pre_tool_call",
    "set_default_shipper",
    "ship_audit_log",
    "TABLES_PREFIX",
    "TALK_TURN_ID_HEADER",
    "TableData",
    "async_init",
    "echo_talk_turn_id",
    "ensure_cms_name",
    "ensure_tables_name",
    "get_talk_turn_id",
    "init",
    "observe",
    "parse_table_from_prompt_response",
    "row_as_name_map",
    "table_lookup",
    "table_search",
    "table_similarity_named",
    "table_similarity_rows",
    "talk_turn_metadata",
    "update_current_observation",
    "update_current_trace",
]


