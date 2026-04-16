from .client import AgentOps, init
from .async_client import AsyncAgentOps, async_init
from .guardrails import Guardrails, GuardrailResult
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


__all__ = [
    "AgentOps",
    "AsyncAgentOps",
    "CMS_PREFIX",
    "GuardrailResult",
    "Guardrails",
    "TABLES_PREFIX",
    "TableData",
    "async_init",
    "ensure_cms_name",
    "ensure_tables_name",
    "init",
    "observe",
    "parse_table_from_prompt_response",
    "row_as_name_map",
    "table_lookup",
    "table_search",
    "table_similarity_named",
    "table_similarity_rows",
    "update_current_trace",
]


