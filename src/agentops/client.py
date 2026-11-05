from __future__ import annotations

import logging
import os
from typing import Any, Optional, Union

import requests

from . import _cloud_run_auth, _http
from ._env import basic_auth_header, get_host, get_public_key, get_secret_key
from .policies import (
    DEFAULT_CACHE_TTL_S,
    DEFAULT_LABEL,
    AgentPolicy,
    PolicyCache,
    PolicyNotFound,
    parse_policy_response,
    policy_url,
    resolve_after_fetch_error,
)

_log = logging.getLogger("agentops")

# Bridge AGENTOPS_* env vars to LANGFUSE_* so the @observe decorator's
# internal singleton picks up the correct host and credentials.
_BRIDGE_MAP = {
    "LANGFUSE_HOST": get_host,
    "LANGFUSE_BASE_URL": get_host,
    "LANGFUSE_PUBLIC_KEY": get_public_key,
    "LANGFUSE_SECRET_KEY": get_secret_key,
}
for _lf_key, _getter in _BRIDGE_MAP.items():
    if not os.environ.get(_lf_key):
        _val = _getter()
        if _val:
            os.environ[_lf_key] = _val

from .tables import (
    TableData,
    ensure_cms_name,
    ensure_tables_name,
    parse_table_from_prompt_response,
    row_as_name_map,
    table_lookup,
    table_search,
    table_similarity_named,
)


class AgentOps:
    """
    AgentOps-branded wrapper around the underlying Python client.

    - Reads configuration from AGENTOPS_* env vars (fallback to AGENTOPS_*).
    - Forwards all attributes/methods to the underlying client instance.
    """

    def __init__(
        self,
        *,
        host: Optional[str] = None,
        public_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self._host = host or get_host()
        self._public_key = public_key or get_public_key()
        self._secret_key = secret_key or get_secret_key()
        self._enabled = True
        self._client: Any = None
        self._policy_cache = PolicyCache()

        missing: list[str] = []
        if not self._host:
            missing.append("AGENTOPS_HOST")
        if not self._public_key:
            missing.append("AGENTOPS_PUBLIC_KEY")
        if not self._secret_key:
            missing.append("AGENTOPS_SECRET_KEY")

        if missing:
            _log.warning(
                "AgentOps SDK disabled — missing env var(s): %s. "
                "Tracing, content, and guardrail calls will be no-ops.",
                ", ".join(missing),
            )
            self._enabled = False
            return

        # Install Cloud Run OIDC interceptor BEFORE constructing the
        # Langfuse client so the very first ingestion call carries the
        # X-Serverless-Authorization header. No-op unless the operator
        # opts in via AGENTOPS_CLOUD_RUN_INVOKER_AUTH=true.
        _cloud_run_auth.install(self._host)

        try:
            from langfuse import Langfuse  # type: ignore[import-untyped]
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "Missing dependency 'langfuse'. Install agentops-sdk with dependencies: "
                "pip install agentops-sdk"
            ) from e

        self._client = Langfuse(
            host=self._host,
            public_key=self._public_key,
            secret_key=self._secret_key,
            **kwargs,
        )

    @property
    def enabled(self) -> bool:
        """Whether the SDK is configured and active."""
        return self._enabled

    def get_content(self, name: str, *args: Any, **kwargs: Any) -> Any:
        """Fetch managed CMS/content by name (policies, templates, non-chat assets).

        If ``name`` does not start with ``cms/``, the ``cms/`` prefix is added
        automatically (e.g. ``"faq/intro"`` → ``"cms/faq/intro"``).

        Uses the same registry API as :meth:`get_prompt`.
        """
        if not self._enabled:
            _log.debug("get_content(%s) skipped — SDK disabled", name)
            return None

        full = ensure_cms_name(name)
        return self._client.get_prompt(full, *args, **kwargs)

    def get_table(self, name: str, *args: Any, **kwargs: Any) -> Any:
        """Fetch a knowledge table by name.

        If ``name`` does not start with ``tables/``, that prefix is added
        (e.g. ``"sku-list"`` → ``"tables/sku-list"``).

        Returns the same structure as :meth:`get_prompt` (raw prompt payload). Use
        :meth:`get_table_data` to parse JSON into rows/columns.
        """
        if not self._enabled:
            _log.debug("get_table(%s) skipped — SDK disabled", name)
            return None

        full = ensure_tables_name(name)
        return self._client.get_prompt(full, *args, **kwargs)

    def get_table_data(self, name: str, **kwargs: Any) -> Optional[TableData]:
        """Fetch a table and parse the prompt body as ``{columns, rows}`` JSON."""
        if not self._enabled:
            return None

        raw = self.get_table(name, **kwargs)
        return parse_table_from_prompt_response(raw)

    def lookup_table(
        self,
        name: str,
        column: str,
        value: Any,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Exact match on a column **name**; returns rows as ``{ColumnName: value, ...}``."""
        if not self._enabled:
            return []

        data = self.get_table_data(name, **kwargs)
        if data is None:
            return []
        rows = table_lookup(data, column, value)
        return [row_as_name_map(data, r) for r in rows]

    def search_table(
        self,
        name: str,
        query: str,
        *,
        limit: int = 20,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Case-insensitive substring search across all cells (each row at most once)."""
        if not self._enabled:
            return []

        data = self.get_table_data(name, **kwargs)
        if data is None:
            return []
        raw_rows = table_search(data, query, limit=limit)
        return [row_as_name_map(data, r) for r in raw_rows]

    def search_table_similar(
        self,
        name: str,
        query: str,
        *,
        limit: int = 20,
        threshold: float = 0.4,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Fuzzy similarity search; returns up to ``limit`` rows with ``_score`` / ``_matched_column``."""
        if not self._enabled:
            return []

        data = self.get_table_data(name, **kwargs)
        if data is None:
            return []
        return table_similarity_named(
            data, query, limit=limit, threshold=threshold
        )

    def get_file(
        self,
        name: str,
        *,
        version: Optional[int] = None,
        download: bool = False,
        download_dir: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        """Fetch a CMS file asset by slug.

        Parameters
        ----------
        name:
            The file slug (e.g. ``"my-config"``).
        version:
            Optional integer version. If omitted, the latest version is returned.
        download:
            If ``True``, download the file content to ``download_dir`` (or cwd)
            and add a ``local_path`` key to the returned dict.
        download_dir:
            Directory to save the file to (defaults to current working directory).

        Returns
        -------
        dict with ``name``, ``version``, ``fileName``, ``contentType``,
        ``contentLength``, ``url`` (signed download URL), ``urlExpiry``, and
        optionally ``local_path``.  Returns ``None`` when SDK is disabled.
        """
        if not self._enabled:
            _log.debug("get_file(%s) skipped — SDK disabled", name)
            return None

        params: dict[str, Any] = {"name": name}
        if version is not None:
            params["version"] = version

        resp = requests.get(
            f"{self._host}/api/public/files",
            params=params,
            headers=basic_auth_header(self._public_key, self._secret_key),
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        if download:
            dest = download_dir or os.getcwd()
            local_path = os.path.join(dest, data["fileName"])
            dl_resp = requests.get(data["url"], timeout=120, stream=True)
            dl_resp.raise_for_status()
            with open(local_path, "wb") as f:
                for chunk in dl_resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            data["local_path"] = local_path

        return data

    def get_policy(
        self,
        agent_name: str,
        label: str = DEFAULT_LABEL,
        *,
        cache_ttl_seconds: int = DEFAULT_CACHE_TTL_S,
        fallback: Union[str, AgentPolicy, None] = None,
        fetch_timeout_seconds: float = 10.0,
    ) -> Optional[AgentPolicy]:
        """Fetch the agent's tool-call policy (AGT manifest) from the registry.

        ``GET /api/public/policies/{agent_name}?label={label}``. Results are
        cached for ``cache_ttl_seconds`` (``0`` disables the cache); when a
        refresh fails the previously cached version is returned with a
        warning, and when nothing is cached ``fallback`` (an AGT YAML string
        or :class:`AgentPolicy`) is used — the same semantics as
        ``get_prompt``. A ``404`` raises :class:`PolicyNotFound` (never masked
        by the cache or ``fallback``); other failures raise
        :class:`PolicyFetchError` / the transport error.

        Returns ``None`` when the SDK is disabled (unless ``fallback`` is given).
        Pass ``policy.to_agt_policy_spec()`` to ``agentops.governance``.
        """
        if not self._enabled:
            _log.debug("get_policy(%s) skipped — SDK disabled", agent_name)
            return AgentPolicy.from_fallback(agent_name, fallback) if fallback is not None else None

        cached, fresh = self._policy_cache.get(agent_name, label, cache_ttl_seconds)
        if cached is not None and fresh:
            return cached
        try:
            policy = self._fetch_policy(agent_name, label, timeout=fetch_timeout_seconds)
        except Exception as e:
            if isinstance(e, PolicyNotFound):
                self._policy_cache.drop(agent_name, label)
            return resolve_after_fetch_error(
                e, agent_name=agent_name, label=label, stale=cached, fallback=fallback
            )
        self._policy_cache.put(agent_name, label, policy)
        return policy

    def _fetch_policy(self, agent_name: str, label: str, *, timeout: float) -> AgentPolicy:
        resp = _http.get_session().get(
            policy_url(self._host, agent_name),
            params={"label": label},
            headers={"Authorization": _http.basic_auth_header(self._public_key, self._secret_key)},
            timeout=timeout,
        )
        try:
            body = resp.json()
        except Exception:
            body = None
        return parse_policy_response(
            resp.status_code,
            body,
            agent_name=agent_name,
            label=label,
            text=getattr(resp, "text", "") or "",
        )

    def clear_policy_cache(self) -> None:
        """Forget cached policies so the next ``get_policy`` refetches."""
        self._policy_cache.clear()

    def check_guardrails(
        self,
        text: str,
        *,
        stage: str = "pre_input",
        agent_name: Optional[str] = None,
        config: Optional[dict[str, Any]] = None,
        use_llm: bool = False,
    ) -> dict[str, Any]:
        """Run guardrail checks server-side.

        Parameters
        ----------
        text:
            The text to check (user input or agent output).
        stage:
            ``"pre_input"`` for user messages, ``"post_output"`` for agent responses.
        agent_name:
            Load guardrail config from this agent's registry entry.
        config:
            Explicit guardrail config dict. Overrides ``agent_name`` lookup.
        use_llm:
            When ``True``, run an additional LLM-based jailbreak/injection
            classifier on the server. Adds latency (~1-3 s) but provides
            much higher accuracy than regex alone. Requires LLM API keys
            to be configured in the project settings.

        Returns
        -------
        dict with ``action`` (``"allow"``, ``"block"``, ``"transform"``),
        ``reasons``, ``tags``, ``metadata``, and optionally ``transformedText``.
        When ``use_llm=True``, ``metadata.llmDetection`` contains the LLM
        classification result with ``isJailbreak``, ``confidence``,
        ``category``, and ``reasoning`` fields.
        When SDK is disabled, returns ``{"action": "allow"}`` (fail-open).
        """
        if not self._enabled:
            _log.debug("check_guardrails skipped — SDK disabled")
            return {"action": "allow", "reasons": [], "tags": [], "metadata": {}}

        body: dict[str, Any] = {"stage": stage, "text": text}
        if config is not None:
            body["config"] = config
        elif agent_name is not None:
            body["agentName"] = agent_name
        if use_llm:
            body["useLlm"] = True

        resp = requests.post(
            f"{self._host}/api/public/guardrails",
            json=body,
            headers=basic_auth_header(self._public_key, self._secret_key),
            timeout=30 if use_llm else 10,
        )
        resp.raise_for_status()
        return resp.json()

    def _check_tool_stage(
        self,
        stage: str,
        tool_name: str,
        key: str,
        payload: Any,
        agent_name: Optional[str],
        config: Optional[dict[str, Any]],
    ) -> dict[str, Any]:
        if not self._enabled:
            return {"action": "allow", "reasons": [], "tags": [], "metadata": {}}
        body: dict[str, Any] = {"stage": stage, "toolName": tool_name, key: payload}
        if config is not None:
            body["config"] = config
        elif agent_name is not None:
            body["agentName"] = agent_name
        resp = requests.post(
            f"{self._host}/api/public/guardrails",
            json=body,
            headers=basic_auth_header(self._public_key, self._secret_key),
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()

    def check_tool_call(
        self,
        tool_name: str,
        args: Any,
        *,
        agent_name: Optional[str] = None,
        config: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Check a tool call's arguments server-side, before running the tool.

        Looks for URLs that reach an internal address, carry data inline
        (``data:``), embed credentials, or point outside the agent's
        ``preTool.egress.allowedDomains``, and for credential-shaped values in
        the arguments. Which checks run, and whether a finding is a
        ``"block"`` or only a flag, comes from the agent's guardrail config
        (``agent_name``) or an explicit ``config``. With no config nothing is
        checked and the result is ``"allow"``.

        Returns the same shape as :meth:`check_guardrails` (``action``,
        ``reasons``, ``tags``, ``metadata``). Findings carry the host and the
        argument path, never the URL or the secret. Fails open when the SDK is
        disabled.
        """
        return self._check_tool_stage("pre_tool", tool_name, "args", args, agent_name, config)

    def check_tool_result(
        self,
        tool_name: str,
        result: Any,
        *,
        agent_name: Optional[str] = None,
        config: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Check what a tool returned before the model sees it.

        Looks for credential-shaped values and for instruction-like text in
        retrieved content (indirect prompt injection), per the agent's
        ``postTool`` guardrail config. ``"block"`` means the caller should
        drop the result; the default is to flag it. Same return shape and
        fail-open behaviour as :meth:`check_tool_call`.
        """
        return self._check_tool_stage("post_tool", tool_name, "result", result, agent_name, config)

    def update_current_trace(
        self,
        *,
        name: Optional[str] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        tags: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Update the active trace created by ``@observe()``.

        Must be called from inside a function decorated with ``@observe()``.
        """
        if not self._enabled:
            return

        from langfuse.decorators import langfuse_context  # type: ignore[import-untyped]

        update_kwargs: dict[str, Any] = {**kwargs}
        if name is not None:
            update_kwargs["name"] = name
        if session_id is not None:
            update_kwargs["session_id"] = session_id
        if user_id is not None:
            update_kwargs["user_id"] = user_id
        if metadata is not None:
            update_kwargs["metadata"] = metadata
        if tags is not None:
            update_kwargs["tags"] = tags

        langfuse_context.update_current_trace(**update_kwargs)

    def update_current_observation(
        self,
        *,
        model: Optional[str] = None,
        input: Optional[Any] = None,
        output: Optional[Any] = None,
        usage_details: Optional[dict[str, Any]] = None,
        cost_details: Optional[dict[str, Any]] = None,
        metadata: Optional[dict[str, Any]] = None,
        level: Optional[str] = None,
        status_message: Optional[str] = None,
        completion_start_time: Optional[Any] = None,
        model_parameters: Optional[dict[str, Any]] = None,
        version: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Update the active observation created by ``@observe()``.

        Must be called from inside a function decorated with ``@observe()``.
        For generation observations (``@observe(as_type="generation")``),
        ``model`` + ``usage_details`` are what unlocks server-side cost
        calculation — without them the ingestion path sets
        ``provided_model_name = null`` and ``findModel`` is skipped.

        Note: the wrapper proxies to ``langfuse_context`` (the v2 decorator
        singleton) rather than ``self._client`` because the underlying
        ``Langfuse`` class itself has no such method in either v2 or v3 —
        observation state is owned by the decorator context, not the client.
        """
        if not self._enabled:
            return

        from langfuse.decorators import langfuse_context  # type: ignore[import-untyped]

        update_kwargs: dict[str, Any] = {**kwargs}
        if model is not None:
            update_kwargs["model"] = model
        if input is not None:
            update_kwargs["input"] = input
        if output is not None:
            update_kwargs["output"] = output
        if usage_details is not None:
            update_kwargs["usage_details"] = usage_details
        if cost_details is not None:
            update_kwargs["cost_details"] = cost_details
        if metadata is not None:
            update_kwargs["metadata"] = metadata
        if level is not None:
            update_kwargs["level"] = level
        if status_message is not None:
            update_kwargs["status_message"] = status_message
        if completion_start_time is not None:
            update_kwargs["completion_start_time"] = completion_start_time
        if model_parameters is not None:
            update_kwargs["model_parameters"] = model_parameters
        if version is not None:
            update_kwargs["version"] = version

        langfuse_context.update_current_observation(**update_kwargs)

    def flush(self) -> None:
        """Flush all pending traces — both the API client and the ``@observe`` decorator buffer."""
        if not self._enabled:
            return

        from langfuse.decorators import langfuse_context  # type: ignore[import-untyped]

        langfuse_context.flush()
        self._client.flush()

    @property
    def client(self) -> Any:
        """Access the underlying tracing client directly."""

        return self._client

    def __getattr__(self, name: str) -> Any:
        if not self._enabled or self._client is None:
            return lambda *a, **kw: None
        return getattr(self._client, name)


def init(**kwargs: Any) -> AgentOps:
    """Convenience constructor: init(...) -> AgentOps(...)."""

    return AgentOps(**kwargs)


