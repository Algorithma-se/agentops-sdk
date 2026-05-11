from __future__ import annotations

import logging
import os
from typing import Any, Optional

import httpx

from . import _cloud_run_auth
from ._env import get_host, get_public_key, get_secret_key

_log = logging.getLogger("agentops")
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


class AsyncAgentOps:
    """Async variant of :class:`AgentOps`.

    Uses ``httpx.AsyncClient`` for HTTP and the Langfuse ``async_api``
    property for async tracing operations.  All public methods are
    coroutines and must be awaited.

    Usage::

        async with AsyncAgentOps() as client:
            content = await client.get_content("faq/intro")
            result = await client.check_guardrails("user input", use_llm=True)
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
        self._http: Optional[httpx.AsyncClient] = None

        missing: list[str] = []
        if not self._host:
            missing.append("AGENTOPS_HOST")
        if not self._public_key:
            missing.append("AGENTOPS_PUBLIC_KEY")
        if not self._secret_key:
            missing.append("AGENTOPS_SECRET_KEY")

        if missing:
            _log.warning(
                "AsyncAgentOps SDK disabled — missing env var(s): %s. "
                "All calls will be no-ops.",
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

    async def _get_http(self) -> httpx.AsyncClient:
        if self._http is None or self._http.is_closed:
            self._http = httpx.AsyncClient(timeout=30.0)
        return self._http

    async def __aenter__(self) -> "AsyncAgentOps":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    @property
    def enabled(self) -> bool:
        """Whether the SDK is configured and active."""
        return self._enabled

    async def close(self) -> None:
        """Close the underlying HTTP client and flush traces."""
        if self._http and not self._http.is_closed:
            await self._http.aclose()
        if self._client:
            self._client.flush()

    async def get_content(self, name: str, *args: Any, **kwargs: Any) -> Any:
        """Fetch managed CMS/content by name (async)."""
        if not self._enabled:
            return None
        full = ensure_cms_name(name)
        return self._client.get_prompt(full, *args, **kwargs)

    async def get_table(self, name: str, *args: Any, **kwargs: Any) -> Any:
        """Fetch a knowledge table by name (async)."""
        if not self._enabled:
            return None
        full = ensure_tables_name(name)
        return self._client.get_prompt(full, *args, **kwargs)

    async def get_table_data(self, name: str, **kwargs: Any) -> Optional[TableData]:
        """Fetch a table and parse the prompt body as ``{columns, rows}`` JSON."""
        if not self._enabled:
            return None
        raw = await self.get_table(name, **kwargs)
        return parse_table_from_prompt_response(raw)

    async def lookup_table(
        self,
        name: str,
        column: str,
        value: Any,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Exact match on a column; returns rows as ``{ColumnName: value, ...}``."""
        if not self._enabled:
            return []
        data = await self.get_table_data(name, **kwargs)
        if data is None:
            return []
        rows = table_lookup(data, column, value)
        return [row_as_name_map(data, r) for r in rows]

    async def search_table(
        self,
        name: str,
        query: str,
        *,
        limit: int = 20,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Case-insensitive substring search across all cells."""
        if not self._enabled:
            return []
        data = await self.get_table_data(name, **kwargs)
        if data is None:
            return []
        raw_rows = table_search(data, query, limit=limit)
        return [row_as_name_map(data, r) for r in raw_rows]

    async def search_table_similar(
        self,
        name: str,
        query: str,
        *,
        limit: int = 20,
        threshold: float = 0.4,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Fuzzy similarity search; returns up to ``limit`` rows."""
        if not self._enabled:
            return []
        data = await self.get_table_data(name, **kwargs)
        if data is None:
            return []
        return table_similarity_named(data, query, limit=limit, threshold=threshold)

    async def get_file(
        self,
        name: str,
        *,
        version: Optional[int] = None,
        download: bool = False,
        download_dir: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        """Fetch a CMS file asset by slug (async).

        Returns ``None`` when SDK is disabled.
        """
        if not self._enabled:
            return None
        http = await self._get_http()
        params: dict[str, Any] = {"name": name}
        if version is not None:
            params["version"] = version

        resp = await http.get(
            f"{self._host}/api/public/files",
            params=params,
            headers={"Authorization": f"Bearer {self._secret_key}"},
        )
        resp.raise_for_status()
        data = resp.json()

        if download:
            import aiofiles  # type: ignore[import-untyped]

            dest = download_dir or os.getcwd()
            local_path = os.path.join(dest, data["fileName"])
            async with http.stream("GET", data["url"]) as dl_resp:
                dl_resp.raise_for_status()
                async with aiofiles.open(local_path, "wb") as f:
                    async for chunk in dl_resp.aiter_bytes(chunk_size=8192):
                        await f.write(chunk)
            data["local_path"] = local_path

        return data

    async def check_guardrails(
        self,
        text: str,
        *,
        stage: str = "pre_input",
        agent_name: Optional[str] = None,
        config: Optional[dict[str, Any]] = None,
        use_llm: bool = False,
    ) -> dict[str, Any]:
        """Run guardrail checks server-side (async).

        Returns ``{"action": "allow"}`` (fail-open) when SDK is disabled.
        """
        if not self._enabled:
            return {"action": "allow", "reasons": [], "tags": [], "metadata": {}}
        http = await self._get_http()
        body: dict[str, Any] = {"stage": stage, "text": text}
        if config is not None:
            body["config"] = config
        elif agent_name is not None:
            body["agentName"] = agent_name
        if use_llm:
            body["useLlm"] = True

        timeout = 30.0 if use_llm else 10.0
        resp = await http.post(
            f"{self._host}/api/public/guardrails",
            json=body,
            headers={"Authorization": f"Bearer {self._secret_key}"},
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json()

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
        """Update the active trace created by ``@observe()``."""
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

    async def flush(self) -> None:
        """Flush all pending traces."""
        if not self._enabled:
            return

        from langfuse.decorators import langfuse_context  # type: ignore[import-untyped]

        langfuse_context.flush()
        self._client.flush()

    @property
    def client(self) -> Any:
        """Access the underlying Langfuse client directly."""
        return self._client

    def __getattr__(self, name: str) -> Any:
        if not self._enabled or self._client is None:
            return lambda *a, **kw: None
        return getattr(self._client, name)


async def async_init(**kwargs: Any) -> AsyncAgentOps:
    """Convenience constructor: async_init(...) -> AsyncAgentOps(...)."""
    return AsyncAgentOps(**kwargs)
