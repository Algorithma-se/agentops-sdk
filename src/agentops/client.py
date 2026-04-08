from __future__ import annotations

import os
from typing import Any, Optional

import requests

from ._env import get_host, get_public_key, get_secret_key
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

        if not self._host:
            raise ValueError("Missing host. Set AGENTOPS_HOST (or pass host=...).")
        if not self._public_key:
            raise ValueError(
                "Missing public key. Set AGENTOPS_PUBLIC_KEY (or pass public_key=...)."
            )
        if not self._secret_key:
            raise ValueError(
                "Missing secret key. Set AGENTOPS_SECRET_KEY (or pass secret_key=...)."
            )

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

    def get_content(self, name: str, *args: Any, **kwargs: Any) -> Any:
        """Fetch managed CMS/content by name (policies, templates, non-chat assets).

        If ``name`` does not start with ``cms/``, the ``cms/`` prefix is added
        automatically (e.g. ``"faq/intro"`` → ``"cms/faq/intro"``).

        Uses the same registry API as :meth:`get_prompt`.
        """

        full = ensure_cms_name(name)
        return self._client.get_prompt(full, *args, **kwargs)

    def get_table(self, name: str, *args: Any, **kwargs: Any) -> Any:
        """Fetch a knowledge table by name.

        If ``name`` does not start with ``tables/``, that prefix is added
        (e.g. ``"sku-list"`` → ``"tables/sku-list"``).

        Returns the same structure as :meth:`get_prompt` (raw prompt payload). Use
        :meth:`get_table_data` to parse JSON into rows/columns.
        """

        full = ensure_tables_name(name)
        return self._client.get_prompt(full, *args, **kwargs)

    def get_table_data(self, name: str, **kwargs: Any) -> TableData:
        """Fetch a table and parse the prompt body as ``{columns, rows}`` JSON."""

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

        data = self.get_table_data(name, **kwargs)
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

        data = self.get_table_data(name, **kwargs)
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

        data = self.get_table_data(name, **kwargs)
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
    ) -> dict[str, Any]:
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
        optionally ``local_path``.
        """
        params: dict[str, Any] = {"name": name}
        if version is not None:
            params["version"] = version

        resp = requests.get(
            f"{self._host}/api/public/files",
            params=params,
            headers={"Authorization": f"Bearer {self._secret_key}"},
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
        """
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
            headers={"Authorization": f"Bearer {self._secret_key}"},
            timeout=30 if use_llm else 10,
        )
        resp.raise_for_status()
        return resp.json()

    @property
    def client(self) -> Any:
        """Access the underlying client directly."""

        return self._client

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


def init(**kwargs: Any) -> AgentOps:
    """Convenience constructor: init(...) -> AgentOps(...)."""

    return AgentOps(**kwargs)


