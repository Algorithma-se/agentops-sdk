"""Agent tool-call policies fetched from the AOS Control registry.

``GET {host}/api/public/policies/{agentName}?label=production`` returns the
AGT manifest (YAML) — and optionally Rego — that the registry holds for an
agent, or a policy translated from the agent's legacy ``preTool.toolAllowlist``
(``source == "translated_allowlist"``). :class:`AgentPolicy` is the parsed
response; :meth:`AgentPolicy.to_agt_policy_spec` hands the manifest to
``agentops.governance.governed(policy=...)`` / ``pre_tool_call(...)`` /
``Guardrails.pre_tool_call(...)`` unchanged.

Caching mirrors ``get_prompt``: results are cached per client for
``cache_ttl_seconds``; when a refresh fails the stale copy is served (with a
warning) and, failing that, ``fallback`` is used. A ``404`` is definitive and
raises :class:`PolicyNotFound` so callers can bridge to their own default.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Union

_log = logging.getLogger("agentops.policies")

POLICIES_PATH = "/api/public/policies"
DEFAULT_LABEL = "production"
DEFAULT_CACHE_TTL_S = 60


class AgentOpsError(Exception):
    """Base class for errors raised by the AgentOps SDK's own API calls."""


class PolicyNotFound(AgentOpsError):
    """The registry has no policy for ``agent_name`` under ``label``."""

    def __init__(self, agent_name: str, label: str, message: Optional[str] = None):
        self.agent_name = agent_name
        self.label = label
        super().__init__(
            message or f"No policy found for agent '{agent_name}' with label '{label}'"
        )


class PolicyFetchError(AgentOpsError):
    """The policy endpoint answered with an unexpected status or payload."""

    def __init__(self, agent_name: str, label: str, status_code: Optional[int], detail: str):
        self.agent_name = agent_name
        self.label = label
        self.status_code = status_code
        super().__init__(
            f"Failed to fetch policy for agent '{agent_name}' (label '{label}')"
            + (f": HTTP {status_code}" if status_code is not None else "")
            + (f" — {detail}" if detail else "")
        )


@dataclass
class AgentPolicy:
    """One version of an agent's tool-call policy as served by the registry."""

    agent_name: str
    version: int
    labels: list[str] = field(default_factory=list)
    manifest: str = ""
    rego: Optional[str] = None
    source: str = "registry"  # registry | translated_allowlist | fallback
    updated_at: Optional[str] = None
    agent_entry_id: Optional[str] = None
    format: str = "agt-yaml"

    def to_agt_policy_spec(self) -> str:
        """The manifest as a ``PolicySpec`` for ``agentops.governance``.

        ``governed(policy=...)`` / ``pre_tool_call(policy, ...)`` accept YAML
        text directly (anything that is not an existing file path is parsed
        as inline YAML), so the manifest string is returned as-is.
        """
        if not self.manifest or not self.manifest.strip():
            raise ValueError(
                f"Policy for agent '{self.agent_name}' (version {self.version}) has an empty manifest"
            )
        return self.manifest

    @classmethod
    def from_response(cls, data: dict[str, Any], *, agent_name: Optional[str] = None) -> "AgentPolicy":
        """Build from the ``GET /api/public/policies/{agentName}`` JSON body."""
        if not isinstance(data, dict):
            raise TypeError(f"policy response must be an object, got {type(data).__name__}")
        manifest = data.get("manifest")
        if not isinstance(manifest, str):
            raise ValueError("policy response is missing the 'manifest' string")
        version_raw = data.get("version", 0)
        try:
            version = int(version_raw)
        except (TypeError, ValueError):
            version = 0
        labels_raw = data.get("labels") or []
        labels = [str(x) for x in labels_raw] if isinstance(labels_raw, (list, tuple)) else []
        return cls(
            agent_name=str(data.get("agentName") or agent_name or ""),
            version=version,
            labels=labels,
            manifest=manifest,
            rego=data.get("rego") if isinstance(data.get("rego"), str) else None,
            source=str(data.get("source") or "registry"),
            updated_at=data.get("updatedAt"),
            agent_entry_id=data.get("agentEntryId"),
            format=str(data.get("format") or "agt-yaml"),
        )

    @classmethod
    def from_fallback(cls, agent_name: str, fallback: Union[str, "AgentPolicy"]) -> "AgentPolicy":
        if isinstance(fallback, AgentPolicy):
            return fallback
        return cls(agent_name=agent_name, version=0, labels=[], manifest=str(fallback), source="fallback")


def policy_url(host: str, agent_name: str) -> str:
    from urllib.parse import quote

    return f"{host.rstrip('/')}{POLICIES_PATH}/{quote(agent_name, safe='')}"


def _error_detail(status_code: int, body: Any, text: str) -> str:
    if isinstance(body, dict):
        return str(body.get("message") or body.get("error") or "")
    return (text or "")[:200]


def parse_policy_response(
    status_code: int,
    body: Any,
    *,
    agent_name: str,
    label: str,
    text: str = "",
) -> AgentPolicy:
    """Turn a policy endpoint response into an :class:`AgentPolicy` or raise.

    Shared by the sync and async clients so both apply exactly the same
    status handling: 200 → policy, 404 → :class:`PolicyNotFound`, anything
    else → :class:`PolicyFetchError`.
    """
    if status_code == 404:
        raise PolicyNotFound(agent_name, label, _error_detail(status_code, body, text) or None)
    if status_code < 200 or status_code >= 300:
        raise PolicyFetchError(agent_name, label, status_code, _error_detail(status_code, body, text))
    try:
        return AgentPolicy.from_response(body, agent_name=agent_name)
    except (TypeError, ValueError) as e:
        raise PolicyFetchError(agent_name, label, status_code, str(e)) from e


class PolicyCache:
    """Per-client TTL cache with stale-on-error semantics (thread-safe)."""

    def __init__(self) -> None:
        self._entries: dict[tuple[str, str], tuple[AgentPolicy, float]] = {}
        self._lock = threading.Lock()

    def get(self, agent_name: str, label: str, ttl_s: float) -> tuple[Optional[AgentPolicy], bool]:
        """Return ``(policy, fresh)``; ``policy`` is ``None`` when absent."""
        with self._lock:
            hit = self._entries.get((agent_name, label))
        if hit is None:
            return None, False
        policy, fetched_at = hit
        fresh = ttl_s > 0 and (time.monotonic() - fetched_at) < ttl_s
        return policy, fresh

    def put(self, agent_name: str, label: str, policy: AgentPolicy) -> None:
        with self._lock:
            self._entries[(agent_name, label)] = (policy, time.monotonic())

    def drop(self, agent_name: str, label: str) -> None:
        with self._lock:
            self._entries.pop((agent_name, label), None)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()


def resolve_after_fetch_error(
    error: Exception,
    *,
    agent_name: str,
    label: str,
    stale: Optional[AgentPolicy],
    fallback: Union[str, AgentPolicy, None],
) -> AgentPolicy:
    """Shared stale/fallback decision used by both clients when a refresh fails.

    Order: stale cached copy → ``fallback`` → re-raise. ``PolicyNotFound`` is
    never masked by a stale copy or fallback: the registry has explicitly
    said there is no policy, and callers (``Guardrails.from_agent``) catch it.
    """
    if isinstance(error, PolicyNotFound):
        raise error
    if stale is not None:
        _log.warning(
            "get_policy(%s, label=%s): refresh failed (%s); serving cached version %s",
            agent_name,
            label,
            error,
            stale.version,
        )
        return stale
    if fallback is not None:
        _log.warning(
            "get_policy(%s, label=%s): fetch failed (%s); using fallback policy",
            agent_name,
            label,
            error,
        )
        return AgentPolicy.from_fallback(agent_name, fallback)
    raise error


__all__ = [
    "DEFAULT_CACHE_TTL_S",
    "DEFAULT_LABEL",
    "POLICIES_PATH",
    "AgentOpsError",
    "AgentPolicy",
    "PolicyCache",
    "PolicyFetchError",
    "PolicyNotFound",
    "parse_policy_response",
    "policy_url",
    "resolve_after_fetch_error",
]
