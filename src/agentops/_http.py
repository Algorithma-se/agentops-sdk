"""Shared synchronous HTTP transport for the SDK's own AOS Control calls.

``client.get_policy`` and ``governance_events.GovernanceEventShipper`` go
through one process-wide ``requests.Session`` so that connection pooling,
the Cloud Run OIDC interceptor (``_cloud_run_auth`` patches
``requests.Session.send`` in place, so it covers this session too) and test
doubles all live in one place. Tests swap the transport with
:func:`set_session`.
"""

from __future__ import annotations

import base64
import threading
from typing import Any, Optional

import requests

_LOCK = threading.Lock()
_SESSION: Optional[Any] = None

DEFAULT_TIMEOUT_S = 10.0


def basic_auth_header(public_key: str, secret_key: str) -> str:
    """``Authorization`` header value for AOS Control's public API
    (``Basic base64(public_key:secret_key)``) — the full-scope scheme the
    server's ``verifyAuthHeaderAndReturnScope`` accepts."""
    token = base64.b64encode(f"{public_key}:{secret_key}".encode("utf-8")).decode("ascii")
    return f"Basic {token}"


def get_session() -> Any:
    """Return the shared ``requests.Session`` (created on first use)."""
    global _SESSION
    with _LOCK:
        if _SESSION is None:
            _SESSION = requests.Session()
        return _SESSION


def set_session(session: Optional[Any]) -> None:
    """Replace the shared session (``None`` recreates a real one lazily).

    Intended for tests and for applications that need a custom transport
    (proxies, extra adapters). Anything with ``.get`` / ``.post`` returning
    response-like objects (``status_code``, ``json()``, ``text``) works.
    """
    global _SESSION
    with _LOCK:
        _SESSION = session


__all__ = ["DEFAULT_TIMEOUT_S", "basic_auth_header", "get_session", "set_session"]
