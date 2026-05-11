"""Optional Cloud Run service-to-service auth.

When the AgentOps server runs on Google Cloud Run with the default
invoker-locked posture, public-API calls need a Google-signed OIDC
token in addition to the AgentOps API key. This module installs a
transparent header-injection layer in front of every outbound HTTP
call made by the SDK (Langfuse internals, direct ``requests`` calls
in :mod:`agentops.client`, and the ``httpx.AsyncClient`` used in
:mod:`agentops.async_client`).

Enable by setting::

    AGENTOPS_CLOUD_RUN_INVOKER_AUTH=true

The token goes in ``X-Serverless-Authorization`` so it doesn't collide
with the SDK's own ``Authorization: Basic <pk:sk>`` header. Both
layers travel on the same request; Cloud Run strips the bearer after
validating it against IAM; the AgentOps container only sees the Basic
auth (or ``Authorization: Bearer <secret>`` on the public REST
endpoints).

Design notes:

- **Off by default.** 99% of SDK users run outside GCP. We do not want
  to even import ``google-auth`` (and thus call the metadata server)
  unless the operator opts in.
- **Soft ``google-auth`` dependency.** If the env flag is on but
  ``google-auth`` isn't installed, we log a warning and proceed as a
  no-op. The user will see a 401 from Cloud Run, which is the correct
  signal that something needs attention.
- **Idempotent.** ``install()`` is safe to call multiple times — both
  ``AgentOps`` and ``AsyncAgentOps`` call it from their constructors,
  and the user may instantiate either repeatedly.
- **Audience-scoped.** Only requests whose host matches the configured
  audience are touched. Outbound calls to OpenAI / Stripe / anything
  else are left alone — we never leak the SA identity to a third
  party.
- **Patches the transport, not the call sites.** Catches Langfuse's
  internal HTTP calls (via ``@observe`` + ``Langfuse()``) without
  requiring a Langfuse code change.

Configuration:

``AGENTOPS_CLOUD_RUN_INVOKER_AUTH``
    Set to ``true`` / ``1`` / ``yes`` / ``on`` to enable. Default off.

``AGENTOPS_CLOUD_RUN_INVOKER_AUDIENCE``
    Optional. Overrides the OIDC audience claim. Defaults to
    ``AGENTOPS_HOST``. Use this if the SDK calls the service via a
    load-balancer / vanity domain but the IAM binding is on the
    underlying ``*.run.app`` URL.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Optional
from urllib.parse import urlparse

_log = logging.getLogger("agentops")

ENV_ENABLED = "AGENTOPS_CLOUD_RUN_INVOKER_AUTH"
ENV_AUDIENCE = "AGENTOPS_CLOUD_RUN_INVOKER_AUDIENCE"

_HEADER = "X-Serverless-Authorization"

# Module-level state for the global patch. Guarded by `_PATCH_LOCK` so
# concurrent construction of multiple AgentOps clients can't double-patch.
_PATCH_LOCK = threading.Lock()
_INSTALLED_AUDIENCE: Optional[str] = None


def _bool_env(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def is_enabled() -> bool:
    """True iff ``AGENTOPS_CLOUD_RUN_INVOKER_AUTH`` is truthy."""
    return _bool_env(ENV_ENABLED)


def install(host: Optional[str]) -> bool:
    """Install the OIDC interceptors for httpx (sync + async) and requests.

    Idempotent: subsequent calls with the same audience are no-ops;
    calls with a *different* audience log a warning and keep the
    first audience in place (an audience change at runtime is almost
    certainly a bug).

    Parameters
    ----------
    host:
        The AgentOps server host. Used as the OIDC audience unless
        ``AGENTOPS_CLOUD_RUN_INVOKER_AUDIENCE`` overrides it.

    Returns
    -------
    bool
        True when the patch is active after the call (either freshly
        installed or already installed for the same audience). False
        when disabled, mis-configured, or ``google-auth`` is missing.
    """
    global _INSTALLED_AUDIENCE

    if not is_enabled():
        return False

    audience_raw = os.environ.get(ENV_AUDIENCE) or host or ""
    audience = audience_raw.rstrip("/")
    if not audience:
        _log.warning(
            "%s is enabled but no audience could be resolved (neither "
            "AGENTOPS_HOST nor %s is set). Skipping OIDC auth — outbound "
            "calls will not include X-Serverless-Authorization.",
            ENV_ENABLED,
            ENV_AUDIENCE,
        )
        return False

    try:
        # Validated up-front so we fail loudly at install time rather than
        # silently per-request later.
        import google.auth.transport.requests  # noqa: F401
        import google.oauth2.id_token  # noqa: F401
    except ImportError:
        # `google-auth` is a hard dependency of agentops-sdk, so this
        # should only fire in unusual environments (vendored install
        # stripped the dep, conflicting pin downgraded it, etc.).
        _log.warning(
            "%s is enabled but `google-auth` could not be imported. "
            "Re-install agentops-sdk or add `google-auth>=2.29.0` "
            "explicitly. Falling back to no-op; expect 401s from "
            "Cloud Run if the AgentOps service is invoker-locked.",
            ENV_ENABLED,
        )
        return False

    with _PATCH_LOCK:
        if _INSTALLED_AUDIENCE is not None:
            if _INSTALLED_AUDIENCE != audience:
                _log.warning(
                    "AgentOps Cloud Run OIDC auth already installed with "
                    "audience %r; ignoring request to switch to %r. "
                    "Restart the process if you really need to change "
                    "audiences.",
                    _INSTALLED_AUDIENCE,
                    audience,
                )
            return True

        _install_httpx_patch(audience)
        _install_requests_patch(audience)
        _INSTALLED_AUDIENCE = audience
        _log.info(
            "AgentOps Cloud Run OIDC auth installed (audience=%s, "
            "header=%s). Outbound HTTP calls to this audience will "
            "carry a Google-signed OIDC token.",
            audience,
            _HEADER,
        )
        return True


# ---------------------------------------------------------------------------
# Token cache
# ---------------------------------------------------------------------------

# `google.auth.transport.requests.Request` is NOT thread-safe; guard the
# fetcher with a single lock. The token itself is cached by `google-auth`
# in-process until ~5 minutes before expiry, so contention is negligible.
_FETCH_LOCK = threading.Lock()
_GOOGLE_REQUEST: Optional[object] = None


def _fetch_token(audience: str) -> Optional[str]:
    """Mint (or return cached) OIDC token for the given audience.

    Returns ``None`` on transient failures — callers fall back to
    sending the request without the header, which lets the AgentOps
    app-layer auth still try and produces a clear 401 at the Cloud
    Run edge if the service is invoker-locked.
    """
    global _GOOGLE_REQUEST

    try:
        import google.auth.exceptions
        import google.auth.transport.requests
        import google.oauth2.id_token
    except ImportError:
        return None

    try:
        with _FETCH_LOCK:
            if _GOOGLE_REQUEST is None:
                _GOOGLE_REQUEST = google.auth.transport.requests.Request()
            return google.oauth2.id_token.fetch_id_token(
                _GOOGLE_REQUEST, audience
            )
    except (
        google.auth.exceptions.DefaultCredentialsError,
        google.auth.exceptions.TransportError,
        google.auth.exceptions.RefreshError,
    ) as e:
        # No GCP credentials in this environment (dev laptop, CI without
        # workload identity, etc.). Don't break the request — the app-layer
        # auth still applies. If Cloud Run is invoker-locked, the user
        # will see a 401 and know to fix their environment.
        _log.debug(
            "OIDC token fetch failed (%s); proceeding without %s",
            e.__class__.__name__,
            _HEADER,
        )
        return None
    except Exception:
        # Unknown failure mode (metadata server hang, etc.). Log loudly
        # but never block the request.
        _log.exception(
            "Unexpected error minting OIDC token; proceeding without %s",
            _HEADER,
        )
        return None


# ---------------------------------------------------------------------------
# Transport patches
# ---------------------------------------------------------------------------


def _host_matches(request_url, target_host: str) -> bool:
    """True iff the request's host matches the configured audience host.

    Both `httpx.URL` and `requests.PreparedRequest.url` expose the URL
    somewhat differently; this helper normalises.
    """
    try:
        if hasattr(request_url, "host"):
            # httpx.URL
            return request_url.host == target_host
        # requests: full string URL
        return urlparse(str(request_url)).hostname == target_host
    except Exception:
        return False


def _install_httpx_patch(audience: str) -> None:
    """Patch ``httpx.Client.send`` and ``httpx.AsyncClient.send`` in place."""
    try:
        import httpx
    except ImportError:  # pragma: no cover — httpx is a hard dep already
        return

    target_host = urlparse(audience).netloc

    # --- sync ---
    _orig_sync_send = httpx.Client.send

    def _patched_sync_send(self, request, **kwargs):
        if _host_matches(request.url, target_host) and _HEADER not in request.headers:
            token = _fetch_token(audience)
            if token is not None:
                request.headers[_HEADER] = f"Bearer {token}"
        return _orig_sync_send(self, request, **kwargs)

    httpx.Client.send = _patched_sync_send  # type: ignore[method-assign]

    # --- async ---
    _orig_async_send = httpx.AsyncClient.send

    async def _patched_async_send(self, request, **kwargs):
        if _host_matches(request.url, target_host) and _HEADER not in request.headers:
            token = _fetch_token(audience)
            if token is not None:
                request.headers[_HEADER] = f"Bearer {token}"
        return await _orig_async_send(self, request, **kwargs)

    httpx.AsyncClient.send = _patched_async_send  # type: ignore[method-assign]


def _install_requests_patch(audience: str) -> None:
    """Patch ``requests.Session.send`` in place.

    Module-level helpers like ``requests.get`` / ``requests.post`` create
    a transient ``Session`` for each call and then dispatch through
    ``Session.send`` — patching this one method covers every public
    surface of the ``requests`` library.
    """
    try:
        import requests
    except ImportError:  # pragma: no cover — requests is a hard dep already
        return

    target_host = urlparse(audience).netloc

    _orig_send = requests.Session.send

    def _patched_send(self, request, **kwargs):
        if _host_matches(request.url, target_host) and _HEADER not in request.headers:
            token = _fetch_token(audience)
            if token is not None:
                request.headers[_HEADER] = f"Bearer {token}"
        return _orig_send(self, request, **kwargs)

    requests.Session.send = _patched_send  # type: ignore[method-assign]
