from __future__ import annotations

import base64
import os
from typing import Optional


def _env(*keys: str) -> Optional[str]:
    for k in keys:
        v = os.getenv(k)
        if v is not None and str(v).strip() != "":
            return v
    return None


def get_host() -> Optional[str]:
    # Support both conventions used in this repo:
    # - AGENTOPS_HOST (common in docs/examples)
    # - AGENTOPS_BASE_URL (used by some scripts/tools)
    return _env("AGENTOPS_HOST", "AGENTOPS_HOST", "AGENTOPS_BASE_URL")


def get_public_key() -> Optional[str]:
    return _env("AGENTOPS_PUBLIC_KEY", "AGENTOPS_PUBLIC_KEY")


def get_secret_key() -> Optional[str]:
    return _env("AGENTOPS_SECRET_KEY", "AGENTOPS_SECRET_KEY")




def basic_auth_header(public_key: Optional[str], secret_key: Optional[str]) -> dict[str, str]:
    """``Authorization`` header for the AOS public REST API.

    The public API authenticates project keys with HTTP Basic ``pk:sk``.
    ``Bearer`` only resolves a *public* key and grants score-only access, so
    ``Bearer <secret key>`` is rejected with a 401 on every route.
    """
    token = base64.b64encode(f"{public_key or ''}:{secret_key or ''}".encode()).decode()
    return {"Authorization": f"Basic {token}"}
