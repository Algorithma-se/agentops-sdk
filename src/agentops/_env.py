from __future__ import annotations

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


