from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make the sibling helper modules (``_http_fakes``, ``test_governance``'s fake
# AGT) importable regardless of pytest's import mode.
_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

from _http_fakes import FakeSession  # noqa: E402


@pytest.fixture
def fake_http(monkeypatch):
    """Route every SDK-owned HTTP call through a recording :class:`FakeSession`."""
    from agentops import _http

    session = FakeSession()
    monkeypatch.setattr(_http, "_SESSION", session)
    yield session
