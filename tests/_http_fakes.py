"""Fake ``requests``-style transport shared by the client/shipper tests.

``agentops._http.get_session()`` is the single seam every SDK-owned HTTP
call goes through (``get_policy``, the governance-event shipper,
``Guardrails.from_agent``'s registry lookup); the ``fake_http`` fixture in
``conftest.py`` swaps it for a :class:`FakeSession` that records calls and
replays scripted responses.
"""

from __future__ import annotations

import collections
import json
import threading
from dataclasses import dataclass
from typing import Any, Optional


class FakeResponse:
    def __init__(self, status_code: int = 200, body: Any = None, text: Optional[str] = None):
        self.status_code = status_code
        self._body = body
        self.text = text if text is not None else (json.dumps(body) if body is not None else "")
        self.ok = 200 <= status_code < 300

    def json(self) -> Any:
        if self._body is None:
            raise ValueError("no JSON body")
        return self._body

    def raise_for_status(self) -> None:
        if not self.ok:
            raise RuntimeError(f"HTTP {self.status_code}")


@dataclass
class Call:
    method: str
    url: str
    params: Any
    json: Any
    headers: Any
    timeout: Any


OK_INGEST = {"accepted": 0, "duplicates": 0, "rejected": []}


class FakeSession:
    """Records every call; replays queued responses (or exceptions), then a default."""

    def __init__(self, default: Optional[FakeResponse] = None):
        self.calls: list[Call] = []
        self.responses: collections.deque[Any] = collections.deque()
        self.default = default or FakeResponse(200, dict(OK_INGEST))
        self._lock = threading.Lock()

    def queue(self, *responses: Any) -> "FakeSession":
        self.responses.extend(responses)
        return self

    def _handle(self, method: str, url: str, **kw: Any) -> Any:
        with self._lock:
            self.calls.append(
                Call(method, url, kw.get("params"), kw.get("json"), kw.get("headers"), kw.get("timeout"))
            )
            nxt = self.responses.popleft() if self.responses else self.default
        if isinstance(nxt, BaseException):
            raise nxt
        if callable(nxt) and not isinstance(nxt, FakeResponse):
            return nxt(method, url, **kw)
        return nxt

    def get(self, url: str, **kw: Any) -> Any:
        return self._handle("GET", url, **kw)

    def post(self, url: str, **kw: Any) -> Any:
        return self._handle("POST", url, **kw)

    def request(self, method: str, url: str, **kw: Any) -> Any:
        return self._handle(method, url, **kw)

    @property
    def posts(self) -> list[Call]:
        return [c for c in self.calls if c.method == "POST"]

    @property
    def gets(self) -> list[Call]:
        return [c for c in self.calls if c.method == "GET"]
