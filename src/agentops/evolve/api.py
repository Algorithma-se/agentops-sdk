"""The AOS Control public API calls Evolve Agent needs.

Synchronous, through the SDK's shared transport (``agentops._http``), with
HTTP Basic ``pk:sk`` from ``AGENTOPS_HOST`` / ``AGENTOPS_PUBLIC_KEY`` /
``AGENTOPS_SECRET_KEY``. Reading turns needs a key that is not ingest-only.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

from .. import _cloud_run_auth, _http
from .._env import get_host, get_public_key, get_secret_key

# The public API rejects a page size above this with HTTP 400.
MAX_PAGE_SIZE = 100
# A publication carries every replayed turn's trace; give it more time.
PUBLISH_TIMEOUT_S = 60.0

# The only keys ``--env-file`` reads.
ENV_FILE_KEYS = ("AGENTOPS_HOST", "AGENTOPS_PUBLIC_KEY", "AGENTOPS_SECRET_KEY")


def parse_env_file(path: Path) -> dict[str, str]:
    """``KEY=value`` lines of a dotenv-style file (``export``, quotes and
    `` #`` comments handled); nothing is printed."""
    values: dict[str, str] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        key, sep, value = line.strip().removeprefix("export ").partition("=")
        key, value = key.strip(), value.strip()
        if not sep or not key or key.startswith("#"):
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            values[key] = value[1:-1]
        else:
            values[key] = value.split(" #", 1)[0].strip()
    return values


def load_env_file(path: Path) -> list[str]:
    """Set ``ENV_FILE_KEYS`` from *path* into ``os.environ`` (the file wins;
    an empty value leaves the environment as is). Returns the keys set."""
    found = {k: v for k, v in parse_env_file(path).items() if k in ENV_FILE_KEYS and v}
    os.environ.update(found)
    return list(found)


class EvolveNotConfiguredError(RuntimeError):
    """``AGENTOPS_HOST`` / ``AGENTOPS_PUBLIC_KEY`` / ``AGENTOPS_SECRET_KEY`` unset."""

    def __init__(self, missing: list[str]):
        self.missing = missing
        super().__init__(f"AgentOps is not configured: {', '.join(missing)} unset")


class EvolveApiError(RuntimeError):
    """A failed or unreadable API response.

    The message names only the method, path and status (plus, for a rejected
    request, the server's problem list): query strings carry session ids and
    other response bodies can echo data.
    """

    def __init__(self, method: str, path: str, status_code: int, problem: str):
        self.status_code = status_code
        super().__init__(f"AgentOps {method} {path} {problem}")


class EvolveApi:
    """Minimal client for evolve turns, questions and publications."""

    def __init__(
        self,
        host: Optional[str] = None,
        public_key: Optional[str] = None,
        secret_key: Optional[str] = None,
    ) -> None:
        self.host = (host or get_host() or "").strip().rstrip("/")
        public_key = public_key or get_public_key()
        secret_key = secret_key or get_secret_key()
        missing = [
            name
            for name, value in (
                ("AGENTOPS_HOST", self.host),
                ("AGENTOPS_PUBLIC_KEY", public_key),
                ("AGENTOPS_SECRET_KEY", secret_key),
            )
            if not value
        ]
        if missing:
            raise EvolveNotConfiguredError(missing)
        self._auth = _http.basic_auth_header(str(public_key), str(secret_key))
        # No-op unless AGENTOPS_CLOUD_RUN_INVOKER_AUTH=true.
        _cloud_run_auth.install(self.host)

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[dict] = None,
        json: Any = None,
        timeout: float = _http.DEFAULT_TIMEOUT_S,
        detail_on: tuple[int, ...] = (400,),
    ) -> dict:
        kwargs: dict[str, Any] = {"headers": {"Authorization": self._auth}, "timeout": timeout}
        if params is not None:
            kwargs["params"] = params
        if json is not None:
            kwargs["json"] = json
        response = _http.get_session().request(method, f"{self.host}/api/public{path}", **kwargs)
        status = response.status_code
        if not 200 <= status < 300:
            problem = f"returned HTTP {status}"
            if status in detail_on:
                try:
                    detail: Any = response.json()
                except ValueError:
                    detail = (getattr(response, "text", "") or "")[:2000]
                problem += f": {detail}"
            raise EvolveApiError(method, path, status, problem)
        try:
            body = response.json()
        except ValueError:
            # ``from None``: a decode error quotes the start of the body.
            raise EvolveApiError(method, path, status, "returned a body that is not JSON") from None
        if not isinstance(body, dict):
            raise EvolveApiError(method, path, status, "returned JSON that is not an object")
        return body

    def project_id(self) -> str:
        """The id of the project this API key belongs to."""
        body = self._request("GET", "/projects")
        data = body.get("data") or []
        if not data or not isinstance(data[0], dict) or not data[0].get("id"):
            raise EvolveApiError("GET", "/projects", 200, "returned no project for this key")
        return str(data[0]["id"])

    def turns(self, filters: dict[str, Any], *, page: int = 1, limit: int = 50) -> dict:
        """One page of turns (``GET /evolve/turns``): ``{data, meta, like}``.
        *filters* are the endpoint's query parameters (camelCase, lists
        comma-separated, e.g. ``include="feedback,context"``); empty values
        are left out."""
        if not 1 <= limit <= MAX_PAGE_SIZE:
            raise ValueError(f"limit must be between 1 and {MAX_PAGE_SIZE}")
        if page < 1:
            raise ValueError("page must be 1 or more")
        params = {k: v for k, v in filters.items() if v not in (None, "")}
        return self._request(
            "GET", "/evolve/turns", params={**params, "page": page, "limit": limit}
        )

    def all_turns(self, filters: dict[str, Any]) -> list[dict]:
        """Every page of ``turns`` (the endpoint has no total: a short page ends it)."""
        found: list[dict] = []
        page = 1
        while True:
            data = self.turns(filters, page=page, limit=MAX_PAGE_SIZE).get("data") or []
            found += data
            if len(data) < MAX_PAGE_SIZE:
                return found
            page += 1

    def ask(
        self, *, feedback_id: str, question: str, author: str, revision: Optional[str] = None
    ) -> dict:
        """Ask the reviewers about a feedback item (``POST /evolve/questions``),
        about its current revision unless *revision* names one (an older one
        is a 409). An open question on the same item comes back with
        ``duplicate: true``."""
        body = {"feedback_id": feedback_id, "question": question, "author": author}
        if revision:
            body["revision"] = revision
        return self._request("POST", "/evolve/questions", json=body, detail_on=(400, 409))

    def withdraw(self, question_id: str) -> dict:
        """Take back a question (``DELETE /evolve/questions/{id}``)."""
        return self._request("DELETE", f"/evolve/questions/{question_id}")

    def media(self, media_id: str) -> bytes:
        """The bytes of a media file of this project (an image a trace holds)."""
        body = self._request("GET", f"/media/{media_id}")
        if not body.get("url"):
            raise EvolveApiError("GET", f"/media/{media_id}", 200, "returned no url")
        response = _http.get_session().get(body["url"], timeout=_http.DEFAULT_TIMEOUT_S)
        if not 200 <= response.status_code < 300:
            raise EvolveApiError(
                "GET", f"/media/{media_id}", response.status_code, "file download failed"
            )
        return response.content

    def publication(self, replay_id: str) -> Optional[dict]:
        """The publication of *replay_id* in this project (any status), or None."""
        body = self._request(
            "GET", "/evolve/publications", params={"replayId": replay_id, "limit": 1}
        )
        data = body.get("data") or []
        found = data[0] if data and isinstance(data[0], dict) else None
        return found if found and found.get("replay_id") == replay_id else None

    def publish(
        self,
        artifact: dict,
        *,
        prompt_versions: Optional[list[dict]] = None,
        published_by: Optional[str] = None,
    ) -> dict:
        """Publish a replay artifact (``POST /evolve/publications``).

        *prompt_versions* (``[{name, template_sha256, version}]``, the
        versions created for the replay's candidate prompts) and
        *published_by* are extra top-level fields, not part of the artifact
        or its digest. A rejected artifact (400/409) raises
        ``EvolveApiError`` with the server's problem list.
        """
        body = dict(artifact)
        if prompt_versions:
            body["prompt_versions"] = prompt_versions
        if published_by:
            body["published_by"] = published_by
        return self._request(
            "POST",
            "/evolve/publications",
            json=body,
            timeout=PUBLISH_TIMEOUT_S,
            detail_on=(400, 409),
        )

    def revert(self, replay_id: str) -> dict:
        """Withdraw a published replay (``POST /evolve/publications/{id}/revert``)."""
        return self._request(
            "POST", f"/evolve/publications/{replay_id}/revert", detail_on=(400, 404, 409)
        )
