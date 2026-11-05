"""AOS prompt versions: ``prompt pull`` writes a candidate file, ``publish
--apply`` creates a version from it.

That version gets no label (AOS adds ``latest``): moving ``production`` or
any serving label to it is a person's decision in AOS Control.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from .prompts import DEFAULT_LABEL, PROMPT_DIR, PromptError, candidate_paths, sha256_text


class PromptStore:
    """The prompt calls Evolve Agent needs, through the AgentOps client.
    Reads bypass the SDK cache: a stale label would defeat the base-version
    check."""

    def __init__(self, client: Any = None) -> None:
        self._client = client

    def _sdk(self) -> Any:
        if self._client is None:
            from ..client import AgentOps

            self._client = AgentOps()
        if not getattr(self._client, "enabled", True):
            raise PromptError(
                "AgentOps SDK is disabled: set AGENTOPS_HOST, AGENTOPS_PUBLIC_KEY "
                "and AGENTOPS_SECRET_KEY"
            )
        return self._client

    def get(self, name: str, *, label: Optional[str] = None) -> tuple[int, str]:
        """``(version, text)`` of a text prompt by label."""
        try:
            prompt = self._sdk().get_prompt(name, label=label, type="text", cache_ttl_seconds=0)
        except PromptError:
            raise
        except Exception as exc:
            raise PromptError(f"could not fetch prompt {name!r}: {exc}") from None
        text, number = getattr(prompt, "prompt", None), getattr(prompt, "version", None)
        if not isinstance(text, str) or not isinstance(number, int):
            raise PromptError(f"prompt {name!r} is not a text prompt")
        return number, text

    def create(self, name: str, text: str, *, commit_message: str) -> int:
        """Create a new version without labels; return its version number."""
        try:
            prompt = self._sdk().create_prompt(
                name=name, prompt=text, labels=[], type="text", commit_message=commit_message
            )
        except PromptError:
            raise
        except Exception as exc:
            raise PromptError(f"could not create a version of {name!r}: {exc}") from None
        number = getattr(prompt, "version", None)
        if not isinstance(number, int):
            raise PromptError(f"creating a version of {name!r} returned no version")
        return number


def pull(
    store: PromptStore, name: str, *, label: str = DEFAULT_LABEL, output_dir: Path = PROMPT_DIR
) -> dict:
    """Write ``NAME.txt`` (the text *label* serves) and ``NAME.json``
    (``{name, label, base_version, sha256}``); return the metadata. Never
    overwrites."""
    text_path, meta_path = candidate_paths(name, output_dir)
    for path in (text_path, meta_path):
        if path.exists():
            raise FileExistsError(
                f"{path} exists; delete it (and its .txt/.json pair) to pull again"
            )
    version, text = store.get(name, label=label)
    text_path.parent.mkdir(parents=True, exist_ok=True)
    text_path.write_text(text, encoding="utf-8")
    meta = {"name": name, "label": label, "base_version": version, "sha256": sha256_text(text)}
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return {**meta, "path": str(text_path)}


def candidates(artifact: dict) -> list[dict]:
    """Distinct candidate prompts the runs used (``run.prompt`` with
    ``candidate: true``): ``{name, base_version, template_sha256}``."""
    found: dict[tuple, dict] = {}
    for entry in artifact.get("entries") or []:
        ref = (entry.get("run") or {}).get("prompt") or {}
        if ref.get("candidate"):
            key = (ref.get("name"), ref.get("template_sha256"))
            found.setdefault(
                key, {k: ref.get(k) for k in ("name", "base_version", "template_sha256")}
            )
    return list(found.values())


def _local(c: dict, prompt_dir: Path) -> tuple[Optional[str], dict, list[str]]:
    """``(text, pull metadata, problems)`` of the local candidate files."""
    text_path, meta_path = candidate_paths(c["name"], prompt_dir)
    text, meta, problems = None, {}, []
    if not text_path.is_file():
        problems.append(f"prompt {c['name']}: {text_path} is missing")
    else:
        text = text_path.read_text(encoding="utf-8")
        if sha256_text(text) != c["template_sha256"]:
            problems.append(
                f"prompt {c['name']}: {text_path} changed after the replay; replay again"
            )
    if not meta_path.is_file():
        problems.append(f"prompt {c['name']}: {meta_path} is missing")
    else:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("name") != c["name"] or meta.get("base_version") != c["base_version"]:
            problems.append(
                f"prompt {c['name']}: {meta_path} does not match the replay's base "
                f"version {c['base_version']}; replay again"
            )
    return text, meta, problems


def candidate_problems(
    artifact: dict, store: PromptStore, *, prompt_dir: Path = PROMPT_DIR
) -> list[str]:
    """Why the candidates cannot be pushed as they are: more than one text per
    prompt, the local file changed since the replay, or the pulled label no
    longer points to the base version."""
    found = candidates(artifact)
    names = [c["name"] for c in found]
    problems = [
        f"prompt {n}: the replay used more than one candidate text"
        for n in sorted({n for n in names if names.count(n) > 1})
    ]
    for c in found:
        _, meta, local = _local(c, prompt_dir)
        problems += local
        label = meta.get("label")
        if not isinstance(c["base_version"], int) or not isinstance(label, str):
            problems.append(f"prompt {c['name']}: no pulled label/base version is known")
            continue
        try:
            current, _ = store.get(c["name"], label=label)
        except PromptError as exc:
            problems.append(str(exc))
            continue
        if current != c["base_version"]:
            problems.append(
                f"prompt {c['name']}: label {label!r} now points to version {current}, not the "
                f"pulled base version {c['base_version']}; someone changed it in AOS Control. "
                "Pull again, re-apply the change and replay"
            )
    return problems


def push_candidates(
    artifact: dict, store: PromptStore, *, prompt_dir: Path = PROMPT_DIR
) -> tuple[list[dict], set[str]]:
    """Create a version per candidate; return ``prompt_versions`` and the
    names whose newest version was reused (it already has exactly this text,
    so a publish that failed after creating it can be retried)."""
    created, reused = [], set()
    for c in candidates(artifact):
        text, _, problems = _local(c, prompt_dir)
        if text is None or problems:
            raise PromptError("; ".join(problems) or f"prompt {c['name']}: no text")
        try:
            latest, latest_text = store.get(c["name"], label="latest")
        except PromptError:
            latest, latest_text = None, None
        if latest is not None and latest_text == text:
            version = latest
            reused.add(c["name"])
        else:
            version = store.create(
                c["name"], text, commit_message=f"Evolve Agent: {artifact.get('name')}"
            )
        created.append(
            {"name": c["name"], "template_sha256": c["template_sha256"], "version": version}
        )
    return created, reused
