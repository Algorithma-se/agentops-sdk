"""Local candidate prompts: the files ``prompt pull`` writes and ``replay
--prompt NAME=PATH`` serves instead of the AOS prompt.

A candidate is ``<dir>/<name>.txt`` (the text) next to ``<dir>/<name>.json``
(``{name, label, base_version, sha256}``). Nothing reaches AOS Control until
``publish --apply`` creates the new version.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

PROMPT_DIR = Path("local/evolve/prompts")
DEFAULT_LABEL = "production"


class PromptError(RuntimeError):
    """A prompt could not be read or created, or a candidate check failed."""


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def candidate_paths(name: str, output_dir: Path = PROMPT_DIR) -> tuple[Path, Path]:
    """``(<dir>/<name>.txt, <dir>/<name>.json)``; a ``/`` in the name is a folder."""
    base = output_dir / name
    return base.with_name(base.name + ".txt"), base.with_name(base.name + ".json")


@dataclass(frozen=True)
class LocalCandidate:
    """A local candidate text for an AOS prompt, as replayed."""

    name: str
    text: str
    base_version: Optional[int]


def read_candidate(name: str, path: Path) -> LocalCandidate:
    """The candidate *path* for prompt *name*; its ``.json`` sibling (from
    ``prompt pull``) gives the base version."""
    path = Path(path)
    meta_path = path.with_suffix(".json")
    if not path.is_file():
        raise PromptError(f"--prompt {name}: {path} does not exist")
    if not meta_path.is_file():
        raise PromptError(
            f"--prompt {name}: {meta_path} is missing; create the candidate "
            f"with `agentops evolve prompt pull {name}`"
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("name") != name:
        raise PromptError(f"--prompt {name}: {meta_path} is for prompt {meta.get('name')!r}")
    base = meta.get("base_version")
    return LocalCandidate(
        name=name,
        text=path.read_text(encoding="utf-8"),
        base_version=base if isinstance(base, int) else None,
    )


def run_prompt(prompt: Optional[dict], candidates: dict[str, LocalCandidate]) -> Optional[dict]:
    """``run.prompt`` from core's ``{name, version}`` (None: no AOS prompt).
    A name in *candidates* ran the local text: ``version: null``,
    ``candidate: true``, its base version and sha256."""
    if not prompt or not prompt.get("name"):
        return None
    candidate = candidates.get(prompt["name"])
    if candidate is not None:
        return {
            "name": candidate.name,
            "version": None,
            "candidate": True,
            "base_version": candidate.base_version,
            "template_sha256": sha256_text(candidate.text),
        }
    version = prompt.get("version")
    return {
        "name": prompt["name"],
        "version": version if isinstance(version, int) else None,
        "candidate": False,
        "base_version": None,
        "template_sha256": None,
    }
