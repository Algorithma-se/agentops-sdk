"""Reading and writing replay files (``schema_version: 1``).

One definition of the format for every command: ``export`` writes a frozen
file, ``replay`` adds each entry's ``run`` and the provenance, ``assess``
adds resolutions, ``publish`` sends it. ``run.events`` holds the replay's
captured trace events (large, only for publishing); read runs through
``show`` or ``render``. ``run.repeats`` holds extra runs of the same turn
(``replay --repeat N``); they are never published as traces.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from .repo import ARTIFACT_DIR

SCHEMA_VERSION = 1

# One item of ``run.repeats``.
REPEAT_KEYS = ("status", "reply", "error", "started_at", "completed_at")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jsonable(value: Any) -> Any:
    """Fallback for provider objects that can end up in a run."""
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return str(value)


def read(path: Path) -> dict:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    if artifact.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"{path}: unsupported schema_version (expected {SCHEMA_VERSION}); "
            "export it again with this version of agentops evolve"
        )
    return artifact


def write(path: Path, artifact: dict) -> str:
    """Write the file only; return the JSON text."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(artifact, ensure_ascii=False, indent=2, default=_jsonable)
    path.write_text(text + "\n", encoding="utf-8")
    return text


def save(path: Path, artifact: dict) -> None:
    """Write the file and its HTML comparison next to it."""
    write_html(path, json.loads(write(path, artifact)))


def write_html(path: Path, artifact: dict) -> Path:
    """(Re)write the comparison page next to the file."""
    from .view import render_html

    html_path = path.with_suffix(".html")
    html_path.write_text(render_html(artifact), encoding="utf-8")
    return html_path


def ensure_new(path: Path) -> None:
    """Earlier evidence is never overwritten."""
    for existing in (path, path.with_suffix(".html")):
        if existing.exists():
            raise FileExistsError(f"{existing} exists; choose a new output path")


def default_replay_path(label: str, *, base: Path = ARTIFACT_DIR) -> Path:
    """``local/evolve/<YYYY-MM-DD>_<NN>-<label>/replay.json``, NN counting up."""
    day = date.today().isoformat()
    taken = [p.name for p in base.glob(f"{day}_*")] if base.exists() else []
    number = 1 + max((int(n[11:13]) for n in taken if n[11:13].isdigit()), default=0)
    return base / f"{day}_{number:02d}-{label}" / "replay.json"
