"""What stops a replay file from being published from this checkout.

The server checks the format again; these checks need the local repo (HEAD,
a clean tree) or catch a bad file before prompt versions are created.
"""

from __future__ import annotations

import re

from .artifact import REPEAT_KEYS


def repeat_problems(run: dict) -> list[str]:
    """What is wrong with *run*'s ``repeats`` (absent is fine)."""
    repeats = run.get("repeats")
    if repeats is None:
        return []
    if not isinstance(repeats, list):
        return ["run.repeats is not a list"]
    found = []
    for n, item in enumerate(repeats, 1):
        if not isinstance(item, dict) or set(item) != set(REPEAT_KEYS):
            found.append(f"run.repeats[{n}] must be an object with {', '.join(REPEAT_KEYS)}")
        elif item.get("status") not in ("complete", "failed"):
            found.append(f"run.repeats[{n}] status must be complete or failed")
        elif any(item[k] is not None and not isinstance(item[k], str) for k in REPEAT_KEYS):
            found.append(f"run.repeats[{n}] values must be strings or null")
    return found


def project_problems(artifact: dict, project_id: str) -> list[str]:
    """The file must belong to the API key's project."""
    if artifact.get("project_id") == project_id:
        return []
    return [
        f"artifact project {artifact.get('project_id')} is not the API key's project {project_id}"
    ]


def problems(artifact: dict, *, head: str, repo_dirty: bool, project_id: str) -> list[str]:
    """Everything that stops this file from being published from here."""
    provenance = artifact.get("provenance") or {}
    commit = provenance.get("git_commit") or ""
    found = []
    if provenance.get("dirty") is not False:
        found.append("the replay ran on a dirty working tree")
    if repo_dirty:
        found.append("the working tree is dirty; commit or stash first")
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        found.append(f"replay commit {commit or '(none)'} is not a 40-hex git SHA")
    elif commit != head:
        found.append(f"replay commit {commit} is not HEAD {head}")
    found += project_problems(artifact, project_id)
    for entry in artifact.get("entries") or []:
        run = entry.get("run") or {}
        label = f"run {entry.get('run_id')}"
        if not run.get("status") or not run.get("completed_at"):
            found.append(f"{label}: not replayed to completion")
        elif run.get("status") == "complete" and not run.get("events"):
            found.append(f"{label}: no trace events were captured; replay again")
        found += [f"{label}: {p}" for p in repeat_problems(run)]
        frozen = {(f.get("feedback_id"), f.get("revision")) for f in entry.get("feedback") or []}
        for r in entry.get("resolutions") or []:
            if (r.get("feedback_id"), r.get("feedback_revision")) not in frozen:
                found.append(
                    f"{label}: assessment of {r.get('feedback_id')} names a feedback "
                    "revision that is not frozen in this entry"
                )
            if not commit or commit[:7] not in (r.get("changes") or ""):
                found.append(
                    f"{label}: assessment of {r.get('feedback_id')} does not name "
                    "the replay commit in its changes"
                )
    return found
