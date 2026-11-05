"""Git facts about the repository the CLI runs in (the current directory)."""

from __future__ import annotations

import functools
import hashlib
import subprocess
from pathlib import Path
from typing import Optional

# Evolve files are written here by default; untracked files below it never
# make the tree dirty.
ARTIFACT_DIR = Path("local/evolve")


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True).stdout


def head() -> str:
    return _git("rev-parse", "HEAD").strip()


def user_name() -> Optional[str]:
    """``git config user.name`` (the developer), or None when unset."""
    result = subprocess.run(
        ["git", "config", "user.name"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() or None


def _untracked() -> list[str]:
    prefix = f"{ARTIFACT_DIR.as_posix()}/"
    return [
        path
        for path in _git("ls-files", "--others", "--exclude-standard").splitlines()
        if path and not path.startswith(prefix)
    ]


def is_dirty() -> bool:
    """Tracked changes, or untracked files outside the artifact directory."""
    return bool(_git("diff", "HEAD", "--name-only").strip() or _untracked())


def snapshot() -> dict:
    """A replay's provenance: commit, dirtiness, diff and untracked file hashes."""
    untracked = _untracked()
    diff = _git("diff", "HEAD")
    return {
        "git_commit": head(),
        "dirty": bool(diff.strip() or untracked),
        "diff": diff,
        "untracked_hashes": {
            path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
            for path in untracked
            if Path(path).is_file()
        },
    }


@functools.lru_cache(maxsize=None)
def is_ancestor(commit: str) -> bool:
    """Whether *commit* is contained in ``HEAD`` (cached per command)."""
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        capture_output=True,
        check=False,
    )
    return result.returncode == 0
