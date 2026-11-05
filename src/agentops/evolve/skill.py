"""Installing the evolve-agent skill (shipped as package data) into a repo.

The skill is linked, not copied: each target gets a symlink to the copy inside
the installed SDK, so upgrading the SDK updates the skill. The link points to
this machine's SDK install, so it is listed in the repository's local
``.git/info/exclude`` (never committed) and does not make the tree dirty.
Where symlinks are not allowed (e.g. Windows without developer mode), the
skill is copied instead.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

SKILL_SOURCE = Path(__file__).resolve().parent / "skill" / "evolve-agent"
DEFAULT_TARGETS = (Path(".claude/skills"), Path(".agents/skills"))


@dataclass(frozen=True)
class Installed:
    path: Path
    linked: bool
    # Tracked by git before (an earlier committed copy): now shows as changed.
    was_tracked: bool = False
    excluded: bool = False


def _git(*args: str) -> Optional[str]:
    result = subprocess.run(["git", *args], capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def _remove(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def _exclude(path: Path) -> bool:
    """List *path* in the repository's ``.git/info/exclude``; False outside git."""
    top = _git("rev-parse", "--show-toplevel")
    info = _git("rev-parse", "--git-path", "info/exclude")
    if top is None or info is None:
        return False
    try:
        # The link itself, not its target: resolve only the parent.
        relative = path.parent.resolve().joinpath(path.name).relative_to(
            Path(top).resolve()
        )
    except ValueError:  # target outside this repository
        return False
    pattern = "/" + relative.as_posix()
    exclude = Path(info)
    lines = exclude.read_text(encoding="utf-8").splitlines() if exclude.exists() else []
    if pattern not in lines:
        exclude.parent.mkdir(parents=True, exist_ok=True)
        lines.append(pattern)
        exclude.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return True


def install(targets: list[Path], *, copy: bool = False) -> list[Installed]:
    """Link (or with *copy* copy) the skill into each target directory,
    replacing an older link or copy."""
    installed = []
    for target in targets:
        destination = target / SKILL_SOURCE.name
        was_tracked = bool(_git("ls-files", "--", str(destination)))
        _remove(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        linked = False
        if not copy:
            try:
                destination.symlink_to(SKILL_SOURCE, target_is_directory=True)
                linked = True
            except OSError:
                pass
        if not linked:
            shutil.copytree(SKILL_SOURCE, destination)
        installed.append(
            Installed(
                path=destination,
                linked=linked,
                was_tracked=was_tracked,
                excluded=linked and _exclude(destination),
            )
        )
    return installed
