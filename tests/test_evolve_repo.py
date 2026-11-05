"""Git facts (``repo``) and replay paths, in a temporary git repository."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pytest

from agentops.evolve import artifact as files
from agentops.evolve import repo, skill

GIT = ["git", "-c", "user.email=t@t", "-c", "user.name=t"]


@pytest.fixture
def git_repo(tmp_path, monkeypatch) -> Path:
    monkeypatch.chdir(tmp_path)
    subprocess.run(["git", "init", "-q"], check=True)
    (tmp_path / "a.txt").write_text("a")
    subprocess.run([*GIT, "add", "a.txt"], check=True)
    subprocess.run([*GIT, "commit", "-qm", "init"], check=True)
    return tmp_path


def test_snapshot_records_dirt_diff_and_untracked_hashes(git_repo):
    clean = repo.snapshot()
    assert clean["git_commit"] == repo.head() and len(clean["git_commit"]) == 40
    repo.is_ancestor.cache_clear()
    assert repo.is_ancestor(clean["git_commit"]) and not repo.is_ancestor("b" * 40)
    assert clean["dirty"] is False and clean["diff"] == "" and clean["untracked_hashes"] == {}
    # Files under local/evolve never make the tree dirty.
    (git_repo / "local/evolve/run").mkdir(parents=True)
    (git_repo / "local/evolve/run/replay.json").write_text("{}")
    assert repo.snapshot()["dirty"] is False and not repo.is_dirty()

    (git_repo / "a.txt").write_text("b")
    (git_repo / "new.txt").write_text("new")
    dirty = repo.snapshot()
    assert dirty["dirty"] is True and repo.is_dirty()
    assert "a.txt" in dirty["diff"] and "+b" in dirty["diff"]
    assert dirty["untracked_hashes"] == {"new.txt": hashlib.sha256(b"new").hexdigest()}


def test_default_replay_path_counts_up(tmp_path):
    first = files.default_replay_path("x", base=tmp_path)
    assert first.name == "replay.json" and first.parent.name.endswith("_01-x")
    first.parent.mkdir()
    assert files.default_replay_path("y", base=tmp_path).parent.name.endswith("_02-y")


# --- install-skill


def test_install_links_the_skill_and_keeps_the_tree_clean(git_repo, capsys):
    from agentops.evolve.cli import main

    assert main(["install-skill"]) == 0
    for target in skill.DEFAULT_TARGETS:
        link = git_repo / target / "evolve-agent"
        assert link.is_symlink() and link.resolve() == skill.SKILL_SOURCE
    assert not repo.is_dirty()
    assert "Linked:" in capsys.readouterr().out
    assert main(["install-skill"]) == 0  # again: no duplicate exclude lines
    exclude = (git_repo / ".git/info/exclude").read_text().splitlines()
    assert exclude.count("/.claude/skills/evolve-agent") == 1
    assert exclude.count("/.agents/skills/evolve-agent") == 1


def test_install_replaces_a_committed_copy_and_says_how_to_untrack_it(git_repo, capsys):
    from agentops.evolve.cli import main

    old = git_repo / ".claude/skills/evolve-agent"
    old.mkdir(parents=True)
    (old / "SKILL.md").write_text("old copy")
    subprocess.run([*GIT, "add", "."], check=True)
    subprocess.run([*GIT, "commit", "-qm", "skill copy"], check=True)
    assert main(["install-skill", "--target", ".claude/skills"]) == 0
    assert old.is_symlink()
    assert "git rm -r --cached" in capsys.readouterr().out


def test_install_copy_copies(git_repo):
    (item,) = skill.install([Path("skills")], copy=True)
    assert not item.linked and not item.path.is_symlink()
    assert (item.path / "SKILL.md").is_file()
