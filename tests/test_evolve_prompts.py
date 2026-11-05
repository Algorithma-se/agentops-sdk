"""Local candidate prompt files and ``run.prompt``."""

from __future__ import annotations

import json

import pytest

from agentops.evolve.prompts import (
    LocalCandidate,
    PromptError,
    read_candidate,
    run_prompt,
    sha256_text,
)


def test_read_candidate(tmp_path):
    txt = tmp_path / "support.txt"
    with pytest.raises(PromptError, match="does not exist"):
        read_candidate("support", txt)
    txt.write_text("Be kind.")
    with pytest.raises(PromptError, match="prompt pull support"):
        read_candidate("support", txt)
    (tmp_path / "support.json").write_text(json.dumps({"name": "support", "base_version": 3}))
    assert read_candidate("support", txt) == LocalCandidate("support", "Be kind.", 3)


def test_run_prompt():
    candidates = {"support": LocalCandidate("support", "Be kind.", 3)}
    assert run_prompt(None, candidates) is None
    assert run_prompt({"name": "support", "version": None}, candidates) == {
        "name": "support",
        "version": None,
        "candidate": True,
        "base_version": 3,
        "template_sha256": sha256_text("Be kind."),
    }
    assert run_prompt({"name": "sales", "version": 4}, candidates) == {
        "name": "sales",
        "version": 4,
        "candidate": False,
        "base_version": None,
        "template_sha256": None,
    }
