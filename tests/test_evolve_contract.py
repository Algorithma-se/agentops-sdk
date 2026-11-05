"""The SDK's Evolve Agent constants agree with the shared contract file
(``packages/shared/src/features/evolve/contract.json``); skipped where the
SDK is tested outside the monorepo."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentops.evolve import cli

REPO = Path(__file__).resolve().parents[3]
CONTRACT = REPO / "packages/shared/src/features/evolve/contract.json"


@pytest.fixture(scope="module")
def contract() -> dict:
    if not CONTRACT.is_file():
        pytest.skip(f"{CONTRACT} is absent (the SDK is tested outside the monorepo)")
    return json.loads(CONTRACT.read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    "key, python",
    [
        ("assessmentStatuses", cli.STATUSES),
        ("resolvedStatuses", cli.RESOLVED),
        ("feedbackStates", cli.FEEDBACK_STATES),
    ],
)
def test_constants_match_the_contract(contract, key, python):
    assert tuple(contract[key]) == python
