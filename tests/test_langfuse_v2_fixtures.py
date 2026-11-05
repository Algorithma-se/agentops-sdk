"""The committed v2 ingestion fixtures still match what the SDK sends.

``tests/fixtures/langfuse_v2_ingestion/*.json`` are replayed by the server's
contract tests (web servertest + worker integration test). This test
re-records the scenarios with the installed ``langfuse`` client and fails
if the wire shape drifted (event types, their order, or the body fields),
so bumping the ``langfuse`` pin can't silently leave the server contract
testing stale payloads. Re-record with
``python scripts/record_langfuse_v2_fixtures.py`` when this fails on purpose.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "fixtures" / "langfuse_v2_ingestion"
RECORDER = ROOT / "scripts" / "record_langfuse_v2_fixtures.py"


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _shape(fixture: dict) -> list:
    return [
        (event["type"], sorted(event["body"].keys()))
        for request in fixture["requests"]
        for event in request["body"]["batch"]
    ]


def _events(fixture: dict) -> list:
    return [e for r in fixture["requests"] for e in r["body"]["batch"]]


@pytest.mark.parametrize("scenario", ["lowlevel", "decorator", "sdk_log"])
def test_committed_fixture_is_well_formed(scenario):
    fixture = _load(FIXTURES / f"{scenario}.json")
    assert fixture["recordedWith"]["langfuse"].startswith("2.")
    assert fixture["unhandledRequests"] == []
    for request in fixture["requests"]:
        assert request["path"] == "/api/public/ingestion"
        assert request["authScheme"] == "Basic"
        assert request["body"]["metadata"]["sdk_name"] == "python"


def test_decorator_fixture_carries_update_current_observation_fields():
    # Securitas relies on update_current_observation(model=, usage_details=)
    # reaching the generation; without it there is no model match or cost.
    events = _events(_load(FIXTURES / "decorator.json"))
    update = next(e for e in events if e["type"] == "generation-update")
    assert update["body"]["model"] == "gpt-4o-mini"
    assert update["body"]["usageDetails"] == {"input": 20, "output": 5}


def test_installed_langfuse_still_emits_the_recorded_shape(tmp_path):
    env = dict(os.environ, AGENTOPS_FIXTURE_OUT_DIR=str(tmp_path))
    for var in ("LANGFUSE_HOST", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"):
        env.pop(var, None)
    subprocess.run(
        [sys.executable, str(RECORDER), "--only", "lowlevel,decorator"],
        env=env,
        check=True,
        capture_output=True,
        timeout=120,
    )
    for scenario in ("lowlevel", "decorator"):
        fresh = _load(tmp_path / f"{scenario}.json")
        committed = _load(FIXTURES / f"{scenario}.json")
        assert _shape(fresh) == _shape(committed), (
            f"{scenario}: the installed langfuse client sends a different "
            "ingestion shape than the committed fixture. If intended, "
            "re-record with scripts/record_langfuse_v2_fixtures.py and "
            "re-run the server contract tests."
        )
