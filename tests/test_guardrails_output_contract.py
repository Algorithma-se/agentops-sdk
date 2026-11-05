"""Output-contract validation in the SDK, driven by the SAME fixture file as the
server-side suite (web/src/__tests__/outputContract.clienttest.ts)."""

import json
import warnings
from pathlib import Path

import pytest

from agentops.guardrails import Guardrails
from agentops.output_contract import (
    check_output_contract,
    parse_output_contract,
    resolve_output_contract,
)

_FIXTURE = (
    Path(__file__).resolve().parents[3]
    / "packages/shared/src/guardrails/__fixtures__/outputContractCases.json"
)
_CASES = json.loads(_FIXTURE.read_text(encoding="utf-8"))


def _sorted(errors):
    return sorted(errors, key=lambda e: (e["path"], e["keyword"]))


@pytest.mark.parametrize("case", _CASES["evaluate"], ids=lambda c: c["name"])
def test_shared_evaluation_cases(case):
    result = check_output_contract(case["contract"], case["text"])
    assert result["ok"] is case["expect"]["ok"]
    if not result["ok"]:
        assert result["code"] == case["expect"]["code"]
        assert _sorted(result["errors"]) == _sorted(case["expect"]["errors"])


@pytest.mark.parametrize("case", _CASES["save"], ids=lambda c: c["name"])
def test_shared_save_cases(case):
    contract, error = parse_output_contract(case["contract"])
    assert (error is None) is case["valid"]
    if error is not None and case.get("errorContains"):
        assert case["errorContains"] in error


def test_errors_never_contain_output_content():
    secret = "SECRET-VALUE-123"
    result = check_output_contract(
        {"kind": "json_schema", "schema": {"type": "object", "properties": {"a": {"type": "integer"}}}},
        json.dumps({"a": secret}),
    )
    assert result["ok"] is False
    assert secret not in json.dumps(result)


def test_resolve_legacy_and_explicit():
    assert resolve_output_contract({"mustBeJson": True}) == {"kind": "json"}
    assert resolve_output_contract({}) == {"kind": "none"}
    assert resolve_output_contract({"mustBeJson": True, "outputContract": {"kind": "none"}}) == {"kind": "none"}
    assert resolve_output_contract({"outputContract": {"kind": "bogus"}}) == {"kind": "none"}


_SCHEMA = {
    "type": "object",
    "required": ["answer"],
    "properties": {"answer": {"type": "string"}},
}


def test_check_output_flags_schema_violation_without_blocking():
    g = Guardrails(output_contract={"kind": "json_schema", "schema": _SCHEMA})
    r = g.check_output('{"answer": 5}')
    assert "schema_invalid" in r.flags
    assert r.metadata["output_contract_errors"] == [{"path": "/answer", "keyword": "type"}]
    assert r.passed  # flag-only by default, like the other output checks


def test_check_output_flags_invalid_json():
    g = Guardrails(output_contract={"kind": "json"})
    r = g.check_output("here you go")
    assert "json_invalid" in r.flags and r.passed


def test_check_output_can_block_when_asked():
    g = Guardrails(
        output_contract={"kind": "json_schema", "schema": _SCHEMA},
        output_contract_action="block",
    )
    r = g.check_output('{"answer": 5}')
    assert r.blocked and r.action == "block"


def test_no_contract_no_signal():
    r = Guardrails().check_output("plain prose")
    assert r.flags == []
    assert "output_contract_errors" not in r.metadata


def test_conforming_output_has_no_flags():
    g = Guardrails(output_contract={"kind": "json_schema", "schema": _SCHEMA})
    assert g.check_output('{"answer": "ok"}').flags == []


def test_from_config_reads_contract_and_legacy_must_be_json():
    g = Guardrails.from_config(
        {"enabled": True, "postOutput": {"outputContract": {"kind": "json_schema", "schema": _SCHEMA}}}
    )
    assert "schema_invalid" in g.check_output('{"answer": 1}').flags
    legacy = Guardrails.from_config({"enabled": True, "postOutput": {"mustBeJson": True}})
    assert "json_invalid" in legacy.check_output("nope").flags


def test_from_config_ignores_a_refused_contract_with_a_warning():
    cfg = {
        "enabled": True,
        "postOutput": {"outputContract": {"kind": "json_schema", "schema": {"$ref": "https://169.254.169.254/x"}}},
    }
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        g = Guardrails.from_config(cfg)
    assert any("outputContract" in str(w.message) for w in caught)
    assert g.check_output("not json").flags == []
