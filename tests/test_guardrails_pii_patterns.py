import warnings

import pytest

from agentops._pii_patterns import resolve_pii_detectors, validate_custom_pii_pattern
from agentops.guardrails import Guardrails, GuardrailConfigWarning

EMP = {"label": "EMPLOYEE_ID", "pattern": r"\bEMP-\d{6}\b"}


@pytest.mark.parametrize(
    "pattern",
    [
        "(a+)+$",
        "(a|a)*b",
        "((a+)b)*c",
        r"(\w+\s?)*x",
        r"(\d+){2,}x",
        r"(a)\1",
        "a*",
        "a" * 201,
        "(",
    ],
)
def test_rejects_unsafe_patterns(pattern):
    assert validate_custom_pii_pattern(pattern) is not None


@pytest.mark.parametrize(
    "pattern",
    [
        r"\bEMP-\d{6}\b",
        r"(?:EMP|CON)-\d{4,8}",
        r"\d{3}-\d{2}-\d{4}",
        r"[A-Z]{2}\d{2}[A-Z0-9]{11,30}",
        r"(\d{3}-?){3}",
    ],
)
def test_accepts_specific_patterns(pattern):
    assert validate_custom_pii_pattern(pattern) is None


def test_flags_are_limited():
    assert validate_custom_pii_pattern(r"a\d", "i") is None
    assert validate_custom_pii_pattern(r"a\d", "g") is not None


def test_resolve_defaults_and_narrowing():
    for raw in (None, "x", [], 5):
        r = resolve_pii_detectors(raw)
        assert r.entity_types == ["EMAIL", "SSN", "PHONE", "CARD"]
        assert r.patterns == []
    assert resolve_pii_detectors({"entityTypes": ["CARD", "EMAIL"]}).entity_types == ["EMAIL", "CARD"]
    assert resolve_pii_detectors({"entityTypes": []}).entity_types == []


def test_resolve_drops_invalid_entries_without_raising():
    r = resolve_pii_detectors(
        {"customPatterns": [EMP, {"label": "EVIL", "pattern": "(a+)+$"}, {"label": "NOPE"}, None, dict(EMP)]}
    )
    assert [p.label for p in r.patterns] == ["EMPLOYEE_ID"]
    assert [label for label, _ in r.invalid] == ["EVIL", "NOPE", "#3", "EMPLOYEE_ID"]


def test_from_config_forwards_the_pii_block():
    guard = Guardrails.from_config(
        {
            "enabled": True,
            "pii": {"entityTypes": [], "customPatterns": [EMP]},
            "preInput": {"detectPii": True, "piiAction": "store_redacted"},
            "postOutput": {"detectPiiLeak": True},
        }
    )
    result = guard.check_input("my id is EMP-123456 ok")
    assert result.transformed_text == "my id is [REDACTED_EMPLOYEE_ID] ok"
    assert result.metadata["redacted"] == ["EMPLOYEE_ID"]

    out = guard.check_output("their id: EMP-654321")
    assert "pii_output" in out.flags
    assert out.transformed_text == "their id: [REDACTED_EMPLOYEE_ID]"


def test_from_config_narrows_builtin_types():
    guard = Guardrails.from_config(
        {
            "enabled": True,
            "pii": {"entityTypes": ["EMAIL"]},
            "preInput": {"detectPii": True, "piiAction": "store_redacted"},
        }
    )
    result = guard.check_input("mail a@b.com ssn 123-45-6789")
    assert result.transformed_text == "mail [REDACTED_EMAIL] ssn 123-45-6789"


def test_no_pii_block_behaves_as_before():
    guard = Guardrails.from_config(
        {"enabled": True, "preInput": {"detectPii": True, "piiAction": "store_redacted"}}
    )
    result = guard.check_input("mail a@b.com ssn 123-45-6789")
    assert result.transformed_text == "mail [REDACTED_EMAIL] ssn [REDACTED_SSN]"


def test_invalid_custom_pattern_warns_and_is_ignored():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        guard = Guardrails.from_config(
            {
                "enabled": True,
                "pii": {"customPatterns": [{"label": "EVIL", "pattern": "(a+)+$"}, EMP]},
                "preInput": {"detectPii": True, "piiAction": "store_redacted"},
            }
        )
    assert any(issubclass(w.category, GuardrailConfigWarning) for w in caught)
    # The healthy detector and the built-ins still work.
    result = guard.check_input("EMP-123456 a@b.com")
    assert result.transformed_text == "[REDACTED_EMPLOYEE_ID] [REDACTED_EMAIL]"


def test_text_past_the_scan_cap_is_not_truncated():
    guard = Guardrails.from_config(
        {
            "enabled": True,
            "pii": {"entityTypes": [], "customPatterns": [EMP]},
            "preInput": {"detectPii": True, "piiAction": "store_redacted"},
        }
    )
    text = "EMP-123456 " + "x" * 250_000
    result = guard.check_input(text)
    assert result.transformed_text is not None
    assert result.transformed_text.startswith("[REDACTED_EMPLOYEE_ID] ")
    assert len(result.transformed_text) == len(text) - len("EMP-123456") + len("[REDACTED_EMPLOYEE_ID]")
