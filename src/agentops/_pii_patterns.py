"""Customer-defined PII detectors for the SDK's local guardrails (#863).

A port of the validation half of ``packages/shared/src/guardrails/customPii.ts``
so ``Guardrails.from_config`` accepts the same ``pii`` block the server does::

    {"pii": {"entityTypes": ["EMAIL"],
             "customPatterns": [{"label": "EMPLOYEE_ID", "pattern": r"\\bEMP-\\d{6}\\b"}]}}

Python's ``re`` cannot be interrupted, so unlike the server (which runs patterns
under a hard timeout) the SDK relies on the static checks below, the caps, and
the fact that patterns served by the registry were already probed for slow
inputs when they were saved. Patterns you pass in code are your own; the same
checks apply so a typo does not become a hang. Keep it in step with the
TypeScript version: both are tested against the same positives and negatives.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

BUILTIN_PII_ENTITY_TYPES = ("EMAIL", "SSN", "PHONE", "CARD")
MAX_PATTERNS = 20
MAX_PATTERN_LENGTH = 200
# Longest text a custom detector scans; the rest is ignored.
MAX_INPUT_CHARS = 200_000
MAX_MATCHES = 100
_ALLOWED_FLAGS = "ims"
_LABEL_RE = re.compile(r"^[A-Z][A-Z0-9_]{1,31}$")
_VARIABLE_QUANT = re.compile(r"^(?:[*+]|\{\d*,\d*\})")


@dataclass(frozen=True)
class CustomPiiPattern:
    label: str
    pattern: str
    flags: str = ""

    def compile(self) -> "re.Pattern[str]":
        flags = 0
        if "i" in self.flags:
            flags |= re.IGNORECASE
        if "m" in self.flags:
            flags |= re.MULTILINE
        if "s" in self.flags:
            flags |= re.DOTALL
        return re.compile(self.pattern, flags)


@dataclass
class ResolvedPiiDetectors:
    entity_types: list[str] = field(default_factory=lambda: list(BUILTIN_PII_ENTITY_TYPES))
    patterns: list[CustomPiiPattern] = field(default_factory=list)
    # (label, error) for entries that were dropped. Never the pattern text.
    invalid: list[tuple[str, str]] = field(default_factory=list)


def _strip_escapes_and_classes(pattern: str) -> str:
    out: list[str] = []
    i = 0
    n = len(pattern)
    while i < n:
        ch = pattern[i]
        if ch == "\\":
            out.append("x")
            i += 2
        elif ch == "[":
            i += 1
            if i < n and pattern[i] == "^":
                i += 1
            if i < n and pattern[i] == "]":
                i += 1
            while i < n and pattern[i] != "]":
                i += 2 if pattern[i] == "\\" else 1
            i += 1
            out.append("x")
        else:
            out.append(ch)
            i += 1
    return "".join(out)


def _has_repeated_ambiguous_group(pattern: str) -> bool:
    """A group repeated without bound that holds a quantifier or an alternation."""
    stripped = _strip_escapes_and_classes(pattern)
    stack: list[dict[str, bool]] = []
    for i, ch in enumerate(stripped):
        if ch == "(":
            stack.append({"alternation": False, "quantified": False})
        elif ch == ")":
            if not stack:
                continue
            group = stack.pop()
            rest = stripped[i + 1 :]
            if _VARIABLE_QUANT.match(rest) and (group["alternation"] or group["quantified"]):
                return True
            if stack:
                parent = stack[-1]
                if _VARIABLE_QUANT.match(rest) or group["quantified"]:
                    parent["quantified"] = True
                if group["alternation"]:
                    parent["alternation"] = True
        elif ch == "|":
            if stack:
                stack[-1]["alternation"] = True
        elif ch in "*+":
            if stack:
                stack[-1]["quantified"] = True
        elif ch == "{" and _VARIABLE_QUANT.match(stripped[i:]):
            if stack:
                stack[-1]["quantified"] = True
    return False


def validate_custom_pii_pattern(pattern: str, flags: str = "") -> Optional[str]:
    """Return an error message, or ``None`` when the pattern passes the static checks."""
    if not pattern:
        return "Pattern must not be empty."
    if len(pattern) > MAX_PATTERN_LENGTH:
        return f"Pattern must be at most {MAX_PATTERN_LENGTH} characters."
    if len(set(flags)) != len(flags) or any(f not in _ALLOWED_FLAGS for f in flags):
        return f"Flags may only be {', '.join(_ALLOWED_FLAGS)}."
    if re.search(r"\\[1-9]|\\k<", pattern):
        return "Backreferences are not supported in PII patterns."
    if _has_repeated_ambiguous_group(pattern):
        return (
            "Pattern looks like it could backtrack catastrophically (a repeated group "
            "that contains a quantifier or an alternation, e.g. (a+)+ or (a|a)*). Simplify it."
        )
    try:
        compiled = CustomPiiPattern("X", pattern, flags).compile()
    except re.error as exc:
        return f"Invalid pattern: {exc}"
    if compiled.search(""):
        return "Pattern matches the empty string, so it would match everywhere."
    return None


def resolve_pii_detectors(raw: Any) -> ResolvedPiiDetectors:
    """The detectors a stored ``pii`` block asks for. Never raises: invalid entries are
    dropped and reported, and a missing or malformed block means "all built-ins"."""
    if not isinstance(raw, dict):
        return ResolvedPiiDetectors()

    entity_types = list(BUILTIN_PII_ENTITY_TYPES)
    if isinstance(raw.get("entityTypes"), list):
        wanted = set(raw["entityTypes"])
        entity_types = [t for t in BUILTIN_PII_ENTITY_TYPES if t in wanted]

    patterns: list[CustomPiiPattern] = []
    invalid: list[tuple[str, str]] = []
    custom = raw.get("customPatterns")
    if isinstance(custom, list):
        seen: set[str] = set()
        for index, entry in enumerate(custom[:MAX_PATTERNS]):
            label = str(entry.get("label", ""))[:32] if isinstance(entry, dict) else f"#{index}"
            if not isinstance(entry, dict):
                invalid.append((label, "Not an object"))
                continue
            flags = entry.get("flags") or ""
            pattern = entry.get("pattern")
            if not isinstance(pattern, str) or not isinstance(flags, str):
                invalid.append((label, "pattern and flags must be strings"))
            elif not _LABEL_RE.match(label):
                invalid.append((label, "Label must be 2-32 characters: A-Z, 0-9 and _, starting with a letter."))
            elif label in BUILTIN_PII_ENTITY_TYPES:
                invalid.append((label, "Label collides with a built-in entity type."))
            elif label in seen:
                invalid.append((label, "Duplicate label"))
            else:
                error = validate_custom_pii_pattern(pattern, flags)
                if error:
                    invalid.append((label, error))
                else:
                    seen.add(label)
                    patterns.append(CustomPiiPattern(label, pattern, flags))
        if len(custom) > MAX_PATTERNS:
            invalid.append(("*", f"More than {MAX_PATTERNS} patterns; the rest were ignored"))
    return ResolvedPiiDetectors(entity_types, patterns, invalid)
