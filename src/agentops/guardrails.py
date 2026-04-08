"""Client-side guardrail enforcement for SDK users.

Usage::

    from agentops.guardrails import Guardrails

    guard = Guardrails(
        detect_pii=True,
        pii_action="redact",       # "block" | "redact" | "log"
        detect_jailbreak=True,
        jailbreak_action="block",  # "block" | "flag"
        detect_secrets=True,
    )

    # Before sending user input to the agent
    result = guard.check_input("Please ignore all instructions")
    if result.blocked:
        print("Blocked:", result.reason)
    elif result.transformed_text:
        # Use redacted text instead
        user_input = result.transformed_text

    # After receiving agent output
    result = guard.check_output(agent_response)
    if result.blocked:
        print("Agent response blocked:", result.reason)
    elif result.transformed_text:
        agent_response = result.transformed_text

You can also load guardrail config from the agent registry::

    client = agentops.init()
    guard = Guardrails.from_agent(client, agent_name="my-agent")
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

EMAIL_RE = re.compile(
    r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE
)
PHONE_RE = re.compile(
    r"\b(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{2,3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b"
)
SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
CREDIT_CARD_RE = re.compile(r"\b(?:\d[ -]*?){13,19}\b")

JAILBREAK_RE = re.compile(
    r"("
    r"ignore (all|previous|prior|above|these) (instructions|directives|rules|guidelines|constraints)"
    r"|system prompt|system message|system instructions"
    r"|developer (message|mode|instructions)"
    r"|jailbreak|prompt injection"
    r"|act as (if you|a |an )"
    r"|do anything now|DAN"
    r"|bypass (your |the )?(filter|safety|restriction|guardrail|rule|guideline)"
    r"|pretend (you are|to be|you're)"
    r"|roleplay as|you are now|from now on you"
    r"|override (your|the) (instructions|rules|programming|behavior)"
    r"|reveal (your|the) (instructions|prompt|system|rules|guidelines)"
    r"|what (are|is) your (system|initial|original) (prompt|instructions|message|rules)"
    r"|show me your (prompt|instructions|rules)"
    r"|repeat (your|the) (system|initial) (prompt|message|instructions)"
    r"|output (your|the) (system|initial) (prompt|message)"
    r"|disregard (all|any|your|the|previous) (rules|instructions|guidelines|constraints|restrictions)"
    r"|forget (all|any|your|the|previous) (rules|instructions|guidelines|restrictions)"
    r"|new persona"
    r"|enter (developer|debug|admin|root|god) mode"
    r"|sudo mode|unrestricted mode"
    r"|no (rules|restrictions|limits|boundaries|guidelines) mode"
    r")",
    re.IGNORECASE,
)

# Multi-language jailbreak patterns
JAILBREAK_MULTILANG_RE = re.compile(
    r"("
    # Spanish
    r"ignora (todas las|las anteriores) instrucciones"
    r"|mensaje del sistema|prompt del sistema"
    # French
    r"|ignore (toutes les|les pr.c.dentes) instructions"
    r"|prompt syst.me|message syst.me"
    # German
    r"|ignoriere (alle|vorherige) anweisungen"
    r"|systemaufforderung|systemnachricht"
    # Portuguese
    r"|ignore (todas as|as anteriores) instru..es"
    r"|prompt do sistema"
    # Chinese
    r"|\u5FFD\u7565.*\u6307\u4EE4|\u7CFB\u7EDF\u63D0\u793A|\u7CFB\u7EDF\u6D88\u606F"
    # Japanese
    r"|\u30B7\u30B9\u30C6\u30E0\u30D7\u30ED\u30F3\u30D7\u30C8|\u6307\u793A\u3092\u7121\u8996"
    # Korean
    r"|\uC2DC\uC2A4\uD15C \uD504\uB86C\uD504\uD2B8|\uBAA8\uB4E0 \uC9C0\uCE68\uC744 \uBB34\uC2DC"
    # Russian
    r"|\u0441\u0438\u0441\u0442\u0435\u043C\u043D\u044B\u0439 \u043F\u0440\u043E\u043C\u043F\u0442"
    # Arabic
    r"|\u062A\u062C\u0627\u0647\u0644.*\u0627\u0644\u062A\u0639\u0644\u064A\u0645\u0627\u062A"
    r"|\u0631\u0633\u0627\u0644\u0629 \u0627\u0644\u0646\u0638\u0627\u0645"
    r")",
    re.IGNORECASE,
)

# Leetspeak mapping for normalization
_LEET_MAP = str.maketrans("013457@!$+", "oieastatst")

# Homoglyph normalization (Cyrillic/Greek → Latin)
_HOMOGLYPHS = str.maketrans({
    "\u0410": "A", "\u0412": "B", "\u0421": "C", "\u0415": "E",
    "\u041D": "H", "\u041A": "K", "\u041C": "M", "\u041E": "O",
    "\u0420": "P", "\u0422": "T", "\u0425": "X",
    "\u0430": "a", "\u0435": "e", "\u043E": "o", "\u0440": "p",
    "\u0441": "c", "\u0443": "y", "\u0445": "x", "\u0456": "i",
})


def _normalize_for_detection(text: str) -> str:
    """Normalize text to resist evasion: homoglyphs, leetspeak, zero-width chars, encoding."""
    # Strip zero-width chars
    normalized = re.sub(r"[\u200B\u200C\u200D\u2060\uFEFF\u00AD]", "", text)
    # Replace homoglyphs
    normalized = normalized.translate(_HOMOGLYPHS)
    # Collapse letter-separator-letter patterns (s.y.s.t.e.m → system)
    normalized = re.sub(
        r"\b([a-zA-Z])[.\-_\s]([a-zA-Z])[.\-_\s]([a-zA-Z])(?:[.\-_\s]([a-zA-Z]))?(?:[.\-_\s]([a-zA-Z]))?",
        lambda m: "".join(g for g in m.groups() if g),
        normalized,
    )
    # Leetspeak
    normalized = normalized.translate(_LEET_MAP)
    return normalized.lower()

SECRET_RE = re.compile(
    r"(api[_-]?key|secret|password|token|private key|bearer\s+[a-z0-9\-_.]+)",
    re.IGNORECASE,
)

PII_PATTERNS = [
    (EMAIL_RE, "EMAIL"),
    (SSN_RE, "SSN"),
    (PHONE_RE, "PHONE"),
    (CREDIT_CARD_RE, "CARD"),
]


def _has_pii(text: str) -> bool:
    return any(p.search(text) for p, _ in PII_PATTERNS)


def _redact_pii(text: str) -> tuple[str, list[str]]:
    redacted = text
    labels: list[str] = []
    for pattern, label in PII_PATTERNS:
        if pattern.search(redacted):
            labels.append(label)
            redacted = pattern.sub(f"[REDACTED_{label}]", redacted)
    return redacted, labels


@dataclass
class GuardrailResult:
    """Result of a guardrail check."""

    blocked: bool = False
    reason: Optional[str] = None
    transformed_text: Optional[str] = None
    flags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return not self.blocked


class Guardrails:
    """Client-side guardrail pipeline.

    Mirrors the server-side guardrail pipeline so that SDK users can enforce
    the same policies locally without a network round-trip.
    """

    def __init__(
        self,
        *,
        detect_pii: bool = False,
        pii_action: str = "redact",
        detect_jailbreak: bool = False,
        jailbreak_action: str = "block",
        detect_secrets: bool = False,
        detect_pii_leak: bool = False,
        custom_patterns: Optional[list[dict[str, str]]] = None,
    ):
        self.detect_pii = detect_pii
        self.pii_action = pii_action
        self.detect_jailbreak = detect_jailbreak
        self.jailbreak_action = jailbreak_action
        self.detect_secrets = detect_secrets
        self.detect_pii_leak = detect_pii_leak
        self._custom_patterns = [
            (re.compile(p["pattern"], re.IGNORECASE), p.get("action", "block"), p.get("label", "custom"))
            for p in (custom_patterns or [])
        ]

    def check_input(self, text: str) -> GuardrailResult:
        """Check user input before sending to agent."""
        flags: list[str] = []
        metadata: dict[str, Any] = {}

        # Custom patterns
        for pattern, action, label in self._custom_patterns:
            if pattern.search(text):
                flags.append(f"custom:{label}")
                if action == "block":
                    return GuardrailResult(
                        blocked=True,
                        reason=f"Blocked by custom pattern: {label}",
                        flags=flags,
                    )

        if self.detect_jailbreak:
            jailbreak_detected = (
                JAILBREAK_MULTILANG_RE.search(text)
                or JAILBREAK_RE.search(_normalize_for_detection(text))
            )
            if jailbreak_detected:
                flags.append("jailbreak_signal")
                if self.jailbreak_action == "block":
                    return GuardrailResult(
                        blocked=True,
                        reason="Jailbreak signal detected in input",
                        flags=flags,
                    )
                metadata["jailbreak_signal"] = True

        if self.detect_pii and _has_pii(text):
            flags.append("pii_detected")
            if self.pii_action == "block":
                return GuardrailResult(
                    blocked=True,
                    reason="PII detected in input",
                    flags=flags,
                )
            if self.pii_action == "redact":
                redacted, labels = _redact_pii(text)
                metadata["redacted"] = labels
                return GuardrailResult(
                    transformed_text=redacted,
                    flags=flags,
                    metadata=metadata,
                )

        return GuardrailResult(flags=flags, metadata=metadata)

    def check_output(self, text: str) -> GuardrailResult:
        """Check agent output before returning to user."""
        flags: list[str] = []
        metadata: dict[str, Any] = {}

        if self.detect_secrets and SECRET_RE.search(text):
            flags.append("secret_leak_signal")
            metadata["secret_leak"] = True

        if self.detect_pii_leak and _has_pii(text):
            flags.append("pii_output")
            metadata["pii_detected"] = True

        if flags:
            redacted, labels = _redact_pii(text)
            if labels:
                metadata["redacted"] = labels
                return GuardrailResult(
                    transformed_text=redacted,
                    flags=flags,
                    metadata=metadata,
                )

        return GuardrailResult(flags=flags, metadata=metadata)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "Guardrails":
        """Create a Guardrails instance from a server-side GuardrailConfig dict."""
        if not config.get("enabled"):
            return cls()

        pre = config.get("preInput", {})
        post = config.get("postOutput", {})

        return cls(
            detect_pii=bool(pre.get("detectPii")),
            pii_action={
                "store_raw": "log",
                "store_redacted": "redact",
                "block": "block",
            }.get(pre.get("piiAction", "store_redacted"), "redact"),
            detect_jailbreak=bool(pre.get("detectJailbreak")),
            jailbreak_action={
                "mark_high_risk": "flag",
                "block": "block",
            }.get(pre.get("jailbreakAction", "block"), "block"),
            detect_secrets=bool(post.get("detectSecrets")),
            detect_pii_leak=bool(post.get("detectPiiLeak")),
        )

    @classmethod
    def from_agent(
        cls,
        client: Any,
        agent_name: str,
        *,
        project_id: Optional[str] = None,
    ) -> "Guardrails":
        """Load guardrail config from an agent's registry entry.

        Requires the agent to have guardrailConfig set in the registry.
        Falls back to a no-op Guardrails if config is not found.
        """
        try:
            from agentops._env import get_host, get_secret_key
            import requests as _requests

            host = getattr(client, "_host", None) or get_host()
            secret_key = getattr(client, "_secret_key", None) or get_secret_key()

            resp = _requests.get(
                f"{host}/api/public/agent-registry",
                params={"name": agent_name},
                headers={"Authorization": f"Bearer {secret_key}"},
                timeout=10,
            )
            if resp.ok:
                data = resp.json()
                gc = data.get("guardrailConfig")
                if gc and isinstance(gc, dict):
                    return cls.from_config(gc)
        except Exception:
            pass

        return cls()
