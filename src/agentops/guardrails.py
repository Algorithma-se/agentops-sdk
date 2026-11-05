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

You can also load guardrail config from the Agent Registry. It warns (or,
with ``strict=True``, raises) instead of silently degrading when the config
is unavailable; see :meth:`Guardrails.from_agent` for server support::

    client = agentops.init()
    guard = Guardrails.from_agent(client, agent_name="my-agent")

For structured PII (SSN, phone, credit card) that needs to survive a
round trip through your own LLM call intact — not just redacted —
``obfuscate``/``deobfuscate`` locally encrypt it with NIST FF3-1
format-preserving encryption (``pip install agentops-sdk[pii]``)::

    guard = Guardrails()
    obfuscated = guard.obfuscate(user_input, context_id=trace_id)
    response = call_your_own_llm(obfuscated)
    restored = guard.deobfuscate(response, context_id=trace_id)
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import warnings
from dataclasses import dataclass, field
from typing import Any, Optional

from agentops._pii_patterns import (
    MAX_INPUT_CHARS,
    CustomPiiPattern,
    ResolvedPiiDetectors,
    resolve_pii_detectors,
)
from .output_contract import check_output_contract, parse_output_contract, resolve_output_contract
from agentops._secrets import find_secrets, redact_secrets

_log = logging.getLogger("agentops")

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

PII_PATTERNS = [
    (EMAIL_RE, "EMAIL"),
    (SSN_RE, "SSN"),
    (PHONE_RE, "PHONE"),
    (CREDIT_CARD_RE, "CARD"),
]


def _has_pii(text: str) -> bool:
    return any(p.search(text) for p, _ in PII_PATTERNS)


# Structured-PII patterns usable with FF3-1 (excludes EMAIL — not a
# fixed-format numeric domain FPE can encrypt in place). Reuses
# SSN_RE/PHONE_RE/CREDIT_CARD_RE directly: Python's `finditer` finds all
# non-overlapping matches inherently, unlike JS regex which needs an
# explicit global flag — so no separate "span" pattern copies are needed
# here the way packages/shared/src/guardrails/fpe.ts requires on the TS
# side. Entity-type labels ("ssn"/"phone"/"creditCard") match the TS
# side's naming for consistency across the two independent
# implementations, even though no code is shared between them.
_FPE_PATTERNS = [
    (SSN_RE, "ssn"),
    (PHONE_RE, "phone"),
    (CREDIT_CARD_RE, "creditCard"),
]

# FF3-1 requires the radix-10 message space to be at least 1,000,000
# (>= 6 digits); the `ff3` package raises ValueError below that, which
# _fpe_transform pre-checks so a too-short span fails open instead of
# raising mid-batch.
_FF3_MIN_DIGITS = 6


def _find_pii_spans(text: str) -> list[tuple[int, int, str, str]]:
    """Find non-overlapping structured-PII spans as (start, end,
    entity_type, matched_text) tuples. Overlaps (e.g. a digit run that
    satisfies two patterns) are resolved by keeping the longer match at
    a given position — the same deterministic tie-break as fpe.ts's
    ``findPiiSpans`` on the TS side.
    """
    candidates: list[tuple[int, int, str, str]] = []
    for pattern, entity_type in _FPE_PATTERNS:
        for match in pattern.finditer(text):
            candidates.append(
                (match.start(), match.end(), entity_type, match.group(0))
            )

    candidates.sort(key=lambda c: (c[0], -(c[1] - c[0])))

    resolved: list[tuple[int, int, str, str]] = []
    for candidate in candidates:
        if resolved:
            last = resolved[-1]
            if candidate[0] < last[1]:
                if (candidate[1] - candidate[0]) > (last[1] - last[0]):
                    resolved[-1] = candidate
                continue
        resolved.append(candidate)
    return resolved


def _derive_fpe_tweak(context_id: str, entity_type: str) -> str:
    """sha256(contextId + ":" + entityType) truncated to 14 hex chars
    (FF3-1's required 7-byte tweak) — identical scheme to fpe.ts's
    ``deriveTweak`` on the TS/ingestion side.
    """
    digest = hashlib.sha256(f"{context_id}:{entity_type}".encode()).hexdigest()
    return digest[:14].upper()


def _redact_pii(text: str) -> tuple[str, list[str]]:
    redacted = text
    labels: list[str] = []
    for pattern, label in PII_PATTERNS:
        if pattern.search(redacted):
            labels.append(label)
            redacted = pattern.sub(f"[REDACTED_{label}]", redacted)
    return redacted, labels


_BUILTIN_PII = {label: pattern for pattern, label in PII_PATTERNS}


@dataclass
class GuardrailResult:
    """Result of a guardrail check (text stages and tool-call policy alike).

    ``reason`` is the primary human-readable explanation; ``reasons`` holds
    machine-readable codes (``"policy_denied:<rule>"`` for tool-call
    denials). ``action`` mirrors the server's vocabulary: ``"block"``,
    ``"transform"`` or ``"allow"``.
    """

    blocked: bool = False
    reason: Optional[str] = None
    transformed_text: Optional[str] = None
    flags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.blocked

    @property
    def action(self) -> str:
        if self.blocked:
            return "block"
        if self.transformed_text is not None:
            return "transform"
        return "allow"

    @property
    def decision(self) -> Any:
        """The ``agentops.governance.Decision`` behind a tool-call result (or ``None``)."""
        return self.metadata.get("decision")


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
        output_contract: Optional[dict[str, Any]] = None,
        output_contract_action: str = "flag",
        custom_patterns: Optional[list[dict[str, str]]] = None,
        pii: Optional[dict[str, Any]] = None,
        agent_name: Optional[str] = None,
        tool_policy: Any = None,
        tool_policy_version: Optional[Any] = None,
        shipper: Any = None,
    ):
        self.detect_pii = detect_pii
        self.pii_action = pii_action
        self.detect_jailbreak = detect_jailbreak
        self.jailbreak_action = jailbreak_action
        self.detect_secrets = detect_secrets
        self.detect_pii_leak = detect_pii_leak
        # Output contract (see agentops.output_contract): ``None``/``{"kind":
        # "none"}`` = no check and no signal. An invalid contract is rejected
        # here, never evaluated. ``output_contract_action`` is "flag"
        # (default, like the other output checks) or "block".
        if output_contract is not None:
            parsed, err = parse_output_contract(output_contract)
            if parsed is None:
                raise ValueError(f"invalid output_contract: {err}")
            output_contract = parsed
        self.output_contract = output_contract
        self.output_contract_action = output_contract_action
        self._custom_patterns = [
            (re.compile(p["pattern"], re.IGNORECASE), p.get("action", "block"), p.get("label", "custom"))
            for p in (custom_patterns or [])
        ]
        # Which PII to look for (#863): the server's ``pii`` block, i.e. a subset
        # of the built-in types and/or customer-defined regex detectors.
        self._pii: ResolvedPiiDetectors = resolve_pii_detectors(pii)
        for label, error in self._pii.invalid:
            warnings.warn(
                f"Ignoring custom PII detector {label}: {error}",
                GuardrailConfigWarning,
                stacklevel=3,
            )
        self._pii_compiled = [(p, p.compile()) for p in self._pii.patterns]
        # Tool-call policy (AGT) — see pre_tool_call()/govern(). ``tool_policy``
        # is an AgentPolicy from client.get_policy(), or anything
        # agentops.governance accepts (YAML text, file path, agentmesh Policy).
        self.agent_name = agent_name
        self.policy: Any = None  # the AgentPolicy from the registry, when loaded that way
        self._agt_policy: Any = None
        self._agt_policy_version: Optional[Any] = None
        self._shipper = shipper
        if tool_policy is not None:
            self.set_tool_policy(tool_policy, version=tool_policy_version)

    def _custom_pii_labels(self, text: str) -> list[str]:
        scan = text[:MAX_INPUT_CHARS]
        return [p.label for p, compiled in self._pii_compiled if compiled.search(scan)]

    def _has_pii_text(self, text: str) -> bool:
        if any(_BUILTIN_PII[t].search(text) for t in self._pii.entity_types):
            return True
        return bool(self._pii_compiled) and bool(self._custom_pii_labels(text))

    def _redact_pii_text(self, text: str) -> tuple[str, list[str]]:
        redacted = text
        labels: list[str] = []
        for entity_type in self._pii.entity_types:
            pattern = _BUILTIN_PII[entity_type]
            if pattern.search(redacted):
                labels.append(entity_type)
                redacted = pattern.sub(f"[REDACTED_{entity_type}]", redacted)
        # Custom detectors only look at the first MAX_INPUT_CHARS characters (the
        # rest is passed through, not truncated), mirroring the server.
        head, tail = redacted[:MAX_INPUT_CHARS], redacted[MAX_INPUT_CHARS:]
        for custom, compiled in self._pii_compiled:
            if compiled.search(head):
                labels.append(custom.label)
                head = compiled.sub(f"[REDACTED_{custom.label}]", head)
        return head + tail, labels

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

        if self.detect_pii and self._has_pii_text(text):
            flags.append("pii_detected")
            if self.pii_action == "block":
                return GuardrailResult(
                    blocked=True,
                    reason="PII detected in input",
                    flags=flags,
                )
            if self.pii_action == "redact":
                redacted, labels = self._redact_pii_text(text)
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

        # Secret-pattern detection (#867): real credential shapes, not the word
        # "password". Findings hold kinds only, never the secret.
        secrets = find_secrets(text) if self.detect_secrets else []
        if secrets:
            flags.append("secret_leak_signal")
            metadata["secret_leak"] = True
            metadata["secret_kinds"] = sorted({f.kind for f in secrets})

        if self.detect_pii_leak and self._has_pii_text(text):
            flags.append("pii_output")
            metadata["pii_detected"] = True

        contract_blocked = False
        if self.output_contract is not None:
            check = check_output_contract(self.output_contract, text)
            if not check["ok"]:
                flags.append(check["code"])
                metadata["output_contract_errors"] = check["errors"]
                contract_blocked = self.output_contract_action == "block"

        if contract_blocked:
            return GuardrailResult(
                blocked=True,
                reason="Output does not satisfy the configured output contract",
                flags=flags,
                metadata=metadata,
            )

        if flags:
            redacted, labels = self._redact_pii_text(text)
            if secrets:
                redacted, secret_kinds = redact_secrets(redacted)
                if secret_kinds:
                    labels = [*labels, "SECRET"]
            if labels:
                metadata["redacted"] = labels
                return GuardrailResult(
                    transformed_text=redacted,
                    flags=flags,
                    metadata=metadata,
                )

        return GuardrailResult(flags=flags, metadata=metadata)

    def _fpe_transform(
        self,
        text: str,
        context_id: str,
        key: Optional[str],
        direction: str,
    ) -> str:
        resolved_key = key or os.environ.get("AGENTOPS_LOCAL_FPE_KEY")
        if not resolved_key:
            return text

        # 256-bit key as 64 hex chars, matching the ingestion-side schema
        # (packages/shared/src/env.ts). Checked up front rather than left
        # to the per-span try/except below: FF3Cipher validates the key
        # lazily on encrypt/decrypt, not at construction, so a malformed
        # key would otherwise be swallowed by that except and silently
        # leave every span unprotected with no signal to the caller.
        if not re.fullmatch(r"[0-9a-fA-F]{64}", resolved_key):
            warnings.warn(
                "AGENTOPS_LOCAL_FPE_KEY is not a valid 256-bit hex key "
                "(expected 64 hex characters) — obfuscate()/deobfuscate() "
                "will leave PII unprotected for this call.",
                stacklevel=3,
            )
            return text

        try:
            from ff3 import FF3Cipher
        except ImportError:
            return text

        spans = _find_pii_spans(text)
        if not spans:
            return text

        result = text
        # Right-to-left by start offset so earlier offsets stay valid as
        # the string is rebuilt — mirrors fpe.ts's transformSpans (belt-
        # and-suspenders: FF3 never changes a span's length, but this
        # keeps the code obviously correct without relying on that).
        for start, end, entity_type, matched in sorted(
            spans, key=lambda s: -s[0]
        ):
            digits = "".join(ch for ch in matched if ch.isdigit())
            if len(digits) < _FF3_MIN_DIGITS:
                continue  # fail open: leave this span untouched
            tweak = _derive_fpe_tweak(context_id, entity_type)
            cipher = FF3Cipher(resolved_key, tweak)
            try:
                transformed_digits = (
                    cipher.encrypt(digits)
                    if direction == "encrypt"
                    else cipher.decrypt(digits)
                )
            except Exception:
                continue  # fail open: leave this span untouched
            digit_iter = iter(transformed_digits)
            rebuilt = "".join(
                next(digit_iter) if ch.isdigit() else ch for ch in matched
            )
            result = result[:start] + rebuilt + result[end:]

        return result

    def obfuscate(
        self,
        text: str,
        context_id: str,
        *,
        key: Optional[str] = None,
    ) -> str:
        """Locally encrypt structured PII (SSN, phone, credit card) via
        NIST FF3-1 format-preserving encryption, so the LLM you call
        next sees a syntactically valid but fake value instead of the
        real one. Runs entirely in this process — no network call, no
        data leaves your machine.

        Requires the ``pii`` extra (``pip install agentops-sdk[pii]``)
        and an FF3 key, passed via ``key`` or read from the
        ``AGENTOPS_LOCAL_FPE_KEY`` env var (256-bit hex). Fails open
        (returns ``text`` unmodified) if ``ff3`` isn't installed, no key
        is configured, or encryption otherwise fails.

        ``context_id`` seeds the encryption tweak — pass the SAME value
        to ``deobfuscate()`` afterward to restore the original values
        from the LLM's response. Different `context_id`s produce
        different ciphertext for the same input.
        """
        return self._fpe_transform(text, context_id, key, "encrypt")

    def deobfuscate(
        self,
        text: str,
        context_id: str,
        *,
        key: Optional[str] = None,
    ) -> str:
        """Reverse ``obfuscate()``: restores the original PII values an
        LLM's response echoed back. ``context_id`` must match the value
        passed to the ``obfuscate()`` call that produced the input text.
        Fails open under the same conditions as ``obfuscate()``.
        """
        return self._fpe_transform(text, context_id, key, "decrypt")

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
            pii=config.get("pii"),
            output_contract=_contract_from_post_output(post),
        )

    @classmethod
    def from_agent(
        cls,
        client: Any,
        agent_name: Any = None,
        *,
        project_id: Optional[str] = None,
        strict: bool = False,
        timeout: float = 10.0,
        policy_label: str = "production",
        load_policy: bool = True,
        shipper: Any = None,
    ) -> "Guardrails":
        """Load guardrail config and the tool-call policy from an agent's
        Agent Registry entry.

        Accepts both ``from_agent(client, "my-agent")`` and
        ``from_agent("my-agent", client)``.

        Text guardrails: looks the agent up by exact name through the public
        Agent Registry API (``GET /api/public/agents?name=...``, then
        ``GET /api/public/agents/{id}``), authenticating with the client's
        public/secret key pair (HTTP Basic), and maps the entry's
        ``guardrailConfig`` through :meth:`from_config`.

        Server support: the public agent resource does not include
        ``guardrailConfig`` today (it is a deliberately stable subset of
        the registry entry), so on current servers this finds the agent and
        then reports that no config is exposed. To enforce an agent's
        registry guardrails today, call
        ``client.check_guardrails(text, agent_name=...)``, which evaluates
        the config server-side. This method picks the config up without a
        code change once the server returns it on either endpoint.

        The config lookup never degrades silently. When the config cannot be
        loaded (the server predates the Agent Registry API and answers 404,
        the key is rejected, no agent has that exact name, the entry exposes
        no guardrail config, or the request fails) it emits a
        :class:`GuardrailConfigWarning`, also logged on the ``agentops``
        logger, that names the agent and the host, and continues with a
        no-op text pipeline. Pass ``strict=True`` to raise
        :class:`GuardrailConfigUnavailableError` instead.

        Tool-call policy: comes from
        ``client.get_policy(agent_name, label=policy_label)`` (the registry's
        AGT manifest, or the translation of a legacy
        ``preTool.toolAllowlist``). A missing policy (``PolicyNotFound``)
        leaves :meth:`pre_tool_call` / :meth:`govern` raising
        ``RuntimeError`` until a policy is provided. ``load_policy=False``
        skips it. ``shipper`` is the ``GovernanceEventShipper`` used for
        decisions (``None`` means the default one, see
        ``agentops.governance.set_default_shipper``).

        ``project_id`` is accepted for backwards compatibility and ignored:
        API keys are project-scoped, so the lookup always runs in the key's
        project.
        """
        del project_id  # kept for signature compatibility; see docstring
        if isinstance(client, str) and not isinstance(agent_name, str):
            client, agent_name = agent_name, client
        if not isinstance(agent_name, str) or not agent_name:
            raise TypeError("Guardrails.from_agent() needs the agent name as a string")

        guard = cls._config_from_registry(
            client, agent_name, strict=strict, timeout=timeout
        )
        guard.agent_name = agent_name
        guard._shipper = shipper
        if load_policy:
            guard._load_registry_policy(client, agent_name, policy_label)
        return guard

    @classmethod
    def _config_from_registry(
        cls, client: Any, agent_name: str, *, strict: bool, timeout: float
    ) -> "Guardrails":
        """Text-guardrail half of :meth:`from_agent`: a :meth:`from_config`
        pipeline, or a no-op one after warning (raising when ``strict``)."""

        from agentops._env import (
            basic_auth_header,
            get_host,
            get_public_key,
            get_secret_key,
        )

        host = (getattr(client, "_host", None) or get_host() or "").rstrip("/")
        public_key = getattr(client, "_public_key", None) or get_public_key()
        secret_key = getattr(client, "_secret_key", None) or get_secret_key()

        def unavailable(reason: str) -> "Guardrails":
            message = (
                f"Guardrails.from_agent: could not load guardrail config for "
                f"agent '{agent_name}' from {host or '<no host configured>'}: "
                f"{reason}"
            )
            if strict:
                raise GuardrailConfigUnavailableError(message)
            message += ". Falling back to a no-op Guardrails()."
            _log.warning(message)
            warnings.warn(message, GuardrailConfigWarning, stacklevel=4)
            return cls()

        if not host or not public_key or not secret_key:
            return unavailable(
                "AGENTOPS_HOST, AGENTOPS_PUBLIC_KEY and AGENTOPS_SECRET_KEY "
                "must all be set"
            )

        import requests as _requests

        auth_headers = basic_auth_header(public_key, secret_key)
        list_url = f"{host}/api/public/agents"

        try:
            resp = _requests.get(
                list_url,
                # `name` is a case-insensitive substring filter; 100 is the
                # public API's page-size cap.
                params={"name": agent_name, "limit": 100},
                headers=auth_headers,
                timeout=timeout,
            )
        except Exception as e:  # network error, DNS, timeout, ...
            return unavailable(f"GET /api/public/agents failed ({e})")

        if resp.status_code == 404:
            return unavailable(
                "GET /api/public/agents returned 404; the server predates "
                "the public Agent Registry API"
            )
        if resp.status_code in (401, 403):
            return unavailable(
                f"GET /api/public/agents returned {resp.status_code}; use a "
                "project-scoped key for the agent's project"
            )
        if not resp.ok:
            return unavailable(
                f"GET /api/public/agents returned HTTP {resp.status_code}"
            )

        try:
            listing = resp.json()
        except ValueError:
            return unavailable("GET /api/public/agents returned a non-JSON body")

        rows = listing.get("data") if isinstance(listing, dict) else None
        matches = [
            row
            for row in (rows if isinstance(rows, list) else [])
            if isinstance(row, dict) and row.get("name") == agent_name
        ]
        if not matches:
            return unavailable(
                "no agent with that exact name in the API key's project"
            )
        if len(matches) > 1:
            return unavailable(
                f"{len(matches)} agents in the API key's project have that "
                "name, so the lookup is ambiguous"
            )
        agent = matches[0]

        config = agent.get("guardrailConfig")
        if config is None and agent.get("id"):
            detail_path = f"/api/public/agents/{agent['id']}"
            try:
                detail = _requests.get(
                    f"{host}{detail_path}", headers=auth_headers, timeout=timeout
                )
            except Exception as e:
                return unavailable(f"GET {detail_path} failed ({e})")
            if detail.ok:
                try:
                    body = detail.json()
                except ValueError:
                    body = None
                if isinstance(body, dict):
                    config = body.get("guardrailConfig")

        if config is None:
            return unavailable(
                "the agent exists, but the server does not return a "
                "guardrailConfig for it (not exposed by this server version, "
                "or none is set). Use AgentOps.check_guardrails(text, "
                "agent_name=...) for server-side enforcement"
            )
        if not isinstance(config, dict):
            return unavailable("the agent's guardrailConfig is not an object")

        return cls.from_config(config)

    # ------------------------------------------------------------------ #
    # Tool-call policy (Agent Governance Toolkit via agentops.governance)
    # ------------------------------------------------------------------ #
    def _load_registry_policy(self, client: Any, agent_name: str, label: str) -> None:
        get_policy = getattr(client, "get_policy", None)
        if not callable(get_policy):
            return
        try:
            from .policies import PolicyNotFound
        except Exception:  # pragma: no cover
            return
        try:
            policy = get_policy(agent_name, label=label)
        except PolicyNotFound:
            return
        except Exception as e:
            import logging

            logging.getLogger("agentops.guardrails").warning(
                "Guardrails.from_agent(%s): could not load tool-call policy (%s)", agent_name, e
            )
            return
        if hasattr(policy, "close") and hasattr(policy, "send") and hasattr(policy, "__await__"):
            # AsyncAgentOps.get_policy returned a coroutine: nothing to do synchronously.
            policy.close()
            import logging

            logging.getLogger("agentops.guardrails").warning(
                "Guardrails.from_agent(%s): async client — await client.get_policy() and pass "
                "tool_policy= to Guardrails() instead",
                agent_name,
            )
            return
        if policy is not None:
            self.set_tool_policy(policy)

    def set_tool_policy(self, policy: Any, *, version: Optional[Any] = None) -> None:
        """Set the AGT policy used by :meth:`pre_tool_call` / :meth:`govern`.

        ``policy`` is an :class:`agentops.policies.AgentPolicy` (its manifest
        and version are used) or any ``PolicySpec`` accepted by
        ``agentops.governance``: YAML text, a file path or an
        ``agentmesh.governance.Policy``.
        """
        spec_fn = getattr(policy, "to_agt_policy_spec", None)
        if callable(spec_fn):
            self.policy = policy
            self._agt_policy = spec_fn()
            self._agt_policy_version = version if version is not None else getattr(policy, "version", None)
        else:
            self.policy = None
            self._agt_policy = policy
            self._agt_policy_version = version

    @property
    def has_tool_policy(self) -> bool:
        return self._agt_policy is not None

    @property
    def tool_policy(self) -> Any:
        """The ``PolicySpec`` handed to ``agentops.governance`` (or ``None``)."""
        return self._agt_policy

    @property
    def agent_id(self) -> str:
        """The AGT agent id policies are evaluated for: the agent's registry name."""
        return self.agent_name or "*"

    def _require_governance(self) -> Any:
        if self._agt_policy is None:
            raise RuntimeError(
                "No tool-call policy is loaded for "
                f"{'agent ' + repr(self.agent_name) if self.agent_name else 'this Guardrails instance'}. "
                "Publish an AGT policy for the agent in the AOS Control registry "
                "(Guardrails.from_agent() fetches it via client.get_policy()) or pass "
                "tool_policy=... to Guardrails(). Tool-call policy also requires the governance "
                'extra: pip install "agentops-sdk[governance]".'
            )
        from . import governance as _governance

        try:
            _governance._agt()
        except ImportError as e:
            raise RuntimeError(
                "Guardrails.pre_tool_call()/govern() need the Agent Governance Toolkit. "
                f'Install the extra: pip install "agentops-sdk[governance]" ({e})'
            ) from e
        return _governance

    def _decision_result(self, decision: Any) -> GuardrailResult:
        action = str(getattr(decision, "action", "") or "")
        rule = getattr(decision, "rule", None)
        blocked = action in ("deny", "require_approval")
        flags = [f"policy_{action}"] if action else []
        reasons: list[str] = []
        reason: Optional[str] = None
        if blocked:
            reasons.append(f"policy_denied:{rule or 'none'}")
            reason = getattr(decision, "reason", None) or (
                f"Tool call '{decision.tool_name}' {action} by policy rule '{rule or 'none'}'"
            )
        elif action in ("warn", "log"):
            reasons.append(f"policy_{action}:{rule or 'none'}")
            reason = getattr(decision, "reason", None)
        metadata: dict[str, Any] = {
            "decision": decision,
            "policy_action": action,
            "rule": rule,
            "policy_name": getattr(decision, "policy_name", None),
            "policy_version": self._agt_policy_version,
            "tool_name": getattr(decision, "tool_name", None),
            "stage": getattr(decision, "stage", None),
            "agent_id": getattr(decision, "agent_id", None),
        }
        return GuardrailResult(blocked=blocked, reason=reason, flags=flags, metadata=metadata, reasons=reasons)

    def pre_tool_call(
        self,
        tool_name: str,
        args: Optional[dict[str, Any]] = None,
        *,
        stage: str = "pre_tool",
        session_id: Any = None,
        **kwargs: Any,
    ) -> GuardrailResult:
        """Evaluate the agent's tool-call policy for ``tool_name(**args)``.

        Delegates to ``agentops.governance.pre_tool_call`` with
        ``agent_id=agent_name`` and the loaded policy, emitting the
        ``agt.policy.evaluate`` span and (when a shipper is configured) a
        ``policy.evaluation`` governance event. Returns a
        :class:`GuardrailResult` whose ``action`` is ``"block"`` when the
        policy decision is ``deny`` or ``require_approval`` and ``"allow"``
        otherwise (``warn`` / ``log`` are allowed but flagged);
        ``reasons == ["policy_denied:<rule>"]`` on a block and
        ``metadata["decision"]`` is the underlying ``Decision``.

        Raises ``RuntimeError`` if no policy is loaded or the
        ``agentops-sdk[governance]`` extra is not installed.
        """
        governance = self._require_governance()
        kwargs.setdefault("shipper", self._shipper)
        kwargs.setdefault("policy_version", self._agt_policy_version)
        decision = governance.pre_tool_call(
            self._agt_policy,
            self.agent_id,
            tool_name,
            args,
            stage=stage,
            session_id=session_id,
            **kwargs,
        )
        return self._decision_result(decision)

    def govern(self, fn: Any = None, **kwargs: Any) -> Any:
        """Wrap a tool function with the agent's policy — ``@guard.govern``,
        ``@guard.govern(tool_name=...)`` or ``guard.govern(fn)``.

        Delegates to ``agentops.governance.governed(fn, policy=<loaded
        policy>, agent_id=agent_name, **kwargs)``; denied calls raise
        ``agentops.governance.PolicyDenied`` unless ``raise_on_deny=False``.
        Raises ``RuntimeError`` if no policy is loaded or the
        ``agentops-sdk[governance]`` extra is not installed.
        """
        governance = self._require_governance()
        kwargs.setdefault("shipper", self._shipper)
        kwargs.setdefault("policy_version", self._agt_policy_version)
        return governance.governed(fn, policy=self._agt_policy, agent_id=self.agent_id, **kwargs)



def _contract_from_post_output(post: dict[str, Any]) -> Optional[dict[str, Any]]:
    """The contract in force for a server ``postOutput`` block, or ``None``.

    A refused/malformed ``outputContract`` is skipped with a warning (the
    server rejects those on save; this keeps a bad stored value from raising
    inside the host application).
    """
    if "outputContract" in post:
        contract, err = parse_output_contract(post["outputContract"])
        if contract is None:
            warnings.warn(
                f"Ignoring invalid outputContract in guardrail config: {err}",
                GuardrailConfigWarning,
                stacklevel=3,
            )
            return None
        return None if contract["kind"] == "none" else contract
    contract = resolve_output_contract(post)
    return None if contract["kind"] == "none" else contract


class GuardrailConfigWarning(UserWarning):
    """Emitted when :meth:`Guardrails.from_agent` falls back to a no-op."""


class GuardrailConfigUnavailableError(RuntimeError):
    """Raised by :meth:`Guardrails.from_agent` with ``strict=True`` when the
    agent's guardrail config cannot be loaded."""
