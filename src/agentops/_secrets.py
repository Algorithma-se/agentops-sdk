"""Secret-pattern detection for the SDK's local guardrails (#867).

A port of ``packages/shared/src/guardrails/secrets.ts`` so the SDK's local
``Guardrails.check_output`` agrees with the server pipeline. It looks for
credential *shapes* (cloud and vendor keys, validated JWTs, private-key
blocks, credentials in URLs) and for a credential keyword followed by a value
that looks like one, instead of flagging the word "password". Findings carry
the kind and character span only, never the secret. Keep it in step with the
TypeScript version: both are tested against the same positives and negatives.
"""

from __future__ import annotations

import base64
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Optional

DEFAULT_MAX_CHARS = 100_000
_DEFAULT_MAX_FINDINGS = 50


@dataclass(frozen=True)
class SecretFinding:
    kind: str
    confidence: str  # "high" | "medium"
    start: int
    end: int


def entropy_bits_per_char(s: str) -> float:
    if not s:
        return 0.0
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in Counter(s).values())


_PLACEHOLDER_RE = re.compile(
    r"^(?:x{3,}|\*{3,}|\.{3,}|\?{3,}|<[^>]*>|\$\{[^}]*\}|\{\{[^}]*\}\}|\[[^\]]*\]"
    r"|your[_\-\s]?\w*|my[_\-\s]?\w*|example\w*|sample\w*|dummy\w*|placeholder\w*"
    r"|changeme|change[_-]me|password\d*|passw0rd|secret|token|null|none|undefined"
    r"|true|false|redacted\w*|todo|tbd|test\w*|abc\w*|12345\w*)$",
    re.IGNORECASE,
)


def _is_placeholder(value: str) -> bool:
    if _PLACEHOLDER_RE.match(value):
        return True
    if re.search(r"redacted|\bexample\b|xxxxx|\*\*\*\*", value, re.IGNORECASE):
        return True
    if re.match(r"^(.)\1{5,}$", value):
        return True
    return False


def _char_classes(value: str) -> int:
    n = 0
    if re.search(r"[a-z]", value):
        n += 1
    if re.search(r"[A-Z]", value):
        n += 1
    if re.search(r"\d", value):
        n += 1
    if re.search(r"[^A-Za-z0-9]", value):
        n += 1
    return n


def _looks_like_credential_value(value: str) -> bool:
    if len(value) < 6 or len(value) > 200:
        return False
    if _is_placeholder(value):
        return False
    if re.match(r"^(?:/|\./|\.\./|[a-z]+://)", value, re.IGNORECASE):
        return False
    if _char_classes(value) < 2:
        return False
    if len(value) >= 20:
        return entropy_bits_per_char(value) >= 3.3
    return bool(re.search(r"\d|[^A-Za-z0-9]", value)) and entropy_bits_per_char(value) >= 2.0


def _decode_b64url(part: str) -> Optional[str]:
    try:
        padded = part.replace("-", "+").replace("_", "/")
        padded += "=" * ((4 - len(padded) % 4) % 4)
        return base64.b64decode(padded).decode("utf-8")
    except Exception:
        return None


def _is_real_jwt(token: str) -> bool:
    parts = token.split(".")
    if len(parts) < 2:
        return False
    header, payload = _decode_b64url(parts[0]), _decode_b64url(parts[1])
    if not header or not payload:
        return False
    try:
        hj, pj = json.loads(header), json.loads(payload)
    except Exception:
        return False
    return isinstance(hj, dict) and isinstance(hj.get("alg"), str) and isinstance(pj, dict)


_Validate = Callable[[str], bool]


@dataclass(frozen=True)
class _Detector:
    kind: str
    confidence: str
    pattern: "re.Pattern[str]"
    group: int = 0
    validate: Optional[_Validate] = None


def _d(kind: str, confidence: str, regex: str, *, flags: int = 0, group: int = 0,
       validate: Optional[_Validate] = None) -> _Detector:
    return _Detector(kind, confidence, re.compile(regex, flags), group, validate)


_not_placeholder: _Validate = lambda v: not _is_placeholder(v)  # noqa: E731

# Every quantifier is bounded and every pattern starts from a distinctive
# anchor, so none can backtrack catastrophically on adversarial text.
_DETECTORS = [
    _d("aws_access_key_id", "high", r"\b(?:AKIA|ASIA|ABIA|ACCA)[A-Z0-9]{16}\b",
       validate=lambda v: "EXAMPLE" not in v),
    _d("aws_secret_access_key", "high",
       r"aws[_\-\s]?secret[_\-\s]?(?:access[_\-\s]?)?key[\"']?\s{0,3}[:=]\s{0,3}[\"']?([A-Za-z0-9/+=]{40})(?![A-Za-z0-9/+=])",
       flags=re.IGNORECASE, group=1, validate=_not_placeholder),
    _d("google_api_key", "high", r"\bAIza[0-9A-Za-z_-]{35}\b"),
    _d("azure_storage_key", "high", r"AccountKey=([A-Za-z0-9+/]{86}==)", group=1),
    _d("github_token", "high", r"\b(?:gh[pousr]_[A-Za-z0-9]{36,255}|github_pat_[A-Za-z0-9_]{22,255})\b"),
    _d("gitlab_token", "high", r"\bglpat-[A-Za-z0-9_-]{20,64}\b"),
    _d("slack_token", "high",
       r"\b(?:xox[abprs]-[A-Za-z0-9-]{10,200}|https://hooks\.slack\.com/services/T[A-Z0-9]{8,12}/B[A-Z0-9]{8,12}/[A-Za-z0-9]{24})"),
    _d("stripe_key", "high", r"\b(?:sk|rk)_live_[0-9a-zA-Z]{24,99}\b"),
    _d("sendgrid_key", "high", r"\bSG\.[A-Za-z0-9_-]{22}\.[A-Za-z0-9_-]{43}\b"),
    _d("npm_token", "high", r"\bnpm_[A-Za-z0-9]{36}\b"),
    _d("openai_key", "high", r"\bsk-(?:(?:proj|svcacct|admin)-[A-Za-z0-9_-]{40,200}|[A-Za-z0-9]{48})\b"),
    _d("anthropic_key", "high", r"\bsk-ant-(?:api|admin)\d{2}-[A-Za-z0-9_-]{80,200}\b"),
    _d("jwt", "high", r"\beyJ[A-Za-z0-9_-]{8,2000}\.eyJ[A-Za-z0-9_-]{8,4000}\.[A-Za-z0-9_-]{10,2000}",
       validate=_is_real_jwt),
    _d("private_key_block", "high",
       r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP |ENCRYPTED )?PRIVATE KEY(?: BLOCK)?-----"),
    _d("credentials_in_url", "high",
       r"\b[a-z][a-z0-9+.-]{1,20}://[^\s:@/]{1,64}:([^\s@/]{3,128})@[^\s/]{1,255}",
       flags=re.IGNORECASE, group=1,
       validate=lambda v: not _is_placeholder(v) and not re.match(r"^\$\{|^%\w+%$|^<|^\{", v)),
    _d("bearer_token", "medium", r"\bBearer\s{1,4}([A-Za-z0-9\-._~+/]{20,2000}=*)", group=1,
       validate=lambda v: not _is_placeholder(v) and entropy_bits_per_char(v) >= 3.3),
    _d("generic_credential", "medium",
       r"\b(?:api[_-]?key|apikey|secret(?:[_-]?key)?|(?:access|auth|refresh|session)[_-]?token"
       r"|client[_-]?secret|passwd|password|pwd)\b[\"'`]?\s{0,3}(?:[:=]|=>|\bis\b)\s{0,3}[\"'`]?"
       r"([^\s\"'`<>{}\[\],;]{6,200})",
       flags=re.IGNORECASE, group=1, validate=_looks_like_credential_value),
]


def _scan_window(text: str, offset: int, out: list[SecretFinding], max_findings: int) -> None:
    for d in _DETECTORS:
        if len(out) >= max_findings:
            return
        for m in d.pattern.finditer(text):
            if not m.group(0):
                continue
            value = m.group(d.group)
            if value is None:
                continue
            if d.validate and not d.validate(value):
                continue
            out.append(SecretFinding(d.kind, d.confidence, m.start(d.group) + offset, m.end(d.group) + offset))
            if len(out) >= max_findings:
                return


def _dedupe(findings: list[SecretFinding]) -> list[SecretFinding]:
    ordered = sorted(findings, key=lambda f: (f.start, -(f.end - f.start), 0 if f.confidence == "high" else 1))
    kept: list[SecretFinding] = []
    for f in ordered:
        if kept and f.start >= kept[-1].start and f.end <= kept[-1].end:
            continue
        kept.append(f)
    return kept


def find_secrets(text: str, *, max_chars: int = DEFAULT_MAX_CHARS,
                 max_findings: int = _DEFAULT_MAX_FINDINGS) -> list[SecretFinding]:
    """Secrets in ``text``. Text over ``max_chars`` is sampled from its head and tail."""
    if not text:
        return []
    found: list[SecretFinding] = []
    if len(text) <= max_chars:
        _scan_window(text, 0, found, max_findings)
    else:
        half = max_chars // 2
        _scan_window(text[:half], 0, found, max_findings)
        tail_start = len(text) - half
        _scan_window(text[tail_start:], tail_start, found, max_findings)
    return _dedupe(found)


def redact_secrets(text: str, findings: Optional[list[SecretFinding]] = None) -> tuple[str, list[str]]:
    """Replace each secret value with ``[REDACTED_SECRET]``; returns the text and the kinds."""
    findings = find_secrets(text) if findings is None else findings
    if not findings:
        return text, []
    out, cursor = [], 0
    for f in sorted(findings, key=lambda f: f.start):
        if f.start < cursor:
            continue
        out.append(text[cursor:f.start])
        out.append("[REDACTED_SECRET]")
        cursor = f.end
    out.append(text[cursor:])
    return "".join(out), sorted({f.kind for f in findings})
