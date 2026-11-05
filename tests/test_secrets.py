"""The SDK's secret detectors agree with the server's (#867).

The fixtures mirror ``worker/src/__tests__/guardrailSecrets.test.ts``: the same
positives (one per detector) and the same negatives. They are assembled at
runtime so no complete credential-shaped string appears in the repository.
"""

import base64
import json
import time

import pytest

from agentops._secrets import entropy_bits_per_char, find_secrets, redact_secrets

ALNUM = "A1b2C3d4E5f6G7h8I9j0KlMnOpQrStUvWxYz0123456789abcdefghijklmnopqrstuvwxyz"
URLSAFE = ALNUM + "_-"
B64 = ALNUM + "+/"
UPPER_DIGITS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567"


def fill(n: int, seed: int = 7, alphabet: str = ALNUM) -> str:
    # mulberry32-style mixing so the filler has real-key entropy.
    a, out = seed & 0xFFFFFFFF, []
    for _ in range(n):
        a = (a + 0x6D2B79F5) & 0xFFFFFFFF
        t = a
        t = ((t ^ (t >> 15)) * (t | 1)) & 0xFFFFFFFF
        t ^= (t + (((t ^ (t >> 7)) * (t | 61)) & 0xFFFFFFFF)) & 0xFFFFFFFF
        out.append(alphabet[((t ^ (t >> 14)) & 0xFFFFFFFF) % len(alphabet)])
    return "".join(out)


def b64url(obj) -> str:
    return base64.urlsafe_b64encode(json.dumps(obj).encode()).decode().rstrip("=")


def kinds(text: str):
    return [f.kind for f in find_secrets(text)]


POSITIVES = [
    ("aws_access_key_id", "key id: " + "AK" + "IA" + fill(16, 1, UPPER_DIGITS)),
    ("aws_secret_access_key", f'aws_secret_access_key = "{fill(40, 2, B64)}"'),
    ("google_api_key", "maps key " + "AI" + "za" + fill(35, 3, URLSAFE)),
    ("azure_storage_key", f"AccountKey={fill(86, 4, B64)}==;EndpointSuffix=x"),
    ("github_token", "token " + "gh" + "p_" + fill(36, 5)),
    ("github_token", "github" + "_pat_" + fill(60, 6, ALNUM + "_")),
    ("gitlab_token", "gl" + "pat-" + fill(24, 8, URLSAFE)),
    ("slack_token", "xo" + "xb-" + fill(12, 9, "0123456789") + "-" + fill(24, 10)),
    ("stripe_key", "sk" + "_live_" + fill(28, 14)),
    ("sendgrid_key", "S" + "G." + fill(22, 15, URLSAFE) + "." + fill(43, 16, URLSAFE)),
    ("npm_token", "np" + "m_" + fill(36, 17)),
    ("openai_key", "sk" + "-proj-" + fill(60, 18, URLSAFE)),
    ("openai_key", "s" + "k-" + fill(48, 19)),
    ("anthropic_key", "sk" + "-ant-api03-" + fill(95, 20, URLSAFE)),
    ("jwt", "got " + b64url({"alg": "HS256", "typ": "JWT"}) + "." + b64url({"sub": "123"}) + "." + fill(43, 21, URLSAFE)),
    ("private_key_block", "-----BEGIN RSA PRIV" + "ATE KEY-----\n" + fill(64, 22, B64)),
    ("credentials_in_url", "postgres://app:" + fill(14, 23) + "@db.internal:5432/prod"),
    ("bearer_token", "Authorization: Bearer " + fill(40, 25, URLSAFE)),
    ("generic_credential", "the admin password is hunter2-LEAKED"),
    ("generic_credential", "API_KEY=" + fill(32, 26)),
    ("generic_credential", '{"client_secret": "' + fill(30, 27) + '"}'),
]


@pytest.mark.parametrize("kind,text", POSITIVES)
def test_flags(kind, text):
    assert kind in kinds(text)


NEGATIVES = [
    "Please reset your password using the link we emailed you.",
    "You can rotate the API key in Settings, then paste the new token.",
    "The secret to good bread is patience.",
    "Tokens used in this request: 1234 input, 56 output.",
    "I can't share the password or any credentials.",
    "api_key=YOUR_API_KEY",
    "API_KEY=<your-api-key>",
    "password: ********",
    "Authorization: Bearer ${TOKEN}",
    "export SECRET_KEY=changeme",
    "password = password123",
    'token: "xxxxxxxxxxxxxxxx"',
    "secret: null",
    "client_secret: [REDACTED]",
    "AWS key " + "AK" + "IA" + "IOSFODNN7EXAMPLE",
    "commit 9fceb02d0ae598e95dc970b74767f19372d61af8",
    "sha256: e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "id 3f2504e0-4f89-41d3-9a0c-0305e82c3301",
    "data:image/png;base64," + fill(400, 40, B64),
    "eyJub3Rqc29u.eyJub3Rqc29u.c2lnbmF0dXJlLXNpZ25hdHVyZQ",
    "see https://example.com/docs/api-keys and mail user@example.com",
    "git@github.com:org/repo.git",
    "password_file = /etc/secrets/db-password.txt",
    "",
]


@pytest.mark.parametrize("text", NEGATIVES)
def test_does_not_flag(text):
    assert find_secrets(text) == []


def test_findings_hold_kind_and_span_only():
    secret = "gh" + "p_" + fill(36, 30)
    text = f"my token {secret} thanks"
    (f,) = find_secrets(text)
    assert text[f.start : f.end] == secret
    assert secret not in repr(f)


def test_jwt_inside_a_bearer_header_is_reported_once():
    jwt = b64url({"alg": "RS256"}) + "." + b64url({"sub": "x"}) + "." + fill(43, 32, URLSAFE)
    found = find_secrets("Authorization: Bearer " + jwt)
    assert [f.kind for f in found] == ["jwt"]


def test_redaction_replaces_values_and_keeps_the_rest():
    gh = "gh" + "p_" + fill(36, 50)
    text = f"deploy with {gh} and the password is hunter2-LEAKED, thanks"
    redacted, found = redact_secrets(text)
    assert gh not in redacted and "hunter2-LEAKED" not in redacted
    assert redacted.startswith("deploy with [REDACTED_SECRET] and")
    assert redacted.endswith(", thanks")
    assert found == ["generic_credential", "github_token"]
    assert redact_secrets("hello there") == ("hello there", [])


def test_long_text_is_sampled_from_head_and_tail():
    head = "gh" + "p_" + fill(36, 70)
    tail = "sk" + "_live_" + fill(28, 71)
    middle = "x " * 100_000
    found = find_secrets(f"{head} {middle} {tail}", max_chars=10_000)
    assert sorted(f.kind for f in found) == ["github_token", "stripe_key"]
    buried = "gh" + "p_" + fill(36, 72)
    filler = "lorem ipsum " * 5_000
    assert find_secrets(filler + buried + filler, max_chars=10_000) == []


def test_findings_are_capped():
    many = " ".join("gh" + "p_" + fill(36, 100 + i) for i in range(200))
    assert len(find_secrets(many, max_findings=10)) == 10


@pytest.mark.parametrize(
    "text",
    [
        "a" * 100_000,
        "eyJ" * 30_000,
        "password " * 11_000,
        "password: " * 10_000,
        "Bearer " * 14_000,
        "https://a:" * 10_000,
        "-----BEGIN " * 9_000,
        "sk-" * 33_000,
        fill(100_000, 80, B64),
    ],
)
def test_adversarial_input_stays_fast(text):
    start = time.perf_counter()
    find_secrets(text)
    assert time.perf_counter() - start < 1.5


def test_entropy():
    assert entropy_bits_per_char("aaaaaaaa") == 0
    assert entropy_bits_per_char(fill(64, 60)) > 4
