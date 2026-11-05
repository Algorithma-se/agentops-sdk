"""Guardrails.from_agent against a mocked Agent Registry public API.

Kept separate from test_guardrails.py, which skips the whole module when
the optional `ff3` extra is not installed.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import requests

from agentops.guardrails import (
    GuardrailConfigUnavailableError,
    GuardrailConfigWarning,
    Guardrails,
)

HOST = "https://aos.example.com"
# base64("pk-lf-test:sk-lf-test")
BASIC = "Basic cGstbGYtdGVzdDpzay1sZi10ZXN0"
CLIENT = SimpleNamespace(_host=HOST, _public_key="pk-lf-test", _secret_key="sk-lf-test")

GUARDRAIL_CONFIG = {
    "enabled": True,
    "preInput": {
        "detectPii": True,
        "piiAction": "block",
        "detectJailbreak": True,
        "jailbreakAction": "mark_high_risk",
    },
    "postOutput": {"detectSecrets": True, "detectPiiLeak": True},
}


def _response(status: int, body=None) -> MagicMock:
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status
    resp.ok = 200 <= status < 400
    if isinstance(body, Exception):
        resp.json.side_effect = body
    else:
        resp.json.return_value = body
    return resp


def _agent(name: str = "support-agent", agent_id: str = "agent-1", **extra):
    return {"id": agent_id, "name": name, "status": "AVAILABLE", **extra}


def _listing(*agents):
    return {"data": list(agents), "meta": {"page": 1, "limit": 100, "totalItems": len(agents), "totalPages": 1}}


def _is_noop(guard: Guardrails) -> bool:
    return not any(
        [guard.detect_pii, guard.detect_jailbreak, guard.detect_secrets, guard.detect_pii_leak]
    )


class TestFromAgentSuccess:
    def test_maps_config_from_agent_detail(self):
        responses = [
            _response(200, _listing(_agent())),
            _response(200, _agent(guardrailConfig=GUARDRAIL_CONFIG)),
        ]
        with patch("requests.get", side_effect=responses) as get:
            guard = Guardrails.from_agent(CLIENT, "support-agent")

        assert guard.detect_pii and guard.pii_action == "block"
        assert guard.detect_jailbreak and guard.jailbreak_action == "flag"
        assert guard.detect_secrets and guard.detect_pii_leak

        list_call, detail_call = get.call_args_list
        assert list_call.args[0] == f"{HOST}/api/public/agents"
        assert list_call.kwargs["params"]["name"] == "support-agent"
        # Public API keys authenticate with HTTP Basic pk:sk, not Bearer sk.
        assert list_call.kwargs["headers"] == {"Authorization": BASIC}
        assert detail_call.args[0] == f"{HOST}/api/public/agents/agent-1"
        assert detail_call.kwargs["headers"] == {"Authorization": BASIC}

    def test_uses_config_from_listing_without_detail_call(self):
        listing = _listing(_agent(guardrailConfig=GUARDRAIL_CONFIG))
        with patch("requests.get", return_value=_response(200, listing)) as get:
            guard = Guardrails.from_agent(CLIENT, "support-agent")

        assert guard.detect_pii
        assert get.call_count == 1

    def test_picks_exact_name_from_substring_matches(self):
        listing = _listing(
            _agent(name="support-agent-v2", agent_id="other"),
            _agent(name="support-agent", agent_id="agent-1"),
        )
        responses = [
            _response(200, listing),
            _response(200, _agent(guardrailConfig=GUARDRAIL_CONFIG)),
        ]
        with patch("requests.get", side_effect=responses) as get:
            guard = Guardrails.from_agent(CLIENT, "support-agent")

        assert guard.detect_pii
        assert get.call_args_list[1].args[0].endswith("/api/public/agents/agent-1")

    def test_disabled_config_is_a_deliberate_noop_without_warning(self, recwarn):
        listing = _listing(_agent(guardrailConfig={"enabled": False}))
        with patch("requests.get", return_value=_response(200, listing)):
            guard = Guardrails.from_agent(CLIENT, "support-agent")

        assert _is_noop(guard)
        assert not [w for w in recwarn if issubclass(w.category, GuardrailConfigWarning)]

    def test_host_trailing_slash_is_normalised(self):
        client = SimpleNamespace(_host=f"{HOST}/", _public_key="pk", _secret_key="sk")
        listing = _listing(_agent(guardrailConfig=GUARDRAIL_CONFIG))
        with patch("requests.get", return_value=_response(200, listing)) as get:
            Guardrails.from_agent(client, "support-agent")

        assert get.call_args.args[0] == f"{HOST}/api/public/agents"


class TestFromAgentFallbacks:
    @pytest.mark.parametrize(
        "responses, expected_reason",
        [
            pytest.param([_response(404, None)], "predates the public Agent Registry API", id="route-missing"),
            pytest.param([_response(401, {"message": "Invalid"})], "returned 401", id="unauthorized"),
            pytest.param([_response(500, None)], "HTTP 500", id="server-error"),
            pytest.param([_response(200, ValueError("no json"))], "non-JSON", id="non-json"),
            pytest.param([_response(200, _listing())], "no agent with that exact name", id="not-found"),
            pytest.param(
                [_response(200, _listing(_agent(name="support-agent-v2")))],
                "no agent with that exact name",
                id="only-substring-match",
            ),
            pytest.param(
                [_response(200, _listing(_agent(agent_id="a"), _agent(agent_id="b")))],
                "ambiguous",
                id="duplicate-names",
            ),
            pytest.param(
                # Today's phase-4 servers: agent exists, config not exposed.
                [_response(200, _listing(_agent())), _response(200, _agent())],
                "does not return a guardrailConfig",
                id="config-not-exposed",
            ),
            pytest.param(
                [_response(200, _listing(_agent(guardrailConfig="nope")))],
                "not an object",
                id="config-wrong-type",
            ),
        ],
    )
    def test_warns_naming_agent_and_host_then_returns_noop(self, responses, expected_reason):
        with patch("requests.get", side_effect=responses):
            with pytest.warns(GuardrailConfigWarning) as record:
                guard = Guardrails.from_agent(CLIENT, "support-agent")

        assert _is_noop(guard)
        message = str(record[0].message)
        assert "'support-agent'" in message
        assert HOST in message
        assert expected_reason in message

    @pytest.mark.parametrize(
        "responses",
        [
            pytest.param([_response(404, None)], id="route-missing"),
            pytest.param([_response(200, _listing())], id="not-found"),
            pytest.param([_response(200, _listing(_agent())), _response(200, _agent())], id="config-not-exposed"),
        ],
    )
    def test_strict_raises_instead_of_degrading(self, responses):
        with patch("requests.get", side_effect=responses):
            with pytest.raises(GuardrailConfigUnavailableError, match="support-agent"):
                Guardrails.from_agent(CLIENT, "support-agent", strict=True)

    def test_network_error_warns(self):
        with patch("requests.get", side_effect=requests.ConnectionError("refused")):
            with pytest.warns(GuardrailConfigWarning, match="refused"):
                guard = Guardrails.from_agent(CLIENT, "support-agent")
        assert _is_noop(guard)

    def test_network_error_strict_raises(self):
        with patch("requests.get", side_effect=requests.Timeout("slow")):
            with pytest.raises(GuardrailConfigUnavailableError, match="slow"):
                Guardrails.from_agent(CLIENT, "support-agent", strict=True)

    def test_missing_credentials_warns_without_request(self, monkeypatch):
        for var in ("AGENTOPS_HOST", "AGENTOPS_BASE_URL", "AGENTOPS_PUBLIC_KEY", "AGENTOPS_SECRET_KEY"):
            monkeypatch.delenv(var, raising=False)
        client = SimpleNamespace(_host=None, _public_key=None, _secret_key=None)
        with patch("requests.get") as get:
            with pytest.warns(GuardrailConfigWarning, match="must all be set"):
                guard = Guardrails.from_agent(client, "support-agent")
        assert _is_noop(guard)
        get.assert_not_called()

    def test_also_logs_on_agentops_logger(self, caplog):
        with patch("requests.get", return_value=_response(404, None)):
            with pytest.warns(GuardrailConfigWarning):
                Guardrails.from_agent(CLIENT, "support-agent")
        assert any(
            r.name == "agentops" and "support-agent" in r.getMessage() for r in caplog.records
        )

    def test_project_id_is_still_accepted(self):
        listing = _listing(_agent(guardrailConfig=GUARDRAIL_CONFIG))
        with patch("requests.get", return_value=_response(200, listing)):
            guard = Guardrails.from_agent(CLIENT, "support-agent", project_id="ignored")
        assert guard.detect_pii
