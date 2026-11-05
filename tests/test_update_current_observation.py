"""``update_current_observation`` on the clients and at module level.

The published 0.4.0 wheel shipped ``update_current_trace`` but not
``update_current_observation``, so ``client.update_current_observation(...)``
fell through ``__getattr__`` to the Langfuse v2 client, which has no such
method (it lives on ``langfuse_context``). Securitas carries a monkey-patch
(``core/app/utils/observability/_agentops_patch.py``) for exactly this. The
shim skips itself when the class defines the method natively, which these
tests pin down.
"""

from unittest.mock import patch

import pytest

import agentops
from agentops import AgentOps, AsyncAgentOps

CREDS = dict(host="http://127.0.0.1:9", public_key="pk-lf-test", secret_key="sk-lf-test")

FULL_KWARGS = dict(
    model="gpt-4o-mini",
    input=[{"role": "user", "content": "hi"}],
    output="hello",
    usage_details={"input": 3, "output": 1},
    cost_details={"input": 0.1},
    metadata={"k": "v"},
    level="WARNING",
    status_message="slow",
    completion_start_time="2026-09-23T00:00:00Z",
    model_parameters={"temperature": 0},
    version="v1",
)


@pytest.fixture(params=[AgentOps, AsyncAgentOps], ids=["sync", "async"])
def client_cls(request):
    return request.param


def test_method_is_defined_on_the_class_not_proxied(client_cls):
    # Securitas' shim returns early when this is a native (unpatched) method.
    native = client_cls.__dict__.get("update_current_observation")
    assert callable(native)
    assert getattr(native, "_agentops_patched", False) is False


def test_forwards_every_field_to_langfuse_context(client_cls):
    client = client_cls(**CREDS)
    with patch("langfuse.decorators.langfuse_context") as ctx:
        client.update_current_observation(**FULL_KWARGS, name="renamed")
    ctx.update_current_observation.assert_called_once_with(**FULL_KWARGS, name="renamed")


def test_omits_unset_fields(client_cls):
    client = client_cls(**CREDS)
    with patch("langfuse.decorators.langfuse_context") as ctx:
        client.update_current_observation(model="m", usage_details={"input": 1})
    ctx.update_current_observation.assert_called_once_with(
        model="m", usage_details={"input": 1}
    )


def test_is_a_no_op_when_the_client_is_disabled(client_cls, monkeypatch):
    for var in ("AGENTOPS_HOST", "AGENTOPS_BASE_URL", "AGENTOPS_PUBLIC_KEY", "AGENTOPS_SECRET_KEY"):
        monkeypatch.delenv(var, raising=False)
    client = client_cls(host=None, public_key=None, secret_key=None)
    assert client.enabled is False
    with patch("langfuse.decorators.langfuse_context") as ctx:
        client.update_current_observation(model="m")
    ctx.update_current_observation.assert_not_called()


def test_module_level_shortcut_forwards():
    with patch("agentops.langfuse_context") as ctx:
        agentops.update_current_observation(model="m", output="o", usage_details={"input": 2})
    ctx.update_current_observation.assert_called_once_with(
        model="m", output="o", usage_details={"input": 2}
    )

