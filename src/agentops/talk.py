"""Talk correlation: let an agent's own trace show up as a trace fold
and cost line in the AgentOps Talk UI.

Talk sends an ``x-aos-talk-turn-id`` header on every call to your agent
(HTTP, SSE, and WS transports alike). If your agent stamps that value
back onto its own trace's ``metadata.talkTurnId``, Talk can resolve
``talk.getReplySteps`` / ``talk.getReplyProvenance`` for the reply and
show a trace fold + a "11:34 · 1.2s · $0.0031 · trace 7f3a…" provenance
line under it. If you don't, Talk still works — it just shows neither,
rather than a fake one (Talk never fabricates trace data it can't back
up with a real correlation).

Two ways to wire this up, matching the two ways this SDK creates a root
trace:

Manual trace (``agentops.trace(...)`` — the pattern used in
``examples/agent_bootstrap.py`` and the agent-instrumentation guide)::

    from agentops.talk import talk_turn_metadata

    trace = agentops.trace(
        name="my-agent",
        session_id=conversation_id,
        metadata=talk_turn_metadata(request.headers),
    )

``@observe()``-decorated request handler (trace lives in the decorator
context rather than a variable you hold)::

    from agentops.talk import echo_talk_turn_id

    @observe()
    def handle_request(request):
        echo_talk_turn_id(request.headers)
        ...

Both accept either a headers mapping (matched case-insensitively, so it
works with Flask's, FastAPI's, or a plain ``dict``) or an
already-extracted turn id string.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Union

# Must match `TALK_TURN_ID_HEADER` in
# web/src/features/talk/server/constants.ts. HTTP header names are
# case-insensitive, so `get_talk_turn_id` below matches it
# case-insensitively rather than assuming any particular framework's
# casing convention.
TALK_TURN_ID_HEADER = "x-aos-talk-turn-id"

# The trace metadata key `getReplyProvenance`/`getReplySteps` look for
# (see web/src/features/talk/server/router.ts).
TALK_TURN_ID_METADATA_KEY = "talkTurnId"

HeadersOrTurnId = Union[Mapping[str, Any], str, None]


def get_talk_turn_id(headers_or_turn_id: HeadersOrTurnId) -> Optional[str]:
    """Extract the Talk correlation id, or ``None`` when there isn't one.

    Accepts a headers mapping (any object with an ``.items()`` method —
    Flask's ``request.headers``, FastAPI's ``request.headers``, a plain
    ``dict``, etc.), an already-extracted turn id string, or ``None``.
    A request that didn't come from Talk simply has no such header —
    that's a normal, expected case, not an error.
    """
    if headers_or_turn_id is None:
        return None
    if isinstance(headers_or_turn_id, str):
        return headers_or_turn_id or None

    try:
        items = headers_or_turn_id.items()
    except AttributeError:
        return None

    for key, value in items:
        if isinstance(key, str) and key.lower() == TALK_TURN_ID_HEADER and value:
            return value
    return None


def talk_turn_metadata(headers_or_turn_id: HeadersOrTurnId) -> dict[str, str]:
    """``{"talkTurnId": <id>}`` when present, else ``{}``.

    Merge this into whatever you pass as a trace's ``metadata`` — e.g.
    ``agentops.trace(..., metadata=talk_turn_metadata(request.headers))``
    or ``trace.update(metadata=talk_turn_metadata(request.headers))``.
    Returning an empty dict (rather than ``None``) means this always
    composes safely with ``{**your_metadata, **talk_turn_metadata(...)}``.
    """
    turn_id = get_talk_turn_id(headers_or_turn_id)
    return {TALK_TURN_ID_METADATA_KEY: turn_id} if turn_id else {}


def echo_talk_turn_id(headers_or_turn_id: HeadersOrTurnId) -> Optional[str]:
    """Stamp the Talk correlation id onto the *active* ``@observe()`` trace.

    Only useful when called from inside a function decorated with
    ``@observe()`` — it updates the decorator's current-trace context,
    not a trace object you're holding a reference to. For the
    ``agentops.trace(...)`` (manual trace) pattern, use
    :func:`talk_turn_metadata` instead.

    A no-op (returns ``None``, updates nothing) when there's no turn id
    to stamp — same "degrade to nothing rather than fake it" rule Talk's
    own UI follows.
    """
    turn_id = get_talk_turn_id(headers_or_turn_id)
    if not turn_id:
        return None

    from langfuse.decorators import langfuse_context  # type: ignore[import-untyped]

    langfuse_context.update_current_trace(
        metadata={TALK_TURN_ID_METADATA_KEY: turn_id}
    )
    return turn_id
