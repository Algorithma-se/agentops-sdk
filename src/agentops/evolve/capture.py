"""Capturing a replay's trace events instead of sending them.

``capture()`` replaces Langfuse v2's ``TaskManager.add_task`` (where every
trace event is queued) for the replay: each event is masked and serialized as
the ingestion consumer would, then kept under its trace id. Nothing is queued,
so nothing is sent; ``LANGFUSE_SAMPLE_RATE=0`` is a second guard. Publishing
the artifact turns the kept events into traces in AOS Control.
"""

from __future__ import annotations

import contextlib
import json
import threading
from typing import Any, Awaitable, Callable, Iterator, TypeVar

T = TypeVar("T")


class TraceCapture:
    """The captured events of the watched trace ids; others are only counted."""

    def __init__(self) -> None:
        self._events: dict[str, list[dict]] = {}
        self._lock = threading.Lock()
        self.unrouted = 0

    def add(self, event: dict) -> None:
        body = event.get("body") or {}
        trace_id = body.get("id") if event.get("type") == "trace-create" else None
        trace_id = trace_id or body.get("traceId")
        with self._lock:
            if trace_id in self._events:
                self._events[trace_id].append(event)
            else:
                self.unrouted += 1

    def watch(self, trace_id: str) -> None:
        with self._lock:
            self._events.setdefault(trace_id, [])

    def events(self, trace_id: str) -> list[dict]:
        with self._lock:
            return list(self._events.get(trace_id, []))


def plain(event: dict, mask: Any) -> dict:
    """*event* as the ingestion consumer sends it: body a dict, input and
    output masked, JSON values only."""
    from langfuse.serializer import EventSerializer  # type: ignore[import-untyped]

    event = dict(event)
    body = event.get("body")
    if body is not None and not isinstance(body, dict) and hasattr(body, "dict"):
        body = body.dict(exclude_none=True)
    body = dict(body or {})
    if mask is not None:
        for key in ("input", "output"):
            if key in body:
                try:
                    body[key] = mask(data=body[key])
                except Exception:
                    body[key] = "<fully masked due to failed mask function>"
    event["body"] = body
    return json.loads(json.dumps(event, cls=EventSerializer))


@contextlib.contextmanager
def capture() -> Iterator[TraceCapture]:
    """Capture every trace event of this process until the block ends."""
    from langfuse._task_manager.task_manager import TaskManager  # type: ignore[import-untyped]
    from langfuse.utils import _get_timestamp  # type: ignore[import-untyped]

    sink = TraceCapture()
    original = TaskManager.add_task

    def add_task(self: Any, event: dict) -> None:
        if not self._enabled:
            return
        event["timestamp"] = _get_timestamp()
        sink.add(plain(event, self._mask))

    TaskManager.add_task = add_task  # type: ignore[method-assign]
    try:
        yield sink
    finally:
        TaskManager.add_task = original  # type: ignore[method-assign]


async def traced(
    sink: TraceCapture, trace_id: str, name: str, run: Callable[[], Awaitable[T]]
) -> T:
    """Run *run* as the root observation of trace *trace_id* named *name*;
    core's spans nest under it through Langfuse v2's decorator context."""
    from langfuse.decorators import observe  # type: ignore[import-untyped]

    sink.watch(trace_id)

    @observe(name=name, capture_input=False, capture_output=False)
    async def root() -> T:
        return await run()

    return await root(langfuse_observation_id=trace_id)  # type: ignore[call-arg]
