"""The replay capture: events are masked and serialized as Langfuse v2's
ingestion consumer would, routed by trace id, and never queued."""

from __future__ import annotations

import datetime as dt

from langfuse._task_manager.task_manager import TaskManager

from agentops.evolve.capture import TraceCapture, capture, plain


class Body:
    def __init__(self, **values):
        self.values = values

    def dict(self, exclude_none=False):
        return {k: v for k, v in self.values.items() if not (exclude_none and v is None)}


def test_plain_masks_and_serializes_like_the_consumer():
    def mask(data):
        if data == "boom":
            raise ValueError
        return "***"

    when = dt.datetime(2026, 10, 26, tzinfo=dt.timezone.utc)
    event = {
        "type": "span-create",
        "body": Body(traceId="t", input="secret", output=None, startTime=when),
    }
    out = plain(event, mask)
    assert out["body"] == {"traceId": "t", "input": "***", "startTime": "2026-10-26T00:00:00Z"}
    assert plain({"body": {"output": "boom"}}, mask)["body"]["output"] == (
        "<fully masked due to failed mask function>"
    )


def test_routes_by_trace_id_and_counts_the_rest():
    sink = TraceCapture()
    sink.watch("r1")
    sink.add({"type": "trace-create", "body": {"id": "r1"}})
    sink.add({"type": "span-create", "body": {"traceId": "r1"}})
    sink.add({"type": "span-create", "body": {"traceId": "other"}})
    assert len(sink.events("r1")) == 2 and sink.unrouted == 1


def test_capture_patches_add_task_only_inside_the_block():
    original = TaskManager.add_task
    manager = TaskManager.__new__(TaskManager)
    manager._enabled, manager._mask = True, None
    with capture() as sink:
        sink.watch("r1")
        TaskManager.add_task(manager, {"type": "trace-create", "body": {"id": "r1"}})
    assert TaskManager.add_task is original
    assert sink.events("r1")[0]["timestamp"]
