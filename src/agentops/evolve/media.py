"""Images of a frozen turn, sent to the agent again.

A trace keeps an image as a media reference (``@@@langfuseMedia:type=…|id=…|
source=…@@@``); the file is in AOS Control. ``Media.load`` downloads every
image the entries refer to before anything runs; ``inline`` gives the agent a
copy with each reference replaced by its ``data:`` URI. ``restore`` puts the
references back into what the replay records (its trace events and
messages), so the artifact and a publication hold no image data and the
published trace shows the same AOS image.
"""

from __future__ import annotations

import base64
import json
import re
from typing import Any, Callable

REFERENCE = re.compile(r"@@@langfuseMedia:type=([^|@]+)\|id=([^|@]+)\|source=[^@]+@@@")


def _replace(value: Any, swaps: dict[str, str]) -> Any:
    """A copy of the JSON value *value* with each key of *swaps* replaced."""
    if not swaps:
        return value
    text = json.dumps(value, ensure_ascii=False)
    for old, new in swaps.items():
        text = text.replace(old, new)
    return json.loads(text)


class Media:
    """Reference -> ``data:`` URI of the images the entries refer to."""

    def __init__(self) -> None:
        self._uris: dict[str, str] = {}

    def load(self, entries: list[dict], fetch: Callable[[str], bytes]) -> None:
        """Download each image the entries' sources refer to (*fetch*: id -> bytes)."""
        text = json.dumps([e["source"] for e in entries], ensure_ascii=False)
        for match in REFERENCE.finditer(text):
            if match[0] not in self._uris:
                data = base64.b64encode(fetch(match[2])).decode()
                self._uris[match[0]] = f"data:{match[1]};base64,{data}"

    def inline(self, value: Any) -> Any:
        """*value* with every loaded reference replaced by its image."""
        return _replace(value, self._uris)

    def restore(self, value: Any) -> Any:
        """*value* with every inlined image replaced by its reference again."""
        return _replace(value, {uri: ref for ref, uri in self._uris.items()})
