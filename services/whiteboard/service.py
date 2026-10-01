from __future__ import annotations

from typing import Literal

from events.bus import Bus
from events.topics import (
    BoardChanged,
    BoardClearRequested,
    BoardItem,
    DiagramFinished,
    DiagramRequested,
)


class Whiteboard:
    """Owns what is on the board. Every change is published as a new revision."""

    def __init__(self, bus: Bus) -> None:
        self._items: dict[str, BoardItem] = {}
        self._revision = 0
        self._changed = bus.publisher(BoardChanged, "whiteboard")
        bus.subscribe(DiagramRequested, "whiteboard", self._diagram_requested)
        bus.subscribe(DiagramFinished, "whiteboard", self._diagram_finished)
        bus.subscribe(BoardClearRequested, "whiteboard", self._clear)

    def _diagram_requested(self, request: DiagramRequested) -> None:
        self._items[request.request_id] = BoardItem(request.request_id, request.description, None)
        self._publish("diagram_requested")

    def _diagram_finished(self, finished: DiagramFinished) -> None:
        requested = self._items.get(finished.request_id)
        description = requested.description if requested else ""
        self._items[finished.request_id] = BoardItem(
            finished.request_id, description, finished.outcome
        )
        self._publish("diagram_finished")

    def _clear(self, _: BoardClearRequested) -> None:
        self._items.clear()
        self._publish("cleared")

    def _publish(self, change: Literal["diagram_requested", "diagram_finished", "cleared"]) -> None:
        self._revision += 1
        self._changed.publish(BoardChanged(self._revision, tuple(self._items.values()), change))
