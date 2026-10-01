from __future__ import annotations

import asyncio
import re

from events.topics import DiagramRequested, Drawn, Failed
from providers.llm.base import ChatModel, Message, Text, TextDelta
from services.shared.runtime.fence import Fence
from services.shared.runtime.policy import Turn

_TITLE = re.compile(r"^\s*Title:\s*(.+)$", re.MULTILINE)
_MERMAID = re.compile(r"```mermaid\s*\n(.*?)```", re.DOTALL)


class DiagramTurn(Turn[Drawn | Failed]):
    """Draws one requested diagram."""

    def __init__(self, request: DiagramRequested, *, model: ChatModel, system: str) -> None:
        super().__init__(request.request_id)
        self._request = request
        self._model = model
        self._system = system
        self._work: asyncio.Task[str] | None = None
        self._stopping = False

    async def run(self, fence: Fence) -> Drawn | Failed:
        if self._stopping:
            return Failed("Drawing was stopped.")
        self._work = asyncio.create_task(self._write())
        try:
            answer = await self._work
        except asyncio.CancelledError:
            if (task := asyncio.current_task()) is not None and task.cancelling():
                raise
            return Failed("Drawing was stopped.")
        return parse_diagram(answer)

    def request_stop(self) -> None:
        self._stopping = True
        if self._work is not None:
            self._work.cancel()

    async def _write(self) -> str:
        answer = ""
        request = Message("user", (Text(self._request.description),))
        async for event in self._model.stream(system=self._system, messages=(request,)):
            if isinstance(event, TextDelta):
                answer += event.text
        return answer


def parse_diagram(answer: str) -> Drawn | Failed:
    source = _MERMAID.search(answer)
    if source is None:
        return Failed("The diagram agent did not return a Mermaid diagram.")
    title = _TITLE.search(answer)
    return Drawn(title.group(1).strip() if title else "Diagram", source.group(1).strip())
