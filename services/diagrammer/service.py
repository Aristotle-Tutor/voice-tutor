from __future__ import annotations

from pathlib import Path

from events.bus import Bus
from events.topics import DiagramFinished, DiagramRequested
from providers.llm.base import ChatModel
from services.diagrammer.policy import DiagrammerPolicy, Outcome
from services.diagrammer.turn import DiagramTurn
from services.shared.runtime.loop import Loop

PROMPT_PATH = Path(__file__).with_name("prompt.md")


class Diagrammer:
    """The diagram agent: its policy on a loop, fed by the bus."""

    def __init__(self, bus: Bus, *, model: ChatModel) -> None:
        system = PROMPT_PATH.read_text()

        def make_turn(request: DiagramRequested) -> DiagramTurn:
            return DiagramTurn(request, model=model, system=system)

        policy = DiagrammerPolicy(make_turn, bus.publisher(DiagramFinished, "diagrammer"))
        self.loop: Loop[Outcome] = Loop("diagrammer", policy)
        bus.subscribe(DiagramRequested, "diagrammer", self.loop.enqueue)

    def start(self) -> None:
        self.loop.start()

    async def aclose(self) -> None:
        await self.loop.aclose()
