from __future__ import annotations

from pathlib import Path

from events.bus import Bus
from events.topics import (
    BoardChanged,
    DiagramFinished,
    DiagramRequested,
    StudentTyped,
    StudentVoice,
    TutorSpeech,
)
from providers.llm.base import CallsTools, Message
from services.shared.runtime.loop import Loop
from services.tutor.policy import TutorPolicy
from services.tutor.ports import Voice
from services.tutor.tools import TutorTools
from services.tutor.turn import TutorSettlement, TutorTurn

PROMPT_PATH = Path(__file__).with_name("prompt.md")


class Tutor:
    """The tutor agent: its policy on a loop, fed by the bus."""

    def __init__(self, bus: Bus, *, model: CallsTools, voice: Voice) -> None:
        system = PROMPT_PATH.read_text()
        tools = TutorTools(bus.publisher(DiagramRequested, "tutor"))
        captions = bus.publisher(TutorSpeech, "tutor")

        def make_turn(turn_id: str, input: Message, history: tuple[Message, ...]) -> TutorTurn:
            return TutorTurn(
                turn_id,
                model=model,
                system=system,
                history=history,
                input=input,
                tools=tools,
                voice=voice,
                captions=captions,
            )

        self.loop: Loop[TutorSettlement] = Loop("tutor", TutorPolicy(make_turn))
        for topic in (StudentVoice, StudentTyped, BoardChanged, DiagramFinished):
            bus.subscribe(topic, "tutor", self.loop.enqueue)

    def start(self) -> None:
        self.loop.start()

    async def aclose(self) -> None:
        await self.loop.aclose()
