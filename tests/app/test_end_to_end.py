"""The whole app on fakes: the student asks for a diagram and it ends up on the whiteboard."""

import asyncio

from apps.voice_tutor import fake
from events.bus import Bus
from events.topics import BoardChanged, Drawn, StudentTyped
from packages.voice.playback import ClockPlayback
from packages.voice.speaker import Speaker
from providers.testing.fakes import FakeSynthesizer, ScriptedChatModel
from services.diagrammer.service import Diagrammer
from services.tutor.service import Tutor
from services.whiteboard.service import Whiteboard


async def test_a_requested_diagram_ends_up_on_the_whiteboard() -> None:
    bus = Bus()
    Whiteboard(bus)
    tutor = Tutor(
        bus,
        model=ScriptedChatModel(fake.tutor_reply),
        voice=Speaker(FakeSynthesizer(seconds_per_word=0.01), ClockPlayback()),
    )
    diagrammer = Diagrammer(bus, model=ScriptedChatModel(fake.diagram_reply, delay_s=0.05))
    boards: list[BoardChanged] = []
    bus.subscribe(BoardChanged, "ui", boards.append)
    tutor.start()
    diagrammer.start()

    bus.publisher(StudentTyped, "ui").publish(StudentTyped("Can you draw a diagram of a heap?"))
    async with asyncio.timeout(5):
        while not boards or not isinstance(boards[-1].items[0].outcome, Drawn):
            await asyncio.sleep(0.01)

    item = boards[-1].items[0]
    assert item.description == "Can you draw a diagram of a heap?"
    assert [board.change for board in boards] == ["diagram_requested", "diagram_finished"]
    await tutor.aclose()
    await diagrammer.aclose()
