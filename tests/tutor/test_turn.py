from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence

import pytest

from events.bus import Bus
from events.topics import DiagramRequested, TutorSpeech
from packages.voice.playback import ClockPlayback
from packages.voice.speaker import Heard, Speaker
from providers.errors import ProviderError
from providers.llm.base import (
    Finish,
    Message,
    StreamEvent,
    Text,
    TextDelta,
    Tool,
    ToolCall,
    ToolResult,
)
from providers.testing.fakes import FakeSynthesizer, ScriptedChatModel
from services.shared.runtime.fence import Fence
from services.tutor.ports import Voice
from services.tutor.tools import TutorTools
from services.tutor.turn import TutorTurn

INPUT = Message("user", (Text("Can you explain heaps?"),))
LONG_REPLY = (
    "A heap is a tree where every parent is smaller than its children. "
    "That makes the smallest item easy to find, because it is always at the top. "
    "Adding an item means placing it at the bottom and swapping it upward."
)


def say(text: str) -> list[StreamEvent]:
    return [*(TextDelta(word + " ") for word in text.split()), Finish("end_turn")]


class RecordingVoice:
    """Remembers what it was asked to say, and reports all of it as heard."""

    def __init__(self) -> None:
        self.said = ""

    def utterance(self, on_heard: Callable[[str], None]) -> RecordingVoice:
        return self

    def say(self, text: str) -> None:
        self.said += text

    def interrupt(self) -> None:
        pass

    async def finish(self) -> Heard:
        return Heard(self.said, interrupted=False)


class Harness:
    def __init__(self, respond, voice: Voice | None = None) -> None:
        bus = Bus()
        self.captions: list[TutorSpeech] = []
        self.requests: list[DiagramRequested] = []
        bus.subscribe(TutorSpeech, "ui", self.captions.append)
        bus.subscribe(DiagramRequested, "diagrammer", self.requests.append)
        self.turn = TutorTurn(
            "turn-1",
            model=ScriptedChatModel(respond, delay_s=0.005),
            system="You are a tutor.",
            history=(),
            input=INPUT,
            tools=TutorTools(bus.publisher(DiagramRequested, "tutor")),
            voice=voice or Speaker(FakeSynthesizer(seconds_per_word=0.03), ClockPlayback()),
            captions=bus.publisher(TutorSpeech, "tutor"),
        )

    def spoken(self) -> str:
        return "".join(caption.text for caption in self.captions if caption.kind == "text")


async def test_a_reply_is_spoken_in_full_and_recorded() -> None:
    harness = Harness(lambda messages, tools: say("Heaps are trees. They keep order."))

    settlement = await harness.turn.run(Fence())

    assert settlement.messages == (
        INPUT,
        Message("assistant", (Text("Heaps are trees. They keep order. "),)),
    )
    assert harness.captions[-1] == TutorSpeech("turn-1", "ended", harness.spoken(), False)


async def test_a_tool_call_requests_a_diagram_and_the_model_continues_with_the_result() -> None:
    voice = RecordingVoice()
    said_before_second_call: list[str] = []

    def respond(messages: Sequence[Message], tools: Sequence[Tool]) -> list[StreamEvent]:
        if any(isinstance(part, ToolResult) for part in messages[-1].parts):
            said_before_second_call.append(voice.said)
            return say("Here it comes.")
        call = ToolCall("c1", "draw_diagram", {"description": "a min-heap"})
        return [TextDelta("Let me draw that."), call, Finish("tool_use")]

    harness = Harness(respond, voice)

    settlement = await harness.turn.run(Fence())

    assert harness.requests == [DiagramRequested("d1", "a min-heap")]
    assert said_before_second_call == ["Let me draw that. "]
    call, result, text = settlement.messages[1:]
    assert call.parts == (
        Text("Let me draw that. "),
        ToolCall("c1", "draw_diagram", {"description": "a min-heap"}),
    )
    assert isinstance(result.parts[0], ToolResult) and not result.parts[0].is_error
    assert text == Message("assistant", (Text("Here it comes. "),))


async def test_a_stop_cuts_the_recorded_reply_to_what_was_heard() -> None:
    harness = Harness(lambda messages, tools: say(LONG_REPLY))
    running = asyncio.create_task(harness.turn.run(Fence()))
    await asyncio.sleep(0.6)

    harness.turn.request_stop()
    settlement = await running

    recorded = settlement.messages[1].parts[0]
    assert isinstance(recorded, Text)
    assert 0 < len(recorded.text) < len(LONG_REPLY)
    assert LONG_REPLY.startswith(recorded.text)
    ended = harness.captions[-1]
    assert ended.kind == "ended" and ended.interrupted and ended.text == recorded.text
    assert harness.spoken() == recorded.text


async def test_a_failed_model_call_still_ends_the_captions() -> None:
    def respond(messages: Sequence[Message], tools: Sequence[Tool]) -> list[StreamEvent]:
        raise ProviderError("anthropic", "overloaded")

    harness = Harness(respond)

    with pytest.raises(ProviderError):
        await harness.turn.run(Fence())

    assert harness.captions[-1] == TutorSpeech("turn-1", "ended", "", True)
