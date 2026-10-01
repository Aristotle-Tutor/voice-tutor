"""Keyless stand-ins for every provider kind, for tests and the app's fake mode."""

from __future__ import annotations

import asyncio
import re
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from typing import cast

from providers.llm.base import (
    CallsTools,
    ChatModel,
    Finish,
    Message,
    StreamEvent,
    TextDelta,
    Tool,
    ToolCall,
    ToolCallStarted,
    ToolResult,
)
from providers.stt.base import (
    STT_SAMPLE_RATE,
    Committed,
    Partial,
    RecognizerStream,
    SpeechRecognizer,
    SttEvent,
)
from providers.tts.base import TTS_SAMPLE_RATE, SpeechSynthesizer

type Respond = Callable[[Sequence[Message], Sequence[Tool]], Sequence[StreamEvent]]

_SAMPLE_VALUES: dict[object, object] = {
    "string": "test",
    "number": 1,
    "integer": 1,
    "boolean": True,
}


class ScriptedChatModel(ChatModel, CallsTools):
    """Streams the events `respond(messages, tools)` returns, `delay_s` apart.

    By default it calls the first offered tool (unless the last message holds
    tool results), and otherwise replies "This is a scripted reply."
    """

    def __init__(self, respond: Respond | None = None, *, delay_s: float = 0.0) -> None:
        self._respond = respond or _respond_like_a_test_model
        self._delay_s = delay_s

    def stream(self, *, system: str, messages: Sequence[Message]) -> AsyncIterator[StreamEvent]:
        return self._stream(messages, ())

    def stream_with_tools(
        self,
        *,
        system: str,
        messages: Sequence[Message],
        tools: Sequence[Tool],
    ) -> AsyncIterator[StreamEvent]:
        return self._stream(messages, tools)

    async def _stream(
        self, messages: Sequence[Message], tools: Sequence[Tool]
    ) -> AsyncIterator[StreamEvent]:
        for event in self._respond(messages, tools):
            await asyncio.sleep(self._delay_s)
            if isinstance(event, ToolCall):
                yield ToolCallStarted(event.call_id, event.name)
            yield event


def _respond_like_a_test_model(
    messages: Sequence[Message], tools: Sequence[Tool]
) -> list[StreamEvent]:
    answered = bool(messages) and any(isinstance(p, ToolResult) for p in messages[-1].parts)
    if tools and not answered:
        tool = tools[0]
        call = ToolCall(f"call-{len(messages)}", tool.name, _sample_arguments(tool.parameters))
        return [call, Finish("tool_use")]
    words = re.findall(r" ?\S+", "This is a scripted reply.")
    return [*(TextDelta(word) for word in words), Finish("end_turn")]


def _sample_arguments(schema: Mapping[str, object]) -> dict[str, object]:
    properties = cast("Mapping[str, Mapping[str, object]]", schema.get("properties", {}))
    return {
        name: _SAMPLE_VALUES[kind]
        for name, spec in properties.items()
        if (kind := spec.get("type")) in _SAMPLE_VALUES
    }


class FakeSynthesizer(SpeechSynthesizer):
    """Silence lasting `seconds_per_word` per word, in ~100 ms chunks."""

    def __init__(self, seconds_per_word: float = 0.3) -> None:
        self._seconds_per_word = seconds_per_word

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        total = 2 * round(len(text.split()) * self._seconds_per_word * TTS_SAMPLE_RATE)
        chunk = 2 * TTS_SAMPLE_RATE // 10
        for start in range(0, total, chunk):
            # Suspend like a network read, so a consumer can be cancelled mid-stream.
            await asyncio.sleep(0)
            yield bytes(min(chunk, total - start))


class FakeRecognizer(SpeechRecognizer):
    """Hears `transcript` in any non-silent audio.

    It adds a word to the `Partial` for every ~0.3 s of sound, and commits the
    whole transcript after 0.5 s of all-zero audio follows the sound.
    """

    def __init__(self, transcript: str) -> None:
        self._transcript = transcript

    @asynccontextmanager
    async def connect(self) -> AsyncIterator[RecognizerStream]:
        stream = _FakeRecognizerStream(self._transcript)
        try:
            yield stream
        finally:
            stream.close()


class _FakeRecognizerStream(RecognizerStream):
    def __init__(self, transcript: str) -> None:
        self._transcript = transcript
        self._words = transcript.split()
        self._events: asyncio.Queue[SttEvent | None] = asyncio.Queue()
        self._sound_s = 0.0
        self._silence_s = 0.0

    async def send(self, pcm: bytes) -> None:
        seconds = len(pcm) / 2 / STT_SAMPLE_RATE
        if any(pcm):
            heard = self._words_heard()
            self._sound_s += seconds
            self._silence_s = 0.0
            if self._words_heard() > heard:
                self._events.put_nowait(Partial(" ".join(self._words[: self._words_heard()])))
        elif self._sound_s:
            self._silence_s += seconds
            if self._silence_s >= 0.5:
                self._events.put_nowait(Committed(self._transcript))
                self._sound_s = self._silence_s = 0.0

    async def events(self) -> AsyncIterator[SttEvent]:
        while (event := await self._events.get()) is not None:
            yield event

    def close(self) -> None:
        self._events.put_nowait(None)

    def _words_heard(self) -> int:
        return min(len(self._words), 1 + int(self._sound_s / 0.3)) if self._sound_s else 0
