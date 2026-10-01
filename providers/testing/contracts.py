"""One behavioral contract per provider kind, run against every implementation, fakes included.

Each `*_CHECKS` tuple holds `(name, check)` pairs. A check takes the subject and
fails with `AssertionError` when the subject breaks the contract.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import dataclass

from providers.llm.base import (
    CallsTools,
    ChatModel,
    Finish,
    Message,
    StreamEvent,
    Text,
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
)
from providers.tts.base import TTS_SAMPLE_RATE, SpeechSynthesizer

type Check[S] = tuple[str, Callable[[S], Awaitable[None]]]

_PROMPTLY_S = 2.0

_SYSTEM = "You are a friendly tutor."
_TOOL_SYSTEM = "You are a friendly tutor. Use a tool whenever one helps."
_WEATHER = Tool(
    "get_weather",
    "Get the current weather for a city.",
    {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
)
_SENTENCE = "Hello! Let's work through this problem together."
_PARAGRAPH = " ".join([_SENTENCE] * 6)


@dataclass(frozen=True)
class Utterance:
    pcm: bytes
    """16 kHz mono PCM16 speech."""
    text: str


def _user(text: str) -> Message:
    return Message("user", (Text(text),))


async def _collect[T](stream: AsyncIterator[T]) -> list[T]:
    return [item async for item in stream]


def _reply_text(events: Sequence[StreamEvent], finish: Finish) -> str:
    assert events and events[-1] == finish, f"expected to end with {finish}: {events}"
    assert not any(isinstance(e, Finish) for e in events[:-1]), f"more than one Finish: {events}"
    return "".join(e.text for e in events if isinstance(e, TextDelta))


async def _stops_when_closed(stream: AsyncIterator[object]) -> None:
    assert isinstance(stream, AsyncGenerator), "a stream must be closable with aclose()"
    await anext(stream)
    async with asyncio.timeout(_PROMPTLY_S):
        await stream.aclose()


async def _stops_when_cancelled(stream: AsyncIterator[object]) -> None:
    pending = asyncio.ensure_future(anext(stream))
    await asyncio.sleep(0)
    pending.cancel()
    await asyncio.wait({pending}, timeout=_PROMPTLY_S)
    assert pending.cancelled(), "cancelling a waiting consumer must raise CancelledError"


async def _streams_text_then_finishes(model: ChatModel) -> None:
    events = await _collect(
        model.stream(system=_SYSTEM, messages=[_user("Say hello in one short sentence.")])
    )
    assert _reply_text(events, Finish("end_turn")).strip()


async def _accepts_ragged_history(model: ChatModel) -> None:
    messages = [
        _user("Hi! I'm studying fractions."),
        _user("Can you help me?"),
        Message("assistant", (Text("Of course. A fraction has a numerator, which is the "),)),
        Message("assistant", (Text(""),)),
        _user("Sorry, I cut you off. What is 1/2 + 1/4? Answer briefly."),
    ]
    events = await _collect(model.stream(system=_SYSTEM, messages=messages))
    assert _reply_text(events, Finish("end_turn")).strip()


def _count(model: ChatModel) -> AsyncIterator[StreamEvent]:
    return model.stream(system=_SYSTEM, messages=[_user("Count from one to fifty in words.")])


CHAT_CHECKS: tuple[Check[ChatModel], ...] = (
    ("streams text then finishes", _streams_text_then_finishes),
    ("accepts same-role runs, empty parts, a cut-off reply", _accepts_ragged_history),
    ("stops when closed", lambda model: _stops_when_closed(_count(model))),
    ("stops when cancelled", lambda model: _stops_when_cancelled(_count(model))),
)


async def _calls_a_tool_then_answers(model: CallsTools) -> None:
    messages = [_user("What's the weather in Paris right now?")]
    events = await _collect(
        model.stream_with_tools(system=_TOOL_SYSTEM, messages=messages, tools=[_WEATHER])
    )
    said = _reply_text(events, Finish("tool_use"))
    calls = [e for e in events if isinstance(e, ToolCall)]
    assert calls, f"expected a tool call: {events}"
    assert all(c.name == "get_weather" and "city" in c.arguments for c in calls), calls
    for call in calls:
        started = events.index(ToolCallStarted(call.call_id, call.name))
        assert started < events.index(call), f"{call.call_id} was not announced first: {events}"
    messages += [
        Message("assistant", (Text(said), *calls)),
        Message("user", tuple(ToolResult(c.call_id, "18°C and sunny") for c in calls)),
    ]
    events = await _collect(
        model.stream_with_tools(system=_TOOL_SYSTEM, messages=messages, tools=[_WEATHER])
    )
    assert _reply_text(events, Finish("end_turn")).strip()


TOOL_CHECKS: tuple[Check[CallsTools], ...] = (
    ("calls a tool, then answers from its result", _calls_a_tool_then_answers),
)


async def _synthesizes_whole_samples(synthesizer: SpeechSynthesizer) -> None:
    chunks = await _collect(synthesizer.synthesize(_SENTENCE))
    assert chunks and all(c and len(c) % 2 == 0 for c in chunks), "chunks must be whole samples"
    seconds = sum(map(len, chunks)) / 2 / TTS_SAMPLE_RATE
    assert 1 < seconds < 10, f"{seconds:.1f} s of audio for a seven-word sentence"


TTS_CHECKS: tuple[Check[SpeechSynthesizer], ...] = (
    ("synthesizes whole samples of plausible length", _synthesizes_whole_samples),
    ("stops when closed", lambda tts: _stops_when_closed(tts.synthesize(_PARAGRAPH))),
    ("stops when cancelled", lambda tts: _stops_when_cancelled(tts.synthesize(_PARAGRAPH))),
)


async def _send_in_real_time(stream: RecognizerStream, pcm: bytes) -> None:
    frame = 2 * STT_SAMPLE_RATE // 50
    for start in range(0, len(pcm), frame):
        await stream.send(pcm[start : start + frame])
        await asyncio.sleep(0.02)


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9']+", text.lower()))


async def _hears_partials_then_commits(recognizer: SpeechRecognizer, speech: Utterance) -> None:
    one_second_of_silence = bytes(2 * STT_SAMPLE_RATE)
    async with recognizer.connect() as stream:
        sender = asyncio.create_task(_send_in_real_time(stream, speech.pcm + one_second_of_silence))
        heard: list[Partial | Committed] = []
        try:
            async with asyncio.timeout(15):
                async for event in stream.events():
                    heard.append(event)
                    if isinstance(event, Committed):
                        break
            await sender
        finally:
            sender.cancel()
    assert any(isinstance(e, Partial) for e in heard), f"no partial before the commit: {heard}"
    committed = heard[-1]
    assert isinstance(committed, Committed), heard
    assert _words(speech.text) <= _words(committed.text), committed


async def _ends_cleanly_mid_utterance(recognizer: SpeechRecognizer, speech: Utterance) -> None:
    async with recognizer.connect() as stream:
        listener = asyncio.create_task(_collect(stream.events()))
        await _send_in_real_time(stream, speech.pcm[: len(speech.pcm) // 2])
    async with asyncio.timeout(_PROMPTLY_S):
        await listener


STT_CHECKS: tuple[tuple[str, Callable[[SpeechRecognizer, Utterance], Awaitable[None]]], ...] = (
    ("hears partials, then commits after silence", _hears_partials_then_commits),
    ("ends events cleanly when closed mid-utterance", _ends_cleanly_mid_utterance),
)
