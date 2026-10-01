import asyncio
import json
from typing import Any, cast

import pytest
from starlette.websockets import WebSocket

from packages.voice.browser_audio import BrowserAudio, Microphone
from packages.voice.speaker import Speaker
from providers.testing.fakes import FakeSynthesizer

THREE_SENTENCES = "First sentence. Second sentence. Third sentence."
SENTENCE_SAMPLES = 1440  # two words at 0.03 s a word, 24 kHz


class FakeTab:
    """The browser tab on the other end of the audio websocket."""

    def __init__(self) -> None:
        self.incoming: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self.sent: list[bytes | str] = []

    async def accept(self) -> None:
        pass

    async def close(self) -> None:
        self.leave()

    async def receive(self) -> dict[str, Any]:
        return await self.incoming.get()

    async def send_bytes(self, data: bytes) -> None:
        self.sent.append(data)

    async def send_text(self, data: str) -> None:
        self.sent.append(data)

    def report(self, kind: str, utterance: int, samples: int) -> None:
        message = {"type": kind, "utterance": utterance, "samples": samples}
        self.incoming.put_nowait({"type": "websocket.receive", "text": json.dumps(message)})

    def leave(self) -> None:
        self.incoming.put_nowait({"type": "websocket.disconnect"})

    def stops(self) -> list[dict[str, Any]]:
        return [json.loads(item) for item in self.sent if isinstance(item, str)]


async def ignore(microphone: Microphone) -> None:
    while await microphone.get() is not None:
        pass


@pytest.fixture
async def connected():
    audio = BrowserAudio(ignore)
    tab = FakeTab()
    serving = asyncio.create_task(audio.serve(cast(WebSocket, tab)))
    await asyncio.sleep(0)
    yield audio, tab
    tab.leave()
    await serving


async def speak(audio: BrowserAudio, text: str):
    utterance = Speaker(FakeSynthesizer(seconds_per_word=0.03), audio).utterance(lambda _: None)
    utterance.say(text)
    finishing = asyncio.create_task(utterance.finish())
    await asyncio.sleep(0.05)
    return utterance, finishing


async def test_audio_the_tab_reports_as_played_is_heard_in_full(connected) -> None:
    audio, tab = connected
    _, finishing = await speak(audio, THREE_SENTENCES)

    tab.report("played", 1, 3 * SENTENCE_SAMPLES)
    heard = await finishing

    assert (heard.text, heard.interrupted) == (THREE_SENTENCES, False)


async def test_an_interrupt_keeps_what_the_tab_says_it_played(connected) -> None:
    audio, tab = connected
    utterance, finishing = await speak(audio, THREE_SENTENCES)

    utterance.interrupt()
    await asyncio.sleep(0.01)
    assert tab.stops() == [{"type": "stop", "utterance": 1}]
    tab.report("stopped", 1, SENTENCE_SAMPLES)
    heard = await finishing

    assert (heard.text, heard.interrupted) == ("First sentence. ", True)


async def test_closing_the_tab_mid_reply_keeps_only_what_had_played(connected) -> None:
    audio, tab = connected
    _, finishing = await speak(audio, THREE_SENTENCES)

    tab.report("played", 1, SENTENCE_SAMPLES + SENTENCE_SAMPLES // 2 + 1)
    tab.leave()
    heard = await finishing

    assert (heard.text, heard.interrupted) == ("First sentence. Second", True)
