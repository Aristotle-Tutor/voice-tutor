import asyncio
from contextlib import AbstractAsyncContextManager

from events.topics import StudentVoice
from packages.voice.listener import Listener
from providers.stt.base import RecognizerStream, SpeechRecognizer
from providers.testing.fakes import FakeRecognizer

FRAME_SAMPLES = 320
SPEECH = b"\x00\x10" * FRAME_SAMPLES
SILENCE = b"\x00\x00" * FRAME_SAMPLES


def frames(seconds: float, frame: bytes) -> list[bytes]:
    return [frame] * int(seconds * 50)


async def listen(*audio: bytes) -> list[StudentVoice]:
    heard: list[StudentVoice] = []
    microphone: asyncio.Queue[bytes | None] = asyncio.Queue()
    for frame in audio:
        microphone.put_nowait(frame)
    microphone.put_nowait(None)
    await Listener(FakeRecognizer("tell me about heaps"), heard.append).listen(microphone)
    return heard


async def test_an_utterance_starts_with_its_first_words_and_ends_with_its_final_text() -> None:
    heard = await listen(*frames(1.5, SPEECH), *frames(0.6, SILENCE))

    assert heard[0].kind == "started"
    assert heard[0].text
    assert {event.kind for event in heard[1:-1]} <= {"interim"}
    assert heard[-1] == StudentVoice(heard[0].utterance_id, "final", "tell me about heaps")


async def test_an_utterance_cut_off_by_the_microphone_still_ends() -> None:
    heard = await listen(*frames(0.8, SPEECH))

    assert heard[0].kind == "started"
    assert heard[-1].kind == "final"
    assert heard[-1].text == heard[-2].text


class FailsOnce(SpeechRecognizer):
    """Fails its first connection with an unexpected error, then behaves."""

    def __init__(self) -> None:
        self._inner = FakeRecognizer("tell me about heaps")
        self.connections = 0

    def connect(self) -> AbstractAsyncContextManager[RecognizerStream]:
        self.connections += 1
        if self.connections == 1:
            raise RuntimeError("unexpected message from the recognizer")
        return self._inner.connect()


async def test_listening_recovers_from_an_unexpected_recognizer_error() -> None:
    heard: list[StudentVoice] = []
    microphone: asyncio.Queue[bytes | None] = asyncio.Queue()
    recognizer = FailsOnce()
    listening = asyncio.create_task(Listener(recognizer, heard.append).listen(microphone))

    await asyncio.sleep(1.2)
    for frame in (*frames(1.5, SPEECH), *frames(0.6, SILENCE), None):
        microphone.put_nowait(frame)
    await listening

    assert recognizer.connections == 2
    assert heard[-1].kind == "final"
    assert heard[-1].text == "tell me about heaps"
