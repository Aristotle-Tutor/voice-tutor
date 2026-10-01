import asyncio
from collections.abc import AsyncIterator

from packages.voice.playback import ClockPlayback
from packages.voice.speaker import Speaker
from providers.testing.fakes import FakeSynthesizer
from providers.tts.base import SpeechSynthesizer

LONG_REPLY = (
    "A heap is a tree where every parent is smaller than its children. "
    "That makes the smallest item easy to find, because it is always at the top. "
    "Adding an item means placing it at the bottom and swapping it upward."
)


def speaker() -> Speaker:
    return Speaker(FakeSynthesizer(seconds_per_word=0.03), ClockPlayback())


async def test_an_uninterrupted_reply_is_heard_in_full_as_it_plays() -> None:
    pieces: list[str] = []
    utterance = speaker().utterance(pieces.append)
    utterance.say("One two three. ")
    utterance.say("Four five.")

    heard = await utterance.finish()

    assert heard.text == "One two three. Four five."
    assert not heard.interrupted
    assert len(pieces) > 1
    assert "".join(pieces) == heard.text


async def test_an_interrupt_leaves_what_had_played_cut_at_a_word() -> None:
    pieces: list[str] = []
    utterance = speaker().utterance(pieces.append)
    utterance.say(LONG_REPLY)
    finishing = asyncio.create_task(utterance.finish())
    await asyncio.sleep(0.5)

    utterance.interrupt()
    heard = await finishing

    assert "".join(pieces) == heard.text
    assert heard.interrupted
    assert 0 < len(heard.text) < len(LONG_REPLY)
    assert LONG_REPLY.startswith(heard.text)
    assert LONG_REPLY[len(heard.text)] == " "


async def test_text_said_after_an_interrupt_is_ignored() -> None:
    utterance = speaker().utterance(lambda _: None)
    utterance.interrupt()
    utterance.say("Too late.")

    heard = await utterance.finish()

    assert heard.text == ""
    assert heard.interrupted


class SlowToStart(SpeechSynthesizer):
    """Waits before its first audio, like a vendor's time to first byte."""

    def __init__(self, delay_s: float) -> None:
        self._delay_s = delay_s
        self._inner = FakeSynthesizer(seconds_per_word=0.03)

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        await asyncio.sleep(self._delay_s)
        async for pcm in self._inner.synthesize(text):
            yield pcm


async def test_a_sentence_whose_audio_has_not_arrived_is_not_heard() -> None:
    utterance = Speaker(SlowToStart(0.3), ClockPlayback()).utterance(lambda _: None)
    utterance.say("Hello there. ")
    await asyncio.sleep(0.05)

    utterance.interrupt()
    heard = await utterance.finish()

    assert (heard.text, heard.interrupted) == ("", True)
