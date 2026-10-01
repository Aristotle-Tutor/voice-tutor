"""Speaks the tutor's text, and reports what the student actually heard."""

from __future__ import annotations

import asyncio
import itertools
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass

from packages.voice.playback import Playback
from packages.voice.sentences import SentenceBuffer
from providers.tts.base import SpeechSynthesizer

logger = logging.getLogger(__name__)

_WORD = re.compile(r"\S+")

PROGRESS_INTERVAL_S = 0.1


@dataclass(frozen=True)
class Heard:
    text: str
    interrupted: bool


@dataclass
class _Segment:
    text: str
    start: int
    end: int
    synthesized: bool = False


class Speaker:
    def __init__(self, synthesizer: SpeechSynthesizer, playback: Playback) -> None:
        self._synthesizer = synthesizer
        self._playback = playback
        self._ids = itertools.count(1)

    def utterance(self, on_heard: Callable[[str], None]) -> SpokenUtterance:
        return SpokenUtterance(next(self._ids), self._synthesizer, self._playback, on_heard)


class SpokenUtterance:
    """One turn's speech.

    Text is split into sentences, and each sentence is synthesized and queued
    for playback as soon as it is complete. Playback reports how many samples
    were actually played, so after an interrupt the heard text is every
    sentence that played in full, plus the share of the words of the sentence
    that was cut off. A sentence still being synthesized counts as unheard,
    because its length isn't known yet.

    The same accounting runs while the speech plays: `on_heard` gets each newly
    heard piece of text, so captions can follow the voice.
    """

    def __init__(
        self,
        utterance_id: int,
        synthesizer: SpeechSynthesizer,
        playback: Playback,
        on_heard: Callable[[str], None],
    ) -> None:
        self._id = utterance_id
        self._synthesizer = synthesizer
        self._playback = playback
        self._on_heard = on_heard
        self._reported = ""
        self._sentences = SentenceBuffer()
        self._queue: asyncio.Queue[str | None] = asyncio.Queue()
        self._segments: list[_Segment] = []
        self._said = ""
        self._written = 0
        self._ending = False
        self._heard: asyncio.Future[Heard] = asyncio.get_running_loop().create_future()
        self._worker = asyncio.create_task(self._speak())
        self._progress = asyncio.create_task(self._report_progress())
        self._stopper: asyncio.Task[None] | None = None

    @property
    def _interrupted(self) -> bool:
        return self._stopper is not None

    def say(self, text: str) -> None:
        if self._ending or self._interrupted:
            return
        self._said += text
        for sentence in self._sentences.add(text):
            self._queue.put_nowait(sentence)

    def interrupt(self) -> None:
        self._worker.cancel()
        self._stop_playback()

    async def finish(self) -> Heard:
        if not self._ending:
            self._ending = True
            self._queue.put_nowait(self._sentences.flush())
            self._queue.put_nowait(None)
        return await asyncio.shield(self._heard)

    async def _speak(self) -> None:
        try:
            while (sentence := await self._queue.get()) is not None:
                segment = _Segment(sentence, self._written, self._written)
                self._segments.append(segment)
                if sentence.strip():
                    async for pcm in self._synthesizer.synthesize(sentence.strip()):
                        self._playback.write(self._id, pcm)
                        self._written += len(pcm) // 2
                        segment.end = self._written
                segment.synthesized = True
            played_everything = await self._playback.drain(self._id)
        except Exception:
            logger.exception("speech failed; treating it as an interruption")
            played_everything = False
        if not played_everything:
            self._stop_playback()
        else:
            self._settle(Heard(self._said, interrupted=False))

    def _stop_playback(self) -> None:
        if not self._heard.done() and not self._interrupted:
            self._stopper = asyncio.create_task(self._stop())

    async def _stop(self) -> None:
        try:
            played = await self._playback.stop(self._id)
        except Exception:
            logger.exception("could not stop playback; assuming nothing was heard")
            played = 0
        self._settle(Heard(self._heard_text(played), interrupted=True))

    def _settle(self, heard: Heard) -> None:
        if not self._heard.done():
            self._progress.cancel()
            self._report(heard.text)
            self._heard.set_result(heard)

    async def _report_progress(self) -> None:
        while True:
            self._report(self._heard_text(self._playback.played(self._id)))
            await asyncio.sleep(PROGRESS_INTERVAL_S)

    def _report(self, heard: str) -> None:
        if len(heard) > len(self._reported) and heard.startswith(self._reported):
            self._on_heard(heard[len(self._reported) :])
            self._reported = heard

    def _heard_text(self, played: int) -> str:
        heard = ""
        for segment in self._segments:
            if not segment.synthesized:
                break
            if played >= segment.end:
                heard += segment.text
                continue
            if played > segment.start:
                share = (played - segment.start) / (segment.end - segment.start)
                heard += _first_words(segment.text, share)
            break
        return heard


def _first_words(text: str, share: float) -> str:
    words = list(_WORD.finditer(text))
    count = int(len(words) * share)
    return text[: words[count - 1].end()] if count else ""
