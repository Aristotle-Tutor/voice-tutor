from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from providers.tts.base import TTS_SAMPLE_RATE


class Playback(ABC):
    """Somewhere the tutor's audio plays, which reports how much has actually been played.

    Audio belongs to an utterance, identified by a number. Only one utterance
    plays at a time.
    """

    @abstractmethod
    def write(self, utterance: int, pcm: bytes) -> None:
        """Queue audio to play after what was already written."""

    @abstractmethod
    async def drain(self, utterance: int) -> bool:
        """Wait until all audio written for the utterance has played.

        Returns False if playback ended first, for example because it was stopped.
        """

    @abstractmethod
    async def stop(self, utterance: int) -> int:
        """Drop the utterance's unplayed audio and return how many samples were played."""

    @abstractmethod
    def played(self, utterance: int) -> int:
        """How many samples of the utterance have played so far."""


@dataclass
class _Clock:
    written: int = 0
    played: float = 0.0
    updated_at: float = field(default_factory=time.monotonic)
    stopped: bool = False

    def advance(self) -> None:
        now = time.monotonic()
        if not self.stopped:
            self.played = min(self.written, self.played + (now - self.updated_at) * TTS_SAMPLE_RATE)
        self.updated_at = now


class ClockPlayback(Playback):
    """Pretends to play audio in real time, for when no speaker is connected."""

    def __init__(self) -> None:
        self._clocks: dict[int, _Clock] = {}

    def write(self, utterance: int, pcm: bytes) -> None:
        clock = self._clocks.setdefault(utterance, _Clock())
        clock.advance()
        clock.written += len(pcm) // 2

    async def drain(self, utterance: int) -> bool:
        clock = self._clocks.setdefault(utterance, _Clock())
        while True:
            clock.advance()
            if clock.stopped or clock.played >= clock.written:
                return clock.played >= clock.written
            await asyncio.sleep((clock.written - clock.played) / TTS_SAMPLE_RATE)

    async def stop(self, utterance: int) -> int:
        clock = self._clocks.setdefault(utterance, _Clock())
        clock.advance()
        clock.stopped = True
        return int(clock.played)

    def played(self, utterance: int) -> int:
        clock = self._clocks.setdefault(utterance, _Clock())
        clock.advance()
        return int(clock.played)
