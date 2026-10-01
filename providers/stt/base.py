"""Streaming speech-to-text.

Audio goes in as 16 kHz mono 16-bit PCM. Transcripts come out as partials
(the recognizer's current guess at the words so far in this utterance) and a
committed transcript once the recognizer decides the utterance has ended.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass

STT_SAMPLE_RATE = 16_000


@dataclass(frozen=True)
class Partial:
    text: str
    """Everything heard so far in the current utterance, not just the new words."""


@dataclass(frozen=True)
class Committed:
    text: str
    """The final transcript of an utterance that has ended."""


type SttEvent = Partial | Committed


class RecognizerStream(ABC):
    @abstractmethod
    async def send(self, pcm: bytes) -> None: ...

    @abstractmethod
    def events(self) -> AsyncIterator[SttEvent]: ...


class SpeechRecognizer(ABC):
    @abstractmethod
    def connect(self) -> AbstractAsyncContextManager[RecognizerStream]:
        """Open one streaming session. Vendor failures raise `ProviderError`."""
