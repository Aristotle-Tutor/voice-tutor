"""Streaming text-to-speech.

Text goes in one speakable chunk at a time, usually a sentence. Audio comes
out as 24 kHz mono 16-bit PCM, streamed as it is synthesized.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

TTS_SAMPLE_RATE = 24_000


class SpeechSynthesizer(ABC):
    @abstractmethod
    def synthesize(self, text: str) -> AsyncIterator[bytes]:
        """Stream PCM for `text`.

        Closing the iterator early stops synthesis. Vendor failures raise
        `ProviderError`.
        """
