from __future__ import annotations

import asyncio
import itertools

from events.bus import Publisher
from events.topics import StudentVoice


class SpeechSimulator:
    """Plays typed text as if the student had said it out loud, at speaking pace."""

    WORDS_PER_SECOND = 2.5

    def __init__(self, publish: Publisher[StudentVoice]) -> None:
        self._publish = publish
        self._ids = itertools.count(1)

    async def say(self, text: str) -> None:
        words = text.split()
        if not words:
            return
        utterance = f"simulated-{next(self._ids)}"
        self._publish.publish(StudentVoice(utterance, "started", words[0]))
        for count in range(2, len(words) + 1):
            await asyncio.sleep(1 / self.WORDS_PER_SECOND)
            self._publish.publish(StudentVoice(utterance, "interim", " ".join(words[:count])))
        await asyncio.sleep(0.5)
        self._publish.publish(StudentVoice(utterance, "final", text))
