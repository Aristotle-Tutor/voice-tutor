"""Turns microphone audio into StudentVoice events.

The recognizer gives partial transcripts while the student talks and a
committed transcript once it has heard enough silence. The first partial with
words starts an utterance, later partials update it, and the committed
transcript ends it.
"""

from __future__ import annotations

import asyncio
import itertools
import logging
from collections.abc import Callable

from events.topics import StudentVoice
from providers.stt.base import Committed, Partial, RecognizerStream, SpeechRecognizer

logger = logging.getLogger(__name__)

RECONNECT_DELAY_S = 1.0


class Listener:
    def __init__(
        self, recognizer: SpeechRecognizer, publish: Callable[[StudentVoice], None]
    ) -> None:
        self._recognizer = recognizer
        self._publish = publish
        self._ids = itertools.count(1)
        self._utterance: str | None = None
        self._text = ""

    async def listen(self, microphone: asyncio.Queue[bytes | None]) -> None:
        """Transcribe microphone frames until a None frame, reconnecting if the recognizer drops."""
        while True:
            try:
                if await self._listen_once(microphone):
                    return
                logger.warning("speech recognition session ended; reconnecting")
            except Exception:
                logger.exception("speech recognition failed; reconnecting")
                await asyncio.sleep(RECONNECT_DELAY_S)
                if _discard_backlog(microphone):
                    return

    async def _listen_once(self, microphone: asyncio.Queue[bytes | None]) -> bool:
        """Returns True when the microphone ran out, False when the recognizer ended first."""
        try:
            async with self._recognizer.connect() as stream:
                sending = asyncio.create_task(_send(microphone, stream))
                hearing = asyncio.create_task(self._hear(stream))
                try:
                    done, _ = await asyncio.wait(
                        (sending, hearing), return_when=asyncio.FIRST_COMPLETED
                    )
                    for task in done:
                        task.result()
                    return sending in done
                finally:
                    for task in (sending, hearing):
                        task.cancel()
                    await asyncio.gather(sending, hearing, return_exceptions=True)
        finally:
            if self._utterance is not None:
                self._end(self._text)

    async def _hear(self, stream: RecognizerStream) -> None:
        async for event in stream.events():
            match event:
                case Partial(text=text) if text.strip():
                    self._update(text)
                case Committed(text=text):
                    if self._utterance is None and text.strip():
                        self._update(text)
                    if self._utterance is not None:
                        self._end(text)
                case _:
                    pass

    def _update(self, text: str) -> None:
        self._text = text
        if self._utterance is None:
            self._utterance = f"u{next(self._ids)}"
            self._publish(StudentVoice(self._utterance, "started", text))
        else:
            self._publish(StudentVoice(self._utterance, "interim", text))

    def _end(self, text: str) -> None:
        assert self._utterance is not None
        self._publish(StudentVoice(self._utterance, "final", text))
        self._utterance = None


def _discard_backlog(microphone: asyncio.Queue[bytes | None]) -> bool:
    """Drop audio that piled up while reconnecting; recognizers expect it in real time.

    Returns True if the microphone ended meanwhile.
    """
    while not microphone.empty():
        if microphone.get_nowait() is None:
            return True
    return False


async def _send(microphone: asyncio.Queue[bytes | None], stream: RecognizerStream) -> None:
    while (frame := await microphone.get()) is not None:
        await stream.send(frame)
