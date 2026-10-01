"""The browser tab's microphone and speaker, over one websocket.

From the browser:
    binary frames                 16 kHz mono PCM16 from the microphone
    {"type": "played", ...}       how many samples of an utterance have played so far
    {"type": "stopped", ...}      the same, after a stop took effect
To the browser:
    binary frames                 a little-endian uint32 utterance number, then 24 kHz PCM16
    {"type": "stop", ...}         drop the rest of an utterance

While no tab is connected, the tutor's audio plays on a simulated clock, so
turns still take as long as the speech would.
"""

from __future__ import annotations

import asyncio
import json
import logging
import struct
from collections.abc import Callable, Coroutine
from typing import Any, Literal

from starlette.websockets import WebSocket, WebSocketDisconnect

from packages.voice.playback import ClockPlayback, Playback
from providers.tts.base import TTS_SAMPLE_RATE

logger = logging.getLogger(__name__)

STOP_ACK_TIMEOUT_S = 1.0
DRAIN_GRACE_S = 3.0

type Microphone = asyncio.Queue[bytes | None]
"""Microphone frames from one connection, ending with None when it closes."""


class BrowserAudio(Playback):
    def __init__(self, on_microphone: Callable[[Microphone], Coroutine[Any, Any, None]]) -> None:
        self._on_microphone = on_microphone
        self._socket: WebSocket | None = None
        self._outbox: asyncio.Queue[bytes | str] = asyncio.Queue()
        self._clock = ClockPlayback()
        self._routes: dict[int, Literal["browser", "clock"]] = {}
        self._written: dict[int, int] = {}
        self._played: dict[int, int] = {}
        self._stopped: set[int] = set()
        self._progress = asyncio.Condition()

    async def serve(self, websocket: WebSocket) -> None:
        """Run one tab's connection until it closes. A newer tab replaces an older one."""
        await websocket.accept()
        if self._socket is not None:
            await self._socket.close()
        self._socket = websocket
        self._outbox = asyncio.Queue()
        microphone: Microphone = asyncio.Queue()
        tasks = (
            asyncio.create_task(self._on_microphone(microphone)),
            asyncio.create_task(_send_all(websocket, self._outbox)),
        )
        try:
            while True:
                message = await websocket.receive()
                if message["type"] == "websocket.disconnect":
                    break
                if (frame := message.get("bytes")) is not None:
                    microphone.put_nowait(frame)
                elif (text := message.get("text")) is not None:
                    await self._report(json.loads(text))
        except WebSocketDisconnect:
            pass
        finally:
            microphone.put_nowait(None)
            if self._socket is websocket:
                self._socket = None
                await self._lose_browser()
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    def write(self, utterance: int, pcm: bytes) -> None:
        route = self._routes.setdefault(utterance, "browser" if self._socket else "clock")
        if route == "clock":
            self._clock.write(utterance, pcm)
            return
        if utterance in self._stopped:
            return
        self._written[utterance] = self._written.get(utterance, 0) + len(pcm) // 2
        self._outbox.put_nowait(struct.pack("<I", utterance) + pcm)

    async def drain(self, utterance: int) -> bool:
        if self._routes.get(utterance) != "browser":
            return await self._clock.drain(utterance)
        unplayed = self._written.get(utterance, 0) - self._played.get(utterance, 0)
        try:
            await self._wait(
                lambda: self._finished(utterance), unplayed / TTS_SAMPLE_RATE + DRAIN_GRACE_S
            )
        except TimeoutError:
            logger.warning("the browser never reported utterance %s as played", utterance)
            return False
        return self._played.get(utterance, 0) >= self._written.get(utterance, 0)

    async def stop(self, utterance: int) -> int:
        if self._routes.get(utterance) != "browser":
            return await self._clock.stop(utterance)
        if utterance not in self._stopped:
            self._outbox.put_nowait(json.dumps({"type": "stop", "utterance": utterance}))
            try:
                await self._wait(lambda: utterance in self._stopped, STOP_ACK_TIMEOUT_S)
            except TimeoutError:
                logger.warning("the browser did not confirm stopping utterance %s", utterance)
                self._stopped.add(utterance)
        return self._played.get(utterance, 0)

    def played(self, utterance: int) -> int:
        if self._routes.get(utterance) == "clock":
            return self._clock.played(utterance)
        return self._played.get(utterance, 0)

    def _finished(self, utterance: int) -> bool:
        return utterance in self._stopped or self._played.get(utterance, 0) >= self._written.get(
            utterance, 0
        )

    async def _wait(self, predicate: Callable[[], bool], timeout_s: float) -> None:
        async with asyncio.timeout(timeout_s), self._progress:
            await self._progress.wait_for(predicate)

    async def _report(self, message: dict[str, object]) -> None:
        utterance, samples = message.get("utterance"), message.get("samples")
        if not isinstance(utterance, int) or not isinstance(samples, int):
            return
        self._played[utterance] = max(self._played.get(utterance, 0), samples)
        if message.get("type") == "stopped":
            self._stopped.add(utterance)
        async with self._progress:
            self._progress.notify_all()

    async def _lose_browser(self) -> None:
        for utterance, route in self._routes.items():
            if route == "browser":
                self._stopped.add(utterance)
        async with self._progress:
            self._progress.notify_all()


async def _send_all(websocket: WebSocket, outbox: asyncio.Queue[bytes | str]) -> None:
    while True:
        item = await outbox.get()
        if isinstance(item, bytes):
            await websocket.send_bytes(item)
        else:
            await websocket.send_text(item)
