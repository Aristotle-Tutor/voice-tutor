from __future__ import annotations

import base64
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from urllib.parse import urlencode

import httpx
import websockets
from websockets.asyncio.client import ClientConnection

from providers.errors import ProviderError
from providers.stt.base import (
    STT_SAMPLE_RATE,
    Committed,
    Partial,
    RecognizerStream,
    SpeechRecognizer,
    SttEvent,
)


@dataclass(frozen=True)
class ElevenLabsSttConfig:
    api_key: str = field(repr=False)
    model: str = "scribe_v2_realtime"
    language: str = "en"
    silence_threshold_s: float = 0.5


class ElevenLabsRecognizer(SpeechRecognizer):
    """Scribe realtime; the server commits an utterance after `silence_threshold_s` of silence.

    Send audio in real time and keep sending silence between utterances: the
    server ends a session on audio sent too fast, or on ~15 s without audio.
    Either ends the session with a `ProviderError`.
    """

    def __init__(self, config: ElevenLabsSttConfig) -> None:
        self._config = config
        # certifi's CA bundle: some Python builds ship without system CAs.
        self._ssl = httpx.create_ssl_context()

    @asynccontextmanager
    async def connect(self) -> AsyncIterator[RecognizerStream]:
        query = urlencode(
            {
                "model_id": self._config.model,
                "language_code": self._config.language,
                "audio_format": f"pcm_{STT_SAMPLE_RATE}",
                "commit_strategy": "vad",
                "vad_silence_threshold_secs": self._config.silence_threshold_s,
            }
        )
        try:
            ws = await websockets.connect(
                f"wss://api.elevenlabs.io/v1/speech-to-text/realtime?{query}",
                ssl=self._ssl,
                additional_headers={"xi-api-key": self._config.api_key},
            )
        except (OSError, websockets.WebSocketException) as exc:
            raise ProviderError("elevenlabs", str(exc)) from exc
        async with ws:
            yield _ElevenLabsStream(ws)


class _ElevenLabsStream(RecognizerStream):
    def __init__(self, ws: ClientConnection) -> None:
        self._ws = ws

    async def send(self, pcm: bytes) -> None:
        chunk = {
            "message_type": "input_audio_chunk",
            "audio_base_64": base64.b64encode(pcm).decode(),
            "commit": False,
            "sample_rate": STT_SAMPLE_RATE,
        }
        try:
            await self._ws.send(json.dumps(chunk))
        except websockets.ConnectionClosed as exc:
            raise ProviderError("elevenlabs", str(exc)) from exc

    async def events(self) -> AsyncIterator[SttEvent]:
        try:
            async for raw in self._ws:
                message = json.loads(raw)
                kind, text = message["message_type"], message.get("text")
                if "error" in message:
                    raise ProviderError("elevenlabs", f"{kind}: {message['error']}")
                if text and kind == "partial_transcript":
                    yield Partial(text)
                elif text and kind == "committed_transcript":
                    yield Committed(text)
        except websockets.ConnectionClosedError as exc:
            raise ProviderError("elevenlabs", str(exc)) from exc
        if self._ws.protocol.close_rcvd_then_sent:
            raise ProviderError("elevenlabs", f"server ended the session ({self._ws.close_code})")
