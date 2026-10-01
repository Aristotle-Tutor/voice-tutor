from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field

import httpx

from providers.errors import ProviderError
from providers.tts.base import TTS_SAMPLE_RATE, SpeechSynthesizer


@dataclass(frozen=True)
class ElevenLabsTtsConfig:
    api_key: str = field(repr=False)
    voice_id: str = "BWGwF36RwZsLxWHtzZ3e"
    model: str = "eleven_flash_v2_5"


class ElevenLabsSynthesizer(SpeechSynthesizer):
    """One streaming HTTP request per call on a pooled connection. Close with `aclose()`."""

    def __init__(self, config: ElevenLabsTtsConfig) -> None:
        self._config = config
        self._client = httpx.AsyncClient(
            base_url="https://api.elevenlabs.io", headers={"xi-api-key": config.api_key}
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        try:
            async with self._client.stream(
                "POST",
                f"/v1/text-to-speech/{self._config.voice_id}/stream",
                params={"output_format": f"pcm_{TTS_SAMPLE_RATE}"},
                json={"text": text, "model_id": self._config.model},
            ) as response:
                if response.is_error:
                    await response.aread()
                    raise ProviderError(
                        "elevenlabs", f"HTTP {response.status_code}: {response.text}"
                    )
                carry = b""
                async for chunk in response.aiter_bytes():
                    pcm = carry + chunk
                    whole = len(pcm) - len(pcm) % 2
                    carry = pcm[whole:]
                    if whole:
                        yield pcm[:whole]
        except httpx.HTTPError as exc:
            raise ProviderError("elevenlabs", str(exc)) from exc
