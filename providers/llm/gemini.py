from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field

import httpx
from google import genai
from google.genai import errors, types

from providers.errors import ProviderError
from providers.llm.base import ChatModel, Finish, Message, StreamEvent, Text, TextDelta

_REFUSALS = {
    types.FinishReason.SAFETY,
    types.FinishReason.PROHIBITED_CONTENT,
    types.FinishReason.BLOCKLIST,
    types.FinishReason.SPII,
    types.FinishReason.RECITATION,
}


@dataclass(frozen=True)
class GeminiConfig:
    api_key: str = field(repr=False)
    model: str = "gemini-3.7-flash"


class GeminiChatModel(ChatModel):
    """Gemini for text-only replies. Tool calls and results in the history are skipped.

    Close with `aclose()` at shutdown.
    """

    def __init__(self, config: GeminiConfig) -> None:
        self._config = config
        # Our own httpx client pins the transport (the SDK picks aiohttp when it is
        # installed), so transport failures are always `httpx.HTTPError`.
        self._http = httpx.AsyncClient()
        self._client = genai.Client(
            api_key=config.api_key,
            http_options=types.HttpOptions(httpx_async_client=self._http),
        )

    async def aclose(self) -> None:
        await self._http.aclose()

    async def stream(
        self, *, system: str, messages: Sequence[Message]
    ) -> AsyncIterator[StreamEvent]:
        contents = [
            types.Content(
                role="model" if message.role == "assistant" else "user",
                parts=[types.Part(text=part.text) for part in texts],
            )
            for message in messages
            if (texts := [p for p in message.parts if isinstance(p, Text) and p.text.strip()])
        ]
        config = types.GenerateContentConfig(
            system_instruction=system or None,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        chunks: asyncio.Queue[types.GenerateContentResponse | None] = asyncio.Queue()

        async def pump() -> None:
            try:
                async for chunk in await self._client.aio.models.generate_content_stream(
                    model=self._config.model, contents=contents, config=config
                ):
                    chunks.put_nowait(chunk)
            finally:
                chunks.put_nowait(None)

        # The SDK's stream is a chain of async generators held in reference cycles:
        # closing it leaves the request open until a GC pass. Cancelling the task
        # that reads it unwinds the whole chain and closes the connection.
        pumping = asyncio.create_task(pump())
        finish = Finish("other")
        try:
            while (chunk := await chunks.get()) is not None:
                if chunk.text:
                    yield TextDelta(chunk.text)
                reason = chunk.candidates[0].finish_reason if chunk.candidates else None
                if reason == types.FinishReason.STOP:
                    finish = Finish("end_turn")
                elif reason == types.FinishReason.MAX_TOKENS:
                    finish = Finish("max_tokens")
                elif reason in _REFUSALS or (
                    chunk.prompt_feedback and chunk.prompt_feedback.block_reason
                ):
                    finish = Finish("refusal")
            await pumping
        except (errors.APIError, httpx.HTTPError) as exc:
            raise ProviderError("gemini", str(exc)) from exc
        finally:
            pumping.cancel()
        yield finish
