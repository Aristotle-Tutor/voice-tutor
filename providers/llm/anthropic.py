from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field

import anthropic
import httpx2
from anthropic.types.beta import BetaContentBlockParam, BetaMessageParam, BetaToolParam

from providers.errors import ProviderError
from providers.llm.base import (
    CallsTools,
    ChatModel,
    Finish,
    Message,
    Part,
    StreamEvent,
    Text,
    TextDelta,
    Tool,
    ToolCall,
    ToolCallStarted,
    ToolResult,
)


@dataclass(frozen=True)
class AnthropicConfig:
    api_key: str = field(repr=False)
    model: str = "claude-sonnet-5-5"
    max_tokens: int = 8192


class AnthropicChatModel(ChatModel, CallsTools):
    """Claude tuned for voice latency: no extended thinking, low effort.

    Policy refusals are retried server-side on Anthropic's default fallback
    model. Close with `aclose()` at shutdown.
    """

    def __init__(self, config: AnthropicConfig) -> None:
        self._config = config
        self._client = anthropic.AsyncAnthropic(api_key=config.api_key)

    async def aclose(self) -> None:
        await self._client.close()

    def stream(self, *, system: str, messages: Sequence[Message]) -> AsyncIterator[StreamEvent]:
        return self._stream(system, messages, ())

    def stream_with_tools(
        self,
        *,
        system: str,
        messages: Sequence[Message],
        tools: Sequence[Tool],
    ) -> AsyncIterator[StreamEvent]:
        return self._stream(system, messages, tools)

    async def _stream(
        self, system: str, messages: Sequence[Message], tools: Sequence[Tool]
    ) -> AsyncIterator[StreamEvent]:
        try:
            async with self._client.beta.messages.stream(
                model=self._config.model,
                max_tokens=self._config.max_tokens,
                system=system,
                messages=_to_api_messages(messages),
                tools=[_to_api_tool(tool) for tool in tools],
                thinking={"type": "between_tools"},
                output_config={"effort": "low"},
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            ) as stream:
                async for event in stream:
                    if event.type == "text":
                        yield TextDelta(event.text)
                    elif (
                        event.type == "content_block_start"
                        and event.content_block.type == "tool_use"
                    ):
                        yield ToolCallStarted(event.content_block.id, event.content_block.name)
                    elif (
                        event.type == "content_block_stop"
                        and event.content_block.type == "tool_use"
                    ):
                        block = event.content_block
                        yield ToolCall(block.id, block.name, block.input)
                message = await stream.get_final_message()
        except (anthropic.APIError, httpx2.HTTPError) as exc:
            raise ProviderError("anthropic", str(exc)) from exc
        match message.stop_reason:
            case "end_turn" | "tool_use" | "max_tokens" | "refusal" as reason:
                yield Finish(reason)
            case _:
                yield Finish("other")


def _to_api_messages(messages: Sequence[Message]) -> list[BetaMessageParam]:
    api_messages: list[BetaMessageParam] = []
    for message in messages:
        content = [
            _to_api_block(part)
            for part in message.parts
            if not (isinstance(part, Text) and not part.text.strip())
        ]
        if content:
            api_messages.append({"role": message.role, "content": content})
    return api_messages


def _to_api_block(part: Part) -> BetaContentBlockParam:
    match part:
        case Text():
            return {"type": "text", "text": part.text}
        case ToolCall():
            return {
                "type": "tool_use",
                "id": part.call_id,
                "name": part.name,
                "input": dict(part.arguments),
            }
        case ToolResult():
            return {
                "type": "tool_result",
                "tool_use_id": part.call_id,
                "content": part.content,
                "is_error": part.is_error,
            }


def _to_api_tool(tool: Tool) -> BetaToolParam:
    return {
        "name": tool.name,
        "description": tool.description,
        "input_schema": dict(tool.parameters),
    }
