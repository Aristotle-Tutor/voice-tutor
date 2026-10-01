"""The vocabulary every chat model speaks.

Services build conversations from these types and read these stream events.
Each vendor translates them to and from its own SDK.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class Text:
    text: str


@dataclass(frozen=True)
class ToolCall:
    call_id: str
    name: str
    arguments: Mapping[str, object]


@dataclass(frozen=True)
class ToolResult:
    call_id: str
    content: str
    is_error: bool = False


type Part = Text | ToolCall | ToolResult


@dataclass(frozen=True)
class Message:
    """One conversation message.

    Assistant messages hold `Text` and `ToolCall` parts. User messages hold
    `Text` and `ToolResult` parts. Two messages in a row may share a role;
    vendors merge them if their API needs strict alternation.
    """

    role: Literal["user", "assistant"]
    parts: tuple[Part, ...]


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: Mapping[str, object]
    """A JSON Schema object describing the arguments."""


@dataclass(frozen=True)
class TextDelta:
    text: str


@dataclass(frozen=True)
class ToolCallStarted:
    """The model has started writing a tool call. Its `ToolCall` follows once complete."""

    call_id: str
    name: str


@dataclass(frozen=True)
class Finish:
    reason: Literal["end_turn", "tool_use", "max_tokens", "refusal", "other"]


type StreamEvent = TextDelta | ToolCallStarted | ToolCall | Finish
"""A stream yields text deltas and tool calls in order, then one `Finish`.

Each tool call is announced with `ToolCallStarted` as soon as the model begins
it, and arrives whole as a `ToolCall` once its arguments are complete.
"""


class ChatModel(ABC):
    @abstractmethod
    def stream(self, *, system: str, messages: Sequence[Message]) -> AsyncIterator[StreamEvent]:
        """Stream one text response.

        Closing the iterator early (or cancelling the task consuming it)
        stops the request. Vendor failures raise `ProviderError`.
        """


class CallsTools(ABC):
    """A model that can call tools. Vendors that support it inherit this too."""

    @abstractmethod
    def stream_with_tools(
        self,
        *,
        system: str,
        messages: Sequence[Message],
        tools: Sequence[Tool],
    ) -> AsyncIterator[StreamEvent]:
        """Like `ChatModel.stream`, but the model may answer with `ToolCall`s."""
