"""Stand-ins for fake mode, which runs the whole app without API keys."""

from __future__ import annotations

import itertools
from collections.abc import Sequence

from providers.llm.base import (
    Finish,
    Message,
    StreamEvent,
    Text,
    TextDelta,
    Tool,
    ToolCall,
    ToolResult,
)

_calls = itertools.count(1)


def tutor_reply(messages: Sequence[Message], tools: Sequence[Tool]) -> list[StreamEvent]:
    """A scripted tutor that repeats back what it hears, so you set how long it talks.

    Asked to draw, it calls `draw_diagram` and then repeats the request.
    """
    last = messages[-1]
    if any(isinstance(part, ToolResult) for part in last.parts):
        request = next(part for part in messages[-2].parts if isinstance(part, ToolCall))
        return _say(f"Sure, I'll draw that. You asked: {request.arguments['description']}")
    texts = [part.text for part in last.parts if isinstance(part, Text)]
    if any("is now on the whiteboard" in text for text in texts):
        return _say("The diagram is up on the whiteboard. Take a look and tell me what you notice.")
    said = " ".join(text for text in texts if not text.startswith("[Note:"))
    if tools and any(word in said.lower() for word in ("draw", "diagram", "show me")):
        call = ToolCall(f"call-{next(_calls)}", "draw_diagram", {"description": said})
        return [call, Finish("tool_use")]
    return _say(f"You said: {said}")


def diagram_reply(messages: Sequence[Message], tools: Sequence[Tool]) -> list[StreamEvent]:
    """A scripted diagram agent that always draws the same small flowchart."""
    answer = (
        "Title: Thinking it through\n\n"
        "```mermaid\nflowchart LR\n  A[Question] --> B[Break it down] --> C[Answer]\n```"
    )
    return [TextDelta(answer), Finish("end_turn")]


def _say(text: str) -> list[StreamEvent]:
    return [*(TextDelta(word + " ") for word in text.split()), Finish("end_turn")]
