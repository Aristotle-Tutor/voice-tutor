from __future__ import annotations

from events.bus import Publisher
from events.topics import DiagramRequested
from providers.llm.base import Tool, ToolCall, ToolResult
from services.shared.runtime.fence import Fence

DRAW_DIAGRAM = Tool(
    name="draw_diagram",
    description=(
        "Ask the diagram agent to draw a diagram on the whiteboard. Drawing takes several "
        "seconds and happens in the background, so this returns right away."
    ),
    parameters={
        "type": "object",
        "properties": {
            "description": {
                "type": "string",
                "description": "What the diagram should show, specific enough to draw from.",
            }
        },
        "required": ["description"],
        "additionalProperties": False,
    },
)


class TutorTools:
    definitions = (DRAW_DIAGRAM,)

    def __init__(self, request_diagram: Publisher[DiagramRequested]) -> None:
        self._request_diagram = request_diagram
        self._requests = 0

    def run(self, call: ToolCall, fence: Fence) -> ToolResult:
        if call.name != DRAW_DIAGRAM.name:
            return ToolResult(call.call_id, f"There is no tool called {call.name}.", is_error=True)
        description = call.arguments.get("description")
        if not isinstance(description, str) or not description.strip():
            return ToolResult(call.call_id, "A description is required.", is_error=True)
        if not fence.live:
            return ToolResult(call.call_id, "Not sent: the turn had already ended.", is_error=True)
        self._requests += 1
        request_id = f"d{self._requests}"
        self._request_diagram.publish(DiagramRequested(request_id, description.strip()))
        return ToolResult(
            call.call_id,
            f"Diagram {request_id} is being drawn. It is not on the whiteboard yet. "
            "You will get a note when it appears.",
        )
