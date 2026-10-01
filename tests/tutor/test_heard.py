from providers.llm.base import Message, Text, ToolCall, ToolResult
from services.tutor.heard import keep_heard

CALL = ToolCall("c1", "draw_diagram", {"description": "a tree"})
RESULT = ToolResult("c1", "drawing")


def test_the_cut_runs_across_model_calls_and_keeps_tool_parts() -> None:
    reply = (
        Message("assistant", (Text("Let me draw it."), CALL)),
        Message("user", (RESULT,)),
        Message("assistant", (Text(" A tree has a root."),)),
    )

    kept = keep_heard(reply, len("Let me draw it. A tree"))

    assert kept == (
        Message("assistant", (Text("Let me draw it."), CALL)),
        Message("user", (RESULT,)),
        Message("assistant", (Text(" A tree"),)),
    )


def test_unheard_text_is_dropped_along_with_messages_left_empty() -> None:
    reply = (Message("assistant", (Text("Hello there."),)),)
    assert keep_heard(reply, 0) == ()
