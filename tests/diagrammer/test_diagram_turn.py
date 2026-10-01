import asyncio

from events.topics import DiagramRequested, Drawn, Failed
from providers.llm.base import Finish, TextDelta
from providers.testing.fakes import ScriptedChatModel
from services.diagrammer.turn import DiagramTurn, parse_diagram
from services.shared.runtime.fence import Fence

ANSWER = 'Title: A heap\n\n```mermaid\nflowchart TD\n  A["1"] --> B["3"]\n```'


def test_a_titled_mermaid_answer_becomes_a_diagram() -> None:
    assert parse_diagram(ANSWER) == Drawn("A heap", 'flowchart TD\n  A["1"] --> B["3"]')


def test_an_answer_without_mermaid_is_a_failure() -> None:
    assert isinstance(parse_diagram("Sorry, I can't draw that."), Failed)


async def test_a_stop_before_the_turn_runs_is_honored() -> None:
    model = ScriptedChatModel(
        lambda messages, tools: [TextDelta(ANSWER), Finish("end_turn")], delay_s=5
    )
    turn = DiagramTurn(DiagramRequested("d1", "a heap"), model=model, system="")

    turn.request_stop()
    async with asyncio.timeout(1):
        outcome = await turn.run(Fence())

    assert isinstance(outcome, Failed)
