import asyncio

import pytest

from events.topics import BoardChanged, StudentTyped, StudentVoice
from providers.llm.base import Message, Text
from services.shared.runtime.fence import Fence
from services.shared.runtime.loop import Idle, Loop, Stopping, Turning
from services.shared.runtime.policy import Turn
from services.tutor.policy import TutorPolicy
from services.tutor.turn import TutorSettlement


class ScriptedTurn(Turn[TutorSettlement]):
    def __init__(self, turn_id: str, input: Message, history: tuple[Message, ...]) -> None:
        super().__init__(turn_id)
        self.input = input
        self.history = history
        self.stop_requested = False
        self._outcome: asyncio.Future[TutorSettlement] = asyncio.get_running_loop().create_future()

    async def run(self, fence: Fence) -> TutorSettlement:
        return await self._outcome

    def request_stop(self) -> None:
        self.stop_requested = True

    def finish(self) -> None:
        reply = Message("assistant", (Text(f"reply from {self.turn_id}"),))
        self._outcome.set_result(TutorSettlement((self.input, reply)))

    def fail(self) -> None:
        self._outcome.set_exception(RuntimeError("model unavailable"))

    @property
    def said(self) -> str:
        return " ".join(p.text for p in self.input.parts if isinstance(p, Text) and not _is_note(p))

    @property
    def notes(self) -> list[str]:
        return [p.text for p in self.input.parts if isinstance(p, Text) and _is_note(p)]


def _is_note(part: Text) -> bool:
    return part.text.startswith("[Note:")


class Tutor:
    def __init__(self) -> None:
        self.turns: list[ScriptedTurn] = []
        self.loop = Loop("tutor", TutorPolicy(self._make_turn))

    def _make_turn(
        self, turn_id: str, input: Message, history: tuple[Message, ...]
    ) -> ScriptedTurn:
        turn = ScriptedTurn(turn_id, input, history)
        self.turns.append(turn)
        return turn

    async def send(self, *stimuli: object) -> None:
        for stimulus in stimuli:
            self.loop.enqueue(stimulus)
        await _settle()

    async def finish_current(self) -> None:
        self.turns[-1].finish()
        await _settle()


async def _settle() -> None:
    for _ in range(10):
        await asyncio.sleep(0)


def started(utterance: str, text: str = "um") -> StudentVoice:
    return StudentVoice(utterance, "started", text)


def final(utterance: str, text: str) -> StudentVoice:
    return StudentVoice(utterance, "final", text)


@pytest.fixture
async def tutor():
    tutor = Tutor()
    tutor.loop.start()
    yield tutor
    await tutor.loop.aclose()


async def test_quiet_and_the_student_finishes_a_sentence_starts_a_reply(tutor: Tutor) -> None:
    await tutor.send(started("u1"), final("u1", "what is a heap"))
    assert [turn.said for turn in tutor.turns] == ["what is a heap"]
    assert isinstance(tutor.loop.state, Turning)


async def test_the_student_starting_to_talk_stops_the_tutor(tutor: Tutor) -> None:
    await tutor.send(StudentTyped("hello"))
    await tutor.send(started("u1"))
    assert tutor.turns[0].stop_requested
    assert isinstance(tutor.loop.state, Stopping)


async def test_after_a_barge_in_the_reply_waits_for_the_student_to_finish(tutor: Tutor) -> None:
    await tutor.send(StudentTyped("hello"), started("u1"))
    await tutor.finish_current()
    assert len(tutor.turns) == 1
    assert isinstance(tutor.loop.state, Idle)

    await tutor.send(final("u1", "wait, go back"))
    assert tutor.turns[1].said == "wait, go back"


async def test_words_said_while_the_tutor_is_busy_stop_it_and_get_the_next_reply(
    tutor: Tutor,
) -> None:
    await tutor.send(StudentTyped("hello"), StudentTyped("actually, explain stacks"))
    assert tutor.turns[0].stop_requested
    await tutor.finish_current()
    assert tutor.turns[1].said == "actually, explain stacks"


async def test_waiting_words_join_what_the_student_says_next(tutor: Tutor) -> None:
    await tutor.send(StudentTyped("hello"), StudentTyped("one thing"), started("u1"))
    await tutor.finish_current()
    assert len(tutor.turns) == 1

    await tutor.send(final("u1", "and another"))
    assert tutor.turns[1].said == "one thing and another"


async def test_a_board_clear_rides_the_next_turn_once(tutor: Tutor) -> None:
    clear = BoardChanged(1, (), "cleared")
    await tutor.send(clear)
    assert tutor.turns == []

    await tutor.send(StudentTyped("hi"))
    await tutor.finish_current()
    await tutor.send(StudentTyped("next"))

    assert tutor.turns[0].notes == ["[Note: The student cleared the whiteboard.]"]
    assert tutor.turns[1].notes == []


async def test_each_turn_sees_the_conversation_so_far(tutor: Tutor) -> None:
    await tutor.send(StudentTyped("first"))
    await tutor.finish_current()
    await tutor.send(StudentTyped("second"))

    history = tutor.turns[1].history
    assert history[0] == tutor.turns[0].input
    assert history[1] == Message("assistant", (Text("reply from turn-1"),))


async def test_a_failed_turn_still_keeps_what_the_student_said(tutor: Tutor) -> None:
    await tutor.send(StudentTyped("first"))
    tutor.turns[0].fail()
    await _settle()
    await tutor.send(StudentTyped("second"))
    assert tutor.turns[1].history == (tutor.turns[0].input,)
