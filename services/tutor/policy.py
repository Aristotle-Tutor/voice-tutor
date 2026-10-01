"""The tutor's decisions: what each arriving message does, given what is happening right now.

What the policy looks at:
- whether the student is talking: from the first words of an utterance until
  its final transcript arrives
- the loop's state: idle, a turn running (thinking or speaking), or a turn
  stopping

What it can do with a message:
- start a turn, when the loop is idle
- stop the running turn; the loop calls `boundary` once it has ended
- keep the student's words waiting until a reply can start
- keep a rider: a private note that rides along with whichever turn starts
  next, once
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from events.topics import (
    BoardChanged,
    DiagramFinished,
    Drawn,
    Failed,
    StudentTyped,
    StudentVoice,
)
from providers.llm.base import Message, Text
from services.shared.runtime.loop import Idle, Loop
from services.shared.runtime.policy import Policy, Turn, TurnEnded
from services.tutor.turn import TutorSettlement

type MakeTurn = Callable[[str, Message, tuple[Message, ...]], Turn[TutorSettlement]]
"""Builds a turn from its id, its input message, and the conversation so far."""

MAX_DIAGRAM_SOURCE_CHARS = 2_000


class TutorPolicy(Policy[TutorSettlement]):
    def __init__(self, make_turn: MakeTurn) -> None:
        self._make_turn = make_turn
        self._history: list[Message] = []
        self._turn_input: Message | None = None
        self._turns = 0
        self._student_utterance: str | None = None
        self._waiting_words: list[str] = []
        self._riders: list[str] = []
        self._diagram_notes: list[str] = []

    @property
    def student_talking(self) -> bool:
        return self._student_utterance is not None

    def verdict(self, stimulus: object, loop: Loop[TutorSettlement]) -> None:
        match stimulus:
            case StudentVoice(kind="started"):
                self._student_utterance = stimulus.utterance_id
                loop.stop_current("the student started talking")
            case StudentVoice(kind="interim"):
                pass
            case StudentVoice(kind="final"):
                if stimulus.utterance_id == self._student_utterance:
                    self._student_utterance = None
                self._student_said(stimulus.text, loop)
            case StudentTyped(text=text):
                self._student_said(text, loop)
            case BoardChanged(change="cleared"):
                self._riders.append("The student cleared the whiteboard.")
            case DiagramFinished():
                self._diagram_finished(stimulus, loop)
            case _:
                pass

    def boundary(self, ended: TurnEnded[TutorSettlement], loop: Loop[TutorSettlement]) -> None:
        if ended.settlement is not None:
            self._history.extend(ended.settlement.messages)
        elif self._turn_input is not None:
            self._history.append(self._turn_input)
        self._turn_input = None

        if self._diagram_notes:
            self._start(loop, notes=_take(self._diagram_notes))
        else:
            self._start_waiting_reply(loop)

    def _student_said(self, text: str, loop: Loop[TutorSettlement]) -> None:
        if text.strip():
            self._waiting_words.append(text.strip())
        if not self._waiting_words:
            return
        if isinstance(loop.state, Idle):
            self._start_waiting_reply(loop)
        else:
            loop.stop_current("the student said something")

    def _start_waiting_reply(self, loop: Loop[TutorSettlement]) -> None:
        """Reply to the student's waiting words, once they have stopped talking."""
        if self._waiting_words and not self.student_talking:
            self._start(loop, said=_take(self._waiting_words))

    def _diagram_finished(self, finished: DiagramFinished, loop: Loop[TutorSettlement]) -> None:
        self._diagram_notes.append(_diagram_note(finished))
        if isinstance(loop.state, Idle):
            self._start(loop, notes=_take(self._diagram_notes))
        else:
            loop.stop_current("a diagram finished")

    def _start(
        self,
        loop: Loop[TutorSettlement],
        *,
        said: Sequence[str] = (),
        notes: Sequence[str] = (),
    ) -> None:
        parts = [Text(f"[Note: {note}]") for note in (*_take(self._riders), *notes)]
        if said:
            parts.append(Text(" ".join(said)))
        self._turn_input = Message("user", tuple(parts))
        self._turns += 1
        turn = self._make_turn(f"turn-{self._turns}", self._turn_input, tuple(self._history))
        loop.spawn(turn)


def _take(items: list[str]) -> tuple[str, ...]:
    taken = tuple(items)
    items.clear()
    return taken


def _diagram_note(finished: DiagramFinished) -> str:
    match finished.outcome:
        case Drawn(title=title, mermaid=mermaid):
            source = mermaid[:MAX_DIAGRAM_SOURCE_CHARS]
            return (
                f'Diagram {finished.request_id} ("{title}") is now on the whiteboard. '
                f"Its Mermaid source:\n{source}"
            )
        case Failed(reason=reason):
            return f"Diagram {finished.request_id} could not be drawn: {reason}"
