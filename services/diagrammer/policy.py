"""The diagram agent's decisions: draw one diagram at a time, in the order they were asked for."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable

from events.bus import Publisher
from events.topics import DiagramFinished, DiagramRequested, Drawn, Failed
from services.shared.runtime.loop import Idle, Loop
from services.shared.runtime.policy import Policy, Turn, TurnEnded

type Outcome = Drawn | Failed


class DiagrammerPolicy(Policy[Outcome]):
    def __init__(
        self,
        make_turn: Callable[[DiagramRequested], Turn[Outcome]],
        finished: Publisher[DiagramFinished],
    ) -> None:
        self._make_turn = make_turn
        self._finished = finished
        self._queue: deque[DiagramRequested] = deque()
        self._drawing: DiagramRequested | None = None

    def verdict(self, stimulus: object, loop: Loop[Outcome]) -> None:
        if isinstance(stimulus, DiagramRequested):
            self._queue.append(stimulus)
            self._start_next(loop)

    def boundary(self, ended: TurnEnded[Outcome], loop: Loop[Outcome]) -> None:
        assert self._drawing is not None
        outcome = ended.settlement
        if outcome is None:
            outcome = Failed(f"The diagram agent failed: {ended.error}")
        self._finished.publish(DiagramFinished(self._drawing.request_id, outcome))
        self._drawing = None
        self._start_next(loop)

    def _start_next(self, loop: Loop[Outcome]) -> None:
        if isinstance(loop.state, Idle) and self._queue:
            self._drawing = self._queue.popleft()
            loop.spawn(self._make_turn(self._drawing))
