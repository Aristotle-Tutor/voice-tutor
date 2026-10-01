"""The always-interruptible agent loop.

One loop runs one agent. It has one inbox and one turn slot.

- Every message from outside (a stimulus) goes into the inbox. The loop
  handles one message at a time, in arrival order, by asking the policy for a
  verdict.
- At most one turn runs at a time. The policy starts turns with `spawn`, and
  only when the loop is idle.
- Any turn can be stopped. `stop_current` asks the running turn to wrap up.
  If it has not ended within `stop_timeout_s`, the loop cancels it and moves
  on without it.
- When a turn ends, for any reason, the loop frees the slot and calls the
  policy's `boundary`. The policy records the result there and decides what
  runs next. A turn has always ended before the next one starts.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

from services.shared.runtime.fence import Fence
from services.shared.runtime.policy import Policy, Turn, TurnEnded

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Idle:
    pass


@dataclass(frozen=True)
class Turning:
    turn_id: str


@dataclass(frozen=True)
class Stopping:
    turn_id: str
    reason: str


@dataclass(frozen=True)
class Closed:
    pass


type LoopState = Idle | Turning | Stopping | Closed


@dataclass(frozen=True)
class _Stimulus:
    value: object


@dataclass(frozen=True)
class _TurnDone:
    spawn: int
    task: asyncio.Task[Any]


@dataclass(frozen=True)
class _StopExpired:
    spawn: int


type _Message = _Stimulus | _TurnDone | _StopExpired


class TurnStopTimeout(Exception):
    pass


class Loop[S]:
    def __init__(self, name: str, policy: Policy[S], *, stop_timeout_s: float = 3.0) -> None:
        self.name = name
        self._policy = policy
        self._stop_timeout_s = stop_timeout_s
        self._inbox: asyncio.Queue[_Message] = asyncio.Queue()
        self._state: LoopState = Idle()
        self._spawn = 0
        self._turn: Turn[S] | None = None
        self._task: asyncio.Task[S] | None = None
        self._fence: Fence | None = None
        self._stop_timer: asyncio.TimerHandle | None = None
        self._runner: asyncio.Task[None] | None = None

    @property
    def state(self) -> LoopState:
        return self._state

    def start(self) -> None:
        self._runner = asyncio.create_task(self._run(), name=f"{self.name}-loop")

    def enqueue(self, stimulus: object) -> None:
        if not isinstance(self._state, Closed):
            self._inbox.put_nowait(_Stimulus(stimulus))

    def spawn(self, turn: Turn[S]) -> None:
        if not isinstance(self._state, Idle):
            raise RuntimeError(f"{self.name} cannot start {turn.turn_id} while {self._state}")
        self._spawn += 1
        spawn = self._spawn
        self._turn = turn
        self._fence = Fence()
        self._task = asyncio.create_task(turn.run(self._fence), name=f"{self.name}-{turn.turn_id}")
        self._task.add_done_callback(lambda task: self._inbox.put_nowait(_TurnDone(spawn, task)))
        self._state = Turning(turn.turn_id)
        logger.info("%s: started %s", self.name, turn.turn_id)

    def stop_current(self, reason: str) -> None:
        """Ask the running turn to wrap up. Does nothing unless a turn is running."""
        if not isinstance(self._state, Turning) or self._turn is None:
            return
        spawn = self._spawn
        self._state = Stopping(self._turn.turn_id, reason)
        self._stop_timer = asyncio.get_running_loop().call_later(
            self._stop_timeout_s, lambda: self._inbox.put_nowait(_StopExpired(spawn))
        )
        logger.info("%s: stopping %s (%s)", self.name, self._turn.turn_id, reason)
        self._turn.request_stop()

    async def aclose(self) -> None:
        self._state = Closed()
        if self._stop_timer is not None:
            self._stop_timer.cancel()
        if self._fence is not None:
            self._fence.revoke()
        tasks: list[asyncio.Task[Any]] = [t for t in (self._task, self._runner) if t is not None]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _run(self) -> None:
        while True:
            message = await self._inbox.get()
            try:
                self._handle(message)
            except Exception:
                logger.exception("%s: failed to handle %r", self.name, message)

    def _handle(self, message: _Message) -> None:
        match message:
            case _Stimulus(value):
                self._policy.verdict(value, self)
            case _TurnDone(spawn, task):
                settlement, error = _result_of(task)
                # A turn abandoned at its stop timeout can still finish later. Its spawn
                # number may match, but the slot was already freed for it.
                if spawn == self._spawn and self._turn is not None:
                    self._end(settlement, error)
            case _StopExpired(spawn):
                if spawn == self._spawn and isinstance(self._state, Stopping):
                    assert self._task is not None
                    self._task.cancel()
                    self._end(None, TurnStopTimeout(f"did not stop within {self._stop_timeout_s}s"))

    def _end(self, settlement: S | None, error: BaseException | None) -> None:
        assert self._turn is not None and self._fence is not None
        turn = self._turn
        self._fence.revoke()
        if self._stop_timer is not None:
            self._stop_timer.cancel()
        self._turn = self._task = self._fence = self._stop_timer = None
        self._state = Idle()
        if error is None:
            logger.info("%s: %s ended", self.name, turn.turn_id)
        else:
            logger.error("%s: %s failed", self.name, turn.turn_id, exc_info=error)
        self._policy.boundary(TurnEnded(turn.turn_id, settlement, error), self)


def _result_of[S](task: asyncio.Task[S]) -> tuple[S | None, BaseException | None]:
    if task.cancelled():
        return None, asyncio.CancelledError()
    if (error := task.exception()) is not None:
        return None, error
    return task.result(), None
