from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

from services.shared.runtime.fence import Fence

if TYPE_CHECKING:
    from services.shared.runtime.loop import Loop


class Turn[S](ABC):
    """One piece of agent work, such as one reply. The loop runs it as its own task."""

    def __init__(self, turn_id: str) -> None:
        self.turn_id = turn_id

    @abstractmethod
    async def run(self, fence: Fence) -> S:
        """Do the work and return its settlement: what the policy records at the boundary."""

    @abstractmethod
    def request_stop(self) -> None:
        """Wrap up as soon as possible and return what you have. Must not block."""


@dataclass(frozen=True)
class TurnEnded[S]:
    turn_id: str
    settlement: S | None
    """What the turn returned, or None when it failed or did not stop in time."""
    error: BaseException | None = None


class Policy[S](ABC):
    """An agent's decisions. Both methods run on the loop and never wait."""

    @abstractmethod
    def verdict(self, stimulus: object, loop: Loop[S]) -> None:
        """Decide what one arriving message does, given the current state."""

    @abstractmethod
    def boundary(self, ended: TurnEnded[S], loop: Loop[S]) -> None:
        """A turn has ended and the slot is free. Record the result and start whatever is next."""
