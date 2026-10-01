"""What the tutor needs from outside. The app supplies an implementation."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol


class Heard(Protocol):
    @property
    def text(self) -> str:
        """What the student actually heard: a prefix of everything said, cut at a word."""
        ...

    @property
    def interrupted(self) -> bool: ...


class Utterance(Protocol):
    """The speech of one tutor turn."""

    def say(self, text: str) -> None:
        """Queue more text to speak. Returns at once."""
        ...

    def interrupt(self) -> None:
        """Stop speaking now. Text said afterwards is ignored."""
        ...

    async def finish(self) -> Heard:
        """Wait until everything said has played, or the interrupt has taken effect."""
        ...


class Voice(Protocol):
    def utterance(self, on_heard: Callable[[str], None]) -> Utterance:
        """Start one turn's speech. `on_heard` gets each newly heard piece of text as it plays."""
        ...
