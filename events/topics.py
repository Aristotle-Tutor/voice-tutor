"""Every message that crosses between services, with who publishes and who consumes it.

Owners:
    room        the voice edge: microphone audio turned into student speech events
    ui          the browser page
    tutor       the tutor agent
    diagrammer  the diagram agent
    whiteboard  the owner of the board
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from events.bus import topic


@topic(publisher="room", consumers=("tutor", "ui"))
@dataclass(frozen=True)
class StudentVoice:
    """What the student is saying, in order, one utterance at a time.

    `started` arrives with the first transcribed words of an utterance,
    `interim` with each better guess at the words so far, and `final` once
    the student has stopped talking. `text` is always the whole utterance so
    far, and may be empty on `final` when nothing intelligible was said.
    """

    utterance_id: str
    kind: Literal["started", "interim", "final"]
    text: str


@topic(publisher="ui", consumers=("tutor",))
@dataclass(frozen=True)
class StudentTyped:
    text: str


@topic(publisher="ui", consumers=("whiteboard",))
@dataclass(frozen=True)
class BoardClearRequested:
    pass


@topic(publisher="tutor", consumers=("diagrammer", "whiteboard"))
@dataclass(frozen=True)
class DiagramRequested:
    request_id: str
    description: str


@dataclass(frozen=True)
class Drawn:
    title: str
    mermaid: str


@dataclass(frozen=True)
class Failed:
    reason: str


@topic(publisher="diagrammer", consumers=("whiteboard", "tutor"))
@dataclass(frozen=True)
class DiagramFinished:
    request_id: str
    outcome: Drawn | Failed


@dataclass(frozen=True)
class BoardItem:
    request_id: str
    description: str
    outcome: Drawn | Failed | None
    """None while the diagram is still being drawn."""


@topic(publisher="whiteboard", consumers=("tutor", "ui"))
@dataclass(frozen=True)
class BoardChanged:
    revision: int
    items: tuple[BoardItem, ...]
    change: Literal["diagram_requested", "diagram_finished", "cleared"]


@topic(publisher="tutor", consumers=("ui",))
@dataclass(frozen=True)
class TutorSpeech:
    """Captions for one tutor turn.

    `started` opens the turn, `text` carries each new piece of the reply as the
    student hears it, and `ended` carries everything the student heard, which
    stops short of the reply when the tutor was interrupted.
    """

    turn_id: str
    kind: Literal["started", "text", "ended"]
    text: str = ""
    interrupted: bool = False
