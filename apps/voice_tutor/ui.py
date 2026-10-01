"""The page: a transcript, the whiteboard, and controls. Every open tab shows the same session."""

from __future__ import annotations

import html
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from nicegui import app, ui
from starlette.websockets import WebSocket

from apps.voice_tutor.compose import VoiceTutorApp
from events.bus import Bus
from events.topics import (
    BoardChanged,
    BoardClearRequested,
    BoardItem,
    Drawn,
    Failed,
    StudentTyped,
    StudentVoice,
    TutorSpeech,
)

AUDIO_SCRIPTS = Path(__file__).parents[2] / "packages" / "voice" / "static"


@dataclass
class Bubble:
    key: str
    speaker: Literal["student", "tutor"]
    text: str = ""
    interrupted: bool = False
    live: bool = True


class View(Protocol):
    def show_bubble(self, bubble: Bubble) -> None: ...
    def remove_bubble(self, key: str) -> None: ...
    def show_board(self, items: tuple[BoardItem, ...]) -> None: ...
    def show_status(self, status: str) -> None: ...


class Conversation:
    """What the page shows, kept up to date from the bus."""

    def __init__(self, bus: Bus) -> None:
        self.bubbles: dict[str, Bubble] = {}
        self.board: tuple[BoardItem, ...] = ()
        self.views: set[View] = set()
        self._typed = itertools.count(1)
        bus.subscribe(StudentVoice, "ui", self._student_voice)
        bus.subscribe(TutorSpeech, "ui", self._tutor_speech)
        bus.subscribe(BoardChanged, "ui", self._board_changed)

    @property
    def status(self) -> str:
        live = [bubble for bubble in self.bubbles.values() if bubble.live]
        if any(bubble.speaker == "student" for bubble in live):
            return "Listening"
        tutor = next((bubble for bubble in live if bubble.speaker == "tutor"), None)
        if tutor is None:
            return "Ready"
        return "Speaking" if tutor.text.strip() else "Thinking"

    def add_typed(self, text: str) -> None:
        self._show(Bubble(f"typed-{next(self._typed)}", "student", text, live=False))

    def _student_voice(self, event: StudentVoice) -> None:
        if event.kind == "final" and not event.text.strip():
            self._remove(event.utterance_id)
        else:
            self._show(
                Bubble(event.utterance_id, "student", event.text, live=event.kind != "final")
            )

    def _tutor_speech(self, event: TutorSpeech) -> None:
        bubble = self.bubbles.get(event.turn_id) or Bubble(event.turn_id, "tutor")
        if event.kind == "text":
            bubble.text += event.text
        elif event.kind == "ended":
            bubble.live = False
            bubble.text = event.text
            bubble.interrupted = event.interrupted
            if not bubble.text.strip():
                self._remove(bubble.key)
                return
        self._show(bubble)

    def _board_changed(self, event: BoardChanged) -> None:
        self.board = event.items
        for view in tuple(self.views):
            view.show_board(event.items)

    def _show(self, bubble: Bubble) -> None:
        self.bubbles[bubble.key] = bubble
        for view in tuple(self.views):
            view.show_bubble(bubble)
            view.show_status(self.status)

    def _remove(self, key: str) -> None:
        self.bubbles.pop(key, None)
        for view in tuple(self.views):
            view.remove_bubble(key)
            view.show_status(self.status)


class Page:
    """One tab's view of the conversation."""

    def __init__(self, voice_tutor: VoiceTutorApp, conversation: Conversation) -> None:
        self._app = voice_tutor
        self._conversation = conversation
        self._bubbles: dict[str, ui.html] = {}

        with ui.header().classes("items-center bg-white text-gray-900 border-b"):
            ui.label("Voice tutor").classes("text-lg font-semibold")
            self._status = ui.badge(conversation.status).props("outline")
            if voice_tutor.fake:
                ui.badge("fake mode: scripted tutor, silent speech", color="orange")
            ui.space()
            ui.button("Voice on", icon="mic").props("flat").on(
                "click",
                js_handler="() => window.voiceTutorAudio.start().catch((e) => alert(e.message))",
            )
            ui.button("Voice off", icon="mic_off").props("flat").on(
                "click", js_handler="() => window.voiceTutorAudio.stop()"
            )

        with ui.row().classes("w-full no-wrap gap-6 h-[calc(100vh-160px)]"):
            with ui.column().classes("w-3/5 h-full"):
                self._scroll = ui.scroll_area().classes("w-full h-full")
                with self._scroll:
                    self._transcript = ui.column().classes("w-full gap-2")
            with ui.column().classes("w-2/5 h-full"):
                with ui.row().classes("w-full items-center"):
                    ui.label("Whiteboard").classes("text-base font-semibold")
                    ui.space()
                    ui.button("Clear", on_click=self._clear_board).props("flat dense")
                with ui.scroll_area().classes("w-full h-full border rounded-lg bg-gray-50"):
                    self._board = ui.column().classes("w-full gap-4 p-2")

        with (
            ui.footer().classes("bg-white border-t"),
            ui.row().classes("w-full items-center no-wrap"),
        ):
            self._input = ui.input(placeholder="Type to the tutor").props("outlined dense")
            self._input.classes("flex-grow").on("keydown.enter", self._send)
            ui.button("Send", on_click=self._send)
            ui.button("Say it aloud", on_click=self._say).props("outline").tooltip(
                "Plays your text as if you had spoken it, at talking speed"
            )

        for bubble in conversation.bubbles.values():
            self.show_bubble(bubble)
        self.show_board(conversation.board)

    def show_bubble(self, bubble: Bubble) -> None:
        element = self._bubbles.get(bubble.key)
        if element is None:
            with self._transcript:
                element = ui.html("", sanitize=False)
            self._bubbles[bubble.key] = element
        element.classes(replace=_bubble_classes(bubble))
        element.set_content(_bubble_html(bubble))
        self._scroll.scroll_to(percent=1.0)

    def remove_bubble(self, key: str) -> None:
        if (element := self._bubbles.pop(key, None)) is not None:
            element.delete()

    def show_board(self, items: tuple[BoardItem, ...]) -> None:
        self._board.clear()
        with self._board:
            if not items:
                ui.label("Nothing on the board yet.").classes("text-gray-400 p-4")
            for item in items:
                with ui.card().classes("w-full"):
                    match item.outcome:
                        case None:
                            with ui.row().classes("items-center"):
                                ui.spinner(size="sm")
                                ui.label(f"Drawing: {item.description}").classes("text-gray-500")
                        case Drawn(title=title, mermaid=mermaid):
                            ui.label(title).classes("font-medium")
                            ui.mermaid(mermaid).classes("w-full")
                        case Failed(reason=reason):
                            ui.label(f"Could not draw: {item.description}").classes("font-medium")
                            ui.label(reason).classes("text-red-600 text-sm")

    def show_status(self, status: str) -> None:
        self._status.set_text(status)

    def _send(self) -> None:
        text = (self._input.value or "").strip()
        if text:
            self._input.value = ""
            self._conversation.add_typed(text)
            self._app.typed.publish(StudentTyped(text))

    async def _say(self) -> None:
        text = (self._input.value or "").strip()
        if text:
            self._input.value = ""
            await self._app.speech.say(text)

    def _clear_board(self) -> None:
        self._app.clear_board.publish(BoardClearRequested())


def _bubble_classes(bubble: Bubble) -> str:
    shared = "max-w-[85%] rounded-2xl px-4 py-2 whitespace-pre-wrap"
    if bubble.speaker == "student":
        return f"{shared} self-end bg-blue-600 text-white" + (" opacity-70" if bubble.live else "")
    return f"{shared} self-start bg-gray-100 text-gray-900"


def _bubble_html(bubble: Bubble) -> str:
    if bubble.speaker == "tutor" and not bubble.text:
        return '<span class="text-gray-400">…</span>'
    marker = ' <span class="text-gray-400">— interrupted</span>' if bubble.interrupted else ""
    return html.escape(bubble.text) + marker


def install(voice_tutor: VoiceTutorApp) -> None:
    """Register the page, the audio websocket, and the audio scripts with NiceGUI."""
    conversation = Conversation(voice_tutor.bus)
    app.add_static_files("/voice", AUDIO_SCRIPTS)

    @app.websocket("/audio")
    async def audio(websocket: WebSocket) -> None:
        await voice_tutor.audio.serve(websocket)

    @ui.page("/")
    def index() -> None:
        ui.add_head_html('<script src="/voice/audio.js"></script>')
        page = Page(voice_tutor, conversation)
        conversation.views.add(page)
        ui.context.client.on_delete(lambda: conversation.views.discard(page))
