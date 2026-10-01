"""One tutor turn: one reply, from the model's first token to the student hearing the last word."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Literal

from events.bus import Publisher
from events.topics import TutorSpeech
from providers.llm.base import (
    CallsTools,
    Finish,
    Message,
    Text,
    TextDelta,
    ToolCall,
    ToolCallStarted,
)
from services.shared.runtime.fence import Fence
from services.shared.runtime.policy import Turn
from services.tutor.heard import keep_heard
from services.tutor.ports import Utterance, Voice
from services.tutor.tools import TutorTools

MAX_MODEL_CALLS = 4


@dataclass(frozen=True)
class TutorSettlement:
    messages: tuple[Message, ...]
    """What this turn adds to the conversation: its input, then the reply as it was heard."""


class TutorTurn(Turn[TutorSettlement]):
    """Streams the model's reply into speech, running tools between model calls.

    A stop interrupts the speech at once and cancels the model call. Either
    way the turn settles with the reply cut down to what the student heard.
    """

    def __init__(
        self,
        turn_id: str,
        *,
        model: CallsTools,
        system: str,
        history: tuple[Message, ...],
        input: Message,
        tools: TutorTools,
        voice: Voice,
        captions: Publisher[TutorSpeech],
    ) -> None:
        super().__init__(turn_id)
        self._model = model
        self._system = system
        self._history = history
        self._input = input
        self._tools = tools
        self._voice = voice
        self._captions = captions
        self._reply: list[Message] = []
        self._unfinished_text = ""
        self._heard_so_far = ""
        self._utterance: Utterance | None = None
        self._work: asyncio.Task[None] | None = None
        self._stopping = False

    async def run(self, fence: Fence) -> TutorSettlement:
        self._caption(fence, "started")

        def on_heard(text: str) -> None:
            self._heard_so_far += text
            self._caption(fence, "text", text)

        utterance = self._utterance = self._voice.utterance(on_heard)
        work = self._work = asyncio.create_task(self._reply_with_tools(fence, utterance))
        if self._stopping:
            self.request_stop()
        try:
            try:
                await work
            except asyncio.CancelledError:
                if _cancelled_from_outside():
                    raise
                if self._unfinished_text:
                    self._reply.append(Message("assistant", (Text(self._unfinished_text),)))
            heard = await utterance.finish()
        except BaseException:
            utterance.interrupt()
            self._caption(fence, "ended", self._heard_so_far, interrupted=True)
            raise
        self._caption(fence, "ended", heard.text, interrupted=heard.interrupted)
        return TutorSettlement((self._input, *keep_heard(self._reply, len(heard.text))))

    def request_stop(self) -> None:
        self._stopping = True
        if self._utterance is not None:
            self._utterance.interrupt()
        if self._work is not None:
            self._work.cancel()

    async def _reply_with_tools(self, fence: Fence, utterance: Utterance) -> None:
        for _ in range(MAX_MODEL_CALLS):
            calls: list[ToolCall] = []
            async for event in self._model.stream_with_tools(
                system=self._system,
                messages=(*self._history, self._input, *self._reply),
                tools=self._tools.definitions,
            ):
                match event:
                    case TextDelta(text):
                        self._speak(fence, utterance, text)
                    case ToolCallStarted():
                        if self._unfinished_text and not self._unfinished_text[-1].isspace():
                            # The speaker holds a finished sentence until it sees the space
                            # after it. A tool call ends this text, so add that space now.
                            self._speak(fence, utterance, " ")
                    case ToolCall():
                        calls.append(event)
                    case Finish():
                        pass
            text, self._unfinished_text = self._unfinished_text, ""
            self._reply.append(Message("assistant", (*([Text(text)] if text else []), *calls)))
            if not calls:
                return
            results = tuple(self._tools.run(call, fence) for call in calls)
            self._reply.append(Message("user", results))

    def _speak(self, fence: Fence, utterance: Utterance, text: str) -> None:
        self._unfinished_text += text
        if fence.live:
            utterance.say(text)

    def _caption(
        self,
        fence: Fence,
        kind: Literal["started", "text", "ended"],
        text: str = "",
        *,
        interrupted: bool = False,
    ) -> None:
        if fence.live:
            self._captions.publish(TutorSpeech(self.turn_id, kind, text, interrupted))


def _cancelled_from_outside() -> bool:
    task = asyncio.current_task()
    return task is not None and task.cancelling() > 0
