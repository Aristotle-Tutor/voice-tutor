"""The composition root: the only place that picks vendors and builds them."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from apps.voice_tutor import fake
from apps.voice_tutor.config import Settings
from apps.voice_tutor.simulated_speech import SpeechSimulator
from events.bus import Bus, Publisher
from events.topics import BoardClearRequested, StudentTyped, StudentVoice
from packages.voice.browser_audio import BrowserAudio, Microphone
from packages.voice.listener import Listener
from packages.voice.speaker import Speaker
from providers.llm.anthropic import AnthropicChatModel, AnthropicConfig
from providers.llm.base import CallsTools, ChatModel
from providers.llm.gemini import GeminiChatModel, GeminiConfig
from providers.stt.elevenlabs import ElevenLabsRecognizer, ElevenLabsSttConfig
from providers.testing.fakes import FakeSynthesizer, ScriptedChatModel
from providers.tts.base import SpeechSynthesizer
from providers.tts.elevenlabs import ElevenLabsSynthesizer, ElevenLabsTtsConfig
from services.diagrammer.service import Diagrammer
from services.tutor.service import Tutor
from services.whiteboard.service import Whiteboard


@dataclass
class VoiceTutorApp:
    bus: Bus
    audio: BrowserAudio
    tutor: Tutor
    diagrammer: Diagrammer
    typed: Publisher[StudentTyped]
    clear_board: Publisher[BoardClearRequested]
    speech: SpeechSimulator
    fake: bool
    vendor_clients: list[Callable[[], Awaitable[None]]] = field(default_factory=list)

    def start(self) -> None:
        self.tutor.start()
        self.diagrammer.start()

    async def aclose(self) -> None:
        await self.tutor.aclose()
        await self.diagrammer.aclose()
        for close in self.vendor_clients:
            await close()


def build(settings: Settings) -> VoiceTutorApp:
    bus = Bus()
    Whiteboard(bus)

    tutor_model: CallsTools
    diagram_model: ChatModel
    synthesizer: SpeechSynthesizer
    vendor_clients: list[Callable[[], Awaitable[None]]] = []
    keys = settings.keys
    if keys is None:
        tutor_model = ScriptedChatModel(fake.tutor_reply, delay_s=0.05)
        diagram_model = ScriptedChatModel(fake.diagram_reply, delay_s=4.0)
        synthesizer = FakeSynthesizer(seconds_per_word=0.4)
        on_microphone = _ignore
    else:
        anthropic = AnthropicChatModel(AnthropicConfig(api_key=keys.anthropic))
        tutor_model = diagram_model = anthropic
        vendor_clients.append(anthropic.aclose)
        if settings.diagram_model == "gemini":
            gemini = GeminiChatModel(GeminiConfig(api_key=keys.gemini))
            diagram_model = gemini
            vendor_clients.append(gemini.aclose)
        elevenlabs = ElevenLabsSynthesizer(ElevenLabsTtsConfig(api_key=keys.elevenlabs))
        synthesizer = elevenlabs
        vendor_clients.append(elevenlabs.aclose)
        recognizer = ElevenLabsRecognizer(ElevenLabsSttConfig(api_key=keys.elevenlabs))
        on_microphone = Listener(recognizer, bus.publisher(StudentVoice, "room").publish).listen

    audio = BrowserAudio(on_microphone)
    return VoiceTutorApp(
        bus=bus,
        audio=audio,
        tutor=Tutor(bus, model=tutor_model, voice=Speaker(synthesizer, audio)),
        diagrammer=Diagrammer(bus, model=diagram_model),
        typed=bus.publisher(StudentTyped, "ui"),
        clear_board=bus.publisher(BoardClearRequested, "ui"),
        speech=SpeechSimulator(bus.publisher(StudentVoice, "room")),
        fake=keys is None,
        vendor_clients=vendor_clients,
    )


async def _ignore(microphone: Microphone) -> None:
    while await microphone.get() is not None:
        pass
