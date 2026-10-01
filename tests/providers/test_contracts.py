"""Every provider implementation, fakes included, against its kind's contract.

Vendor subjects are marked `live`; run them with `uv run pytest -m live`.
"""

from __future__ import annotations

import os
from collections.abc import Awaitable, Callable, Sequence
from contextlib import AbstractAsyncContextManager, aclosing, nullcontext
from pathlib import Path

import pytest

from providers.llm.anthropic import AnthropicChatModel, AnthropicConfig
from providers.llm.base import CallsTools, ChatModel
from providers.llm.gemini import GeminiChatModel, GeminiConfig
from providers.stt.base import SpeechRecognizer
from providers.stt.elevenlabs import ElevenLabsRecognizer, ElevenLabsSttConfig
from providers.testing.contracts import CHAT_CHECKS, STT_CHECKS, TOOL_CHECKS, TTS_CHECKS, Utterance
from providers.testing.fakes import FakeRecognizer, FakeSynthesizer, ScriptedChatModel
from providers.tts.base import SpeechSynthesizer
from providers.tts.elevenlabs import ElevenLabsSynthesizer, ElevenLabsTtsConfig

type Make[S] = Callable[[], AbstractAsyncContextManager[S]]

_FIXTURES = Path(__file__).parents[1] / "fixtures"
_SPEECH = Utterance(
    pcm=(_FIXTURES / "quick_brown_fox.pcm").read_bytes(),
    text=(_FIXTURES / "quick_brown_fox.txt").read_text().strip(),
)
_LIVE = pytest.mark.live


def _key(name: str) -> str:
    if key := os.environ.get(name):
        return key
    pytest.fail(f"{name} is not set; live tests read it from the environment or .env")


def _anthropic() -> AbstractAsyncContextManager[AnthropicChatModel]:
    return aclosing(AnthropicChatModel(AnthropicConfig(api_key=_key("ANTHROPIC_API_KEY"))))


def _gemini() -> AbstractAsyncContextManager[GeminiChatModel]:
    return aclosing(GeminiChatModel(GeminiConfig(api_key=_key("GEMINI_API_KEY"))))


def _elevenlabs_tts() -> AbstractAsyncContextManager[ElevenLabsSynthesizer]:
    config = ElevenLabsTtsConfig(api_key=_key("ELEVENLABS_API_KEY"))
    return aclosing(ElevenLabsSynthesizer(config))


def _elevenlabs_stt() -> AbstractAsyncContextManager[ElevenLabsRecognizer]:
    config = ElevenLabsSttConfig(api_key=_key("ELEVENLABS_API_KEY"))
    return nullcontext(ElevenLabsRecognizer(config))


_SCRIPTED = pytest.param(lambda: nullcontext(ScriptedChatModel()), id="scripted")
_CHAT_MODELS = [
    _SCRIPTED,
    pytest.param(_anthropic, id="anthropic", marks=_LIVE),
    pytest.param(_gemini, id="gemini", marks=_LIVE),
]
_TOOL_MODELS = [_SCRIPTED, pytest.param(_anthropic, id="anthropic", marks=_LIVE)]
_SYNTHESIZERS = [
    pytest.param(lambda: nullcontext(FakeSynthesizer()), id="fake"),
    pytest.param(_elevenlabs_tts, id="elevenlabs", marks=_LIVE),
]
_RECOGNIZERS = [
    pytest.param(lambda: nullcontext(FakeRecognizer(_SPEECH.text)), id="fake"),
    pytest.param(_elevenlabs_stt, id="elevenlabs", marks=_LIVE),
]


def _named(checks: Sequence[tuple[str, object]]) -> list[object]:
    return [pytest.param(check, id=name) for name, check in checks]


@pytest.mark.parametrize("check", _named(CHAT_CHECKS))
@pytest.mark.parametrize("make", _CHAT_MODELS)
async def test_chat_model(
    make: Make[ChatModel], check: Callable[[ChatModel], Awaitable[None]]
) -> None:
    async with make() as model:
        await check(model)


@pytest.mark.parametrize("check", _named(TOOL_CHECKS))
@pytest.mark.parametrize("make", _TOOL_MODELS)
async def test_tool_calling_model(
    make: Make[CallsTools], check: Callable[[CallsTools], Awaitable[None]]
) -> None:
    async with make() as model:
        await check(model)


@pytest.mark.parametrize("check", _named(TTS_CHECKS))
@pytest.mark.parametrize("make", _SYNTHESIZERS)
async def test_speech_synthesizer(
    make: Make[SpeechSynthesizer], check: Callable[[SpeechSynthesizer], Awaitable[None]]
) -> None:
    async with make() as synthesizer:
        await check(synthesizer)


@pytest.mark.parametrize("check", _named(STT_CHECKS))
@pytest.mark.parametrize("make", _RECOGNIZERS)
async def test_speech_recognizer(
    make: Make[SpeechRecognizer],
    check: Callable[[SpeechRecognizer, Utterance], Awaitable[None]],
) -> None:
    async with make() as recognizer:
        await check(recognizer, _SPEECH)
