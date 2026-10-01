from __future__ import annotations

from collections.abc import Sequence

from providers.llm.base import Message, Text


def keep_heard(reply: Sequence[Message], heard_chars: int) -> tuple[Message, ...]:
    """Cut a reply down to what the student heard.

    The reply's assistant text, read across every message in order, is the
    text that was spoken, so the first `heard_chars` characters of it are what
    the student heard. Later text is dropped. Tool calls and their results are
    kept, because they really happened.
    """
    remaining = heard_chars
    kept: list[Message] = []
    for message in reply:
        parts = []
        for part in message.parts:
            if message.role == "assistant" and isinstance(part, Text):
                part = Text(part.text[:remaining])
                remaining -= len(part.text)
                if not part.text:
                    continue
            parts.append(part)
        if parts:
            kept.append(Message(message.role, tuple(parts)))
    return tuple(kept)
