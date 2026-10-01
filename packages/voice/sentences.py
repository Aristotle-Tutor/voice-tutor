from __future__ import annotations

import re

_SENTENCE_END = re.compile(r"[.!?]+[\"')\]]*\s+|\n+")


class SentenceBuffer:
    """Collects streamed text and hands it back one sentence at a time.

    No character is dropped: joined together, the sentences are exactly the
    text that was added.
    """

    def __init__(self) -> None:
        self._text = ""

    def add(self, text: str) -> list[str]:
        self._text += text
        sentences: list[str] = []
        while match := _SENTENCE_END.search(self._text):
            sentences.append(self._text[: match.end()])
            self._text = self._text[match.end() :]
        return sentences

    def flush(self) -> str:
        rest, self._text = self._text, ""
        return rest
