"""An in-process publish/subscribe bus.

Every message type is a topic. A topic names the one owner allowed to publish
it and the owners allowed to consume it. Publishing is synchronous, and
messages are delivered to subscribers in one global order: a message published
while another is being delivered waits until that delivery finishes.
"""

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TopicSpec:
    publisher: str
    consumers: tuple[str, ...]


_TOPICS: dict[type, TopicSpec] = {}


def topic[T: type](*, publisher: str, consumers: tuple[str, ...]) -> Callable[[T], T]:
    """Declare a message class as a topic."""

    def declare(message_type: T) -> T:
        _TOPICS[message_type] = TopicSpec(publisher, consumers)
        return message_type

    return declare


def spec_of(message_type: type) -> TopicSpec:
    try:
        return _TOPICS[message_type]
    except KeyError:
        raise TypeError(f"{message_type.__name__} is not a topic") from None


class Publisher[T]:
    def __init__(self, message_type: type[T], send: Callable[[object], None]) -> None:
        self._message_type = message_type
        self._send = send

    def publish(self, message: T) -> None:
        if type(message) is not self._message_type:
            raise TypeError(f"expected {self._message_type.__name__}, got {type(message).__name__}")
        self._send(message)


class Bus:
    def __init__(self) -> None:
        self._handlers: dict[type, list[Callable[[Any], None]]] = {}
        self._pending: deque[object] = deque()
        self._delivering = False

    def publisher[T](self, message_type: type[T], owner: str) -> Publisher[T]:
        spec = spec_of(message_type)
        if owner != spec.publisher:
            raise PermissionError(
                f"{owner} may not publish {message_type.__name__}; only {spec.publisher} may"
            )
        return Publisher(message_type, self._send)

    def subscribe[T](
        self, message_type: type[T], owner: str, handler: Callable[[T], None]
    ) -> Callable[[], None]:
        """Call `handler` with every message of this topic. Returns an unsubscribe function."""
        spec = spec_of(message_type)
        if owner not in spec.consumers:
            raise PermissionError(f"{owner} is not a consumer of {message_type.__name__}")
        handlers = self._handlers.setdefault(message_type, [])
        handlers.append(handler)
        return lambda: handlers.remove(handler)

    def _send(self, message: object) -> None:
        self._pending.append(message)
        if self._delivering:
            return
        self._delivering = True
        try:
            while self._pending:
                self._deliver(self._pending.popleft())
        finally:
            self._delivering = False

    def _deliver(self, message: object) -> None:
        for handler in tuple(self._handlers.get(type(message), ())):
            try:
                handler(message)
            except Exception:
                logger.exception("a %s handler failed", type(message).__name__)
