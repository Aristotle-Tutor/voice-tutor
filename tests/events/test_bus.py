from dataclasses import dataclass

import pytest

from events.bus import Bus, topic


@topic(publisher="a", consumers=("b", "c"))
@dataclass(frozen=True)
class Ping:
    n: int


@topic(publisher="b", consumers=("c",))
@dataclass(frozen=True)
class Pong:
    n: int


def test_only_the_declared_publisher_may_publish() -> None:
    with pytest.raises(PermissionError):
        Bus().publisher(Ping, "b")


def test_only_declared_consumers_may_subscribe() -> None:
    with pytest.raises(PermissionError):
        Bus().subscribe(Pong, "b", lambda _: None)


def test_a_publish_during_delivery_waits_for_that_delivery_to_finish() -> None:
    bus = Bus()
    pong = bus.publisher(Pong, "b")
    seen: list[object] = []
    bus.subscribe(Ping, "b", lambda ping: pong.publish(Pong(ping.n)))
    bus.subscribe(Ping, "c", seen.append)
    bus.subscribe(Pong, "c", seen.append)

    bus.publisher(Ping, "a").publish(Ping(1))

    assert seen == [Ping(1), Pong(1)]


def test_a_failing_handler_does_not_stop_the_others() -> None:
    bus = Bus()
    seen: list[Ping] = []

    def fail(_: Ping) -> None:
        raise RuntimeError("boom")

    bus.subscribe(Ping, "b", fail)
    bus.subscribe(Ping, "c", seen.append)
    bus.publisher(Ping, "a").publish(Ping(1))

    assert seen == [Ping(1)]
