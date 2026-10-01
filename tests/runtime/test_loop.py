import asyncio

import pytest

from services.shared.runtime.fence import Fence
from services.shared.runtime.loop import Idle, Loop, Stopping, TurnStopTimeout
from services.shared.runtime.policy import Policy, Turn, TurnEnded


class GatedTurn(Turn[str]):
    """Runs until released, or until stopped when it honors stops."""

    def __init__(self, turn_id: str, *, honors_stop: bool = True) -> None:
        super().__init__(turn_id)
        self.release = asyncio.Event()
        self.honors_stop = honors_stop
        self.fence: Fence | None = None
        self.result = f"{turn_id} done"

    async def run(self, fence: Fence) -> str:
        self.fence = fence
        await self.release.wait()
        if self.result == "fail":
            raise RuntimeError("turn failed")
        return self.result

    def request_stop(self) -> None:
        if self.honors_stop:
            self.result = f"{self.turn_id} stopped"
            self.release.set()


class RecordingPolicy(Policy[str]):
    def __init__(self) -> None:
        self.stimuli: list[object] = []
        self.ended: list[TurnEnded[str]] = []
        self.on_stimulus = lambda stimulus, loop: None

    def verdict(self, stimulus: object, loop: Loop[str]) -> None:
        self.stimuli.append(stimulus)
        self.on_stimulus(stimulus, loop)

    def boundary(self, ended: TurnEnded[str], loop: Loop[str]) -> None:
        self.ended.append(ended)


async def settle() -> None:
    for _ in range(5):
        await asyncio.sleep(0)


@pytest.fixture
async def loop():
    policy = RecordingPolicy()
    loop = Loop("test", policy, stop_timeout_s=0.05)
    loop.start()
    yield loop, policy
    await loop.aclose()


async def test_stimuli_are_handled_in_arrival_order(loop) -> None:
    loop_, policy = loop
    for n in range(5):
        loop_.enqueue(n)
    await settle()
    assert policy.stimuli == [0, 1, 2, 3, 4]


async def test_only_one_turn_runs_at_a_time(loop) -> None:
    loop_, _ = loop
    loop_.spawn(GatedTurn("t1"))
    with pytest.raises(RuntimeError):
        loop_.spawn(GatedTurn("t2"))


async def test_a_stopped_turn_settles_with_what_it_has_and_reaches_the_boundary(loop) -> None:
    loop_, policy = loop
    loop_.spawn(GatedTurn("t1"))
    loop_.stop_current("test")
    assert isinstance(loop_.state, Stopping)
    await settle()
    assert [ended.settlement for ended in policy.ended] == ["t1 stopped"]
    assert isinstance(loop_.state, Idle)


async def test_a_turn_that_ignores_a_stop_is_abandoned_after_the_timeout(loop) -> None:
    loop_, policy = loop
    turn = GatedTurn("t1", honors_stop=False)
    loop_.spawn(turn)
    loop_.stop_current("test")
    await asyncio.sleep(0.1)
    assert len(policy.ended) == 1
    assert isinstance(policy.ended[0].error, TurnStopTimeout)
    assert isinstance(loop_.state, Idle)

    loop_.spawn(GatedTurn("t2"))
    turn.release.set()
    await settle()
    assert len(policy.ended) == 1


async def test_a_failed_turn_reaches_the_boundary_and_the_loop_keeps_going(loop) -> None:
    loop_, policy = loop
    turn = GatedTurn("t1")
    turn.result = "fail"
    loop_.spawn(turn)
    turn.release.set()
    await settle()
    assert isinstance(policy.ended[0].error, RuntimeError)

    loop_.enqueue("next")
    await settle()
    assert policy.stimuli == ["next"]


async def test_the_fence_closes_when_the_turn_ends(loop) -> None:
    loop_, _ = loop
    turn = GatedTurn("t1")
    loop_.spawn(turn)
    await settle()
    assert turn.fence is not None and turn.fence.live
    turn.release.set()
    await settle()
    assert not turn.fence.live


async def test_a_policy_error_does_not_stop_the_loop(loop) -> None:
    loop_, policy = loop

    def explode(stimulus: object, _: Loop[str]) -> None:
        if stimulus == "bad":
            raise RuntimeError("policy bug")

    policy.on_stimulus = explode
    loop_.enqueue("bad")
    loop_.enqueue("good")
    await settle()
    assert policy.stimuli == ["bad", "good"]


async def test_closing_cancels_the_running_turn_and_drops_later_stimuli() -> None:
    policy = RecordingPolicy()
    loop = Loop("test", policy)
    loop.start()
    turn = GatedTurn("t1", honors_stop=False)
    loop.spawn(turn)
    await settle()
    await loop.aclose()
    loop.enqueue("late")
    await settle()
    assert policy.stimuli == []
    assert turn.fence is not None and not turn.fence.live
