class Fence:
    """Lets a turn's effects through until the turn has ended.

    A model stream or tool call can outlive the turn that started it, for
    example when a turn is slow to stop. Effects check the fence first, so a
    turn that has ended can no longer speak, publish, or change history.
    """

    def __init__(self) -> None:
        self._live = True

    @property
    def live(self) -> bool:
        return self._live

    def revoke(self) -> None:
        self._live = False
