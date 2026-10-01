class ProviderError(Exception):
    """A vendor call failed. The vendor's own exception is the `__cause__`."""

    def __init__(self, vendor: str, message: str) -> None:
        super().__init__(f"{vendor}: {message}")
        self.vendor = vendor
