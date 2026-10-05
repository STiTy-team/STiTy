from ..registry import Component, Speech

__all__ = ["Detector", "Speech"]


class Detector(Component):

    spans: list
    # End (seconds) of the last window heard as speech; None when the detector cannot tell.
    voiced_until: float | None = None

    def start(self, **_) -> None:
        self.spans = []

    def detect(self, audio: bytes) -> Speech | None:
        raise NotImplementedError
