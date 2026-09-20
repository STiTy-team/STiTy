from ..registry import Component, Speech

__all__ = ["Detector", "Speech"]


class Detector(Component):

    spans: list

    def start(self, **_) -> None:
        self.spans = []

    def detect(self, audio: bytes) -> Speech | None:
        raise NotImplementedError
