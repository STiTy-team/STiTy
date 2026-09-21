from ..registry import Component, Speech

__all__ = ["Detector", "Speech"]


class Detector(Component):

    def detect(self, audio: bytes) -> Speech | None:
        raise NotImplementedError
