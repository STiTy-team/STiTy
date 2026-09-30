from dataclasses import dataclass

from ..registry import Component


@dataclass(frozen=True)
class Routed:
    vad: bytes
    asr: bytes


class Enhancer(Component):

    def enhance(self, pcm: bytes) -> Routed:
        raise NotImplementedError
