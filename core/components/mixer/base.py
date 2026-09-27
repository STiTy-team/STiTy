from core.utils.audio import MultiChannelAudio

from ..registry import Component


class Mixer(Component):

    def mix(self, audio: MultiChannelAudio) -> bytes:
        raise NotImplementedError
