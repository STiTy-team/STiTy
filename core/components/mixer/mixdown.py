import numpy as np

from core.utils.audio import MultiChannelAudio, to_pcm_bytes

from . import mixers
from .base import Mixer


@mixers.register("mixdown")
class Mixdown(Mixer):

    def mix(self, audio: MultiChannelAudio) -> bytes:
        if not audio.channels:
            return b""
        return to_pcm_bytes(np.mean(audio.samples, axis=0))
