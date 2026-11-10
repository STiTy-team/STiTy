from dataclasses import dataclass

import numpy as np

from core.utils.audio import MultiChannelAudio


@dataclass(frozen=True)
class SegmentSource:
    participant_id: str
    share: float


class SourceLevels:
    """How much of a segment's signal energy each client's microphone carried."""

    def __init__(self) -> None:
        self._energy: dict[str, float] = {}

    def add(self, audio: MultiChannelAudio) -> None:
        for channel, samples in zip(audio.channels, audio.samples):
            energy = float(np.dot(samples, samples))
            self._energy[channel] = self._energy.get(channel, 0.0) + energy

    def take(self) -> tuple[SegmentSource, ...]:
        energy, self._energy = self._energy, {}
        total = sum(energy.values())
        if total <= 0:
            return ()
        ranked = sorted(energy.items(), key=lambda item: item[1], reverse=True)
        return tuple(SegmentSource(channel, round(value / total, 3)) for channel, value in ranked)
