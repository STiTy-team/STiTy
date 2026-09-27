import asyncio
from collections import deque
from collections.abc import AsyncIterator

import numpy as np

from core.utils import clock
from core.utils.audio import SAMPLING_RATE, MultiChannelAudio, from_pcm_bytes

CHUNK_MS = 100
MAX_WAIT_MS = 200


class ChannelBuffer:
    def __init__(self) -> None:
        self._pieces: deque[tuple[float, np.ndarray]] = deque()
        self._size = 0

    @property
    def size(self) -> int:
        return self._size

    def push(self, samples: np.ndarray) -> None:
        self._pieces.append((clock.monotonic(), samples))
        self._size += len(samples)

    def arrived_at(self, n: int) -> float | None:
        if self._size < n:
            return None
        seen = 0
        for arrived, samples in self._pieces:
            seen += len(samples)
            if seen >= n:
                return arrived
        return None

    def take(self, n: int) -> np.ndarray:
        out = np.zeros(n, dtype=np.float32)
        filled = 0
        while filled < n and self._pieces:
            arrived, samples = self._pieces.popleft()
            used = min(n - filled, len(samples))
            out[filled : filled + used] = samples[:used]
            filled += used
            if used < len(samples):
                self._pieces.appendleft((arrived, samples[used:]))
        self._size -= filled
        return out


class ConversationAudio:
    def __init__(self) -> None:
        self._chunk = SAMPLING_RATE * CHUNK_MS // 1000
        self._max_wait = MAX_WAIT_MS / 1000
        self._buffers: dict[str, ChannelBuffer] = {}
        self._closed: set[str] = set()
        self._position = 0
        self._ending = False
        self._pushed = asyncio.Event()

    @property
    def seconds(self) -> float:
        return self._position / SAMPLING_RATE

    def open_channel(self, channel_id: str) -> None:
        self._closed.discard(channel_id)
        self._buffers.setdefault(channel_id, ChannelBuffer())

    def close_channel(self, channel_id: str) -> None:
        if channel_id in self._buffers:
            self._closed.add(channel_id)
            self._forget_drained()
        self._pushed.set()

    def push(self, channel_id: str, pcm: bytes) -> None:
        buffer = self._buffers.get(channel_id)
        if buffer is None or channel_id in self._closed or self._ending or not pcm:
            return
        buffer.push(from_pcm_bytes(pcm[: len(pcm) - len(pcm) % 2]))
        self._pushed.set()

    def end(self) -> None:
        self._ending = True
        self._pushed.set()

    async def chunks(self) -> AsyncIterator[MultiChannelAudio]:
        while True:
            wait = self._seconds_until_due()
            if wait == 0:
                yield self._next_chunk()
                continue
            if self._ending:
                return
            self._pushed.clear()
            try:
                await asyncio.wait_for(self._pushed.wait(), timeout=wait)
            except TimeoutError:
                pass

    def _seconds_until_due(self) -> float | None:
        if not any(buffer.size for buffer in self._buffers.values()):
            return None
        if self._ending:
            return 0
        arrivals = {c: buffer.arrived_at(self._chunk) for c, buffer in self._buffers.items()}
        ready = [t for t in arrivals.values() if t is not None]
        if not ready:
            return None
        if all(t is not None or c in self._closed for c, t in arrivals.items()):
            return 0
        return max(min(ready) + self._max_wait - clock.monotonic(), 0)

    def _next_chunk(self) -> MultiChannelAudio:
        channels = tuple(c for c, buffer in self._buffers.items() if buffer.size)
        rows = np.stack([self._buffers[c].take(self._chunk) for c in channels])
        self._forget_drained()
        start = self.seconds
        self._position += self._chunk
        return MultiChannelAudio(channels=channels, samples=rows, start=start)

    def _forget_drained(self) -> None:
        for channel_id in [c for c in self._closed if self._buffers[c].size == 0]:
            del self._buffers[channel_id]
            self._closed.discard(channel_id)
