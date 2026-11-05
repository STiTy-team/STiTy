import time
import wave
from datetime import datetime
from pathlib import Path
from typing import Literal

from core.utils import logging
from core.utils.audio import SAMPLING_RATE
from core.utils.json import dumps, write_json

from app.config import ServerConfig

log = logging.getLogger(__name__)

SAMPLE_WIDTH = 2


class SessionRecording:
    def __init__(self, directory: Path, meta: dict):
        directory.mkdir(parents=True, exist_ok=True)
        self.directory = directory
        self._meta = meta
        self._started = time.monotonic()
        self._frames = 0
        self._audio = wave.open(str(directory / "audio.wav"), "wb")
        self._audio.setnchannels(1)
        self._audio.setsampwidth(SAMPLE_WIDTH)
        self._audio.setframerate(SAMPLING_RATE)
        self._events = open(directory / "events.jsonl", "w", encoding="utf-8")
        write_json(directory / "meta.json", meta)

    @property
    def audio_seconds(self) -> float:
        return self._frames / SAMPLING_RATE

    def audio(self, pcm: bytes) -> None:
        self._audio.writeframes(pcm)
        self._frames += len(pcm) // SAMPLE_WIDTH

    def message(self, direction: Literal["in", "out"], message: dict) -> None:
        event = {
            "t": round(time.monotonic() - self._started, 3),
            "audioSeconds": round(self.audio_seconds, 3),
            "direction": direction,
            "message": message,
        }
        self._events.write(dumps(event) + "\n")
        self._events.flush()

    def close(self) -> None:
        self._audio.close()
        self._events.close()
        write_json(
            self.directory / "meta.json",
            {
                **self._meta,
                "endedAt": datetime.now().isoformat(timespec="seconds"),
                "audioSeconds": round(self.audio_seconds, 3),
            },
        )


class RecordingService:
    def __init__(self, config: ServerConfig):
        self._settings = config.recording
        self._pipeline = config.pipeline_name

    def open(self, participant_id: str) -> SessionRecording | None:
        if not self._settings.enabled:
            return None
        now = datetime.now()
        directory = (
            self._settings.dir / now.strftime("%Y-%m-%d") / f"{now:%H%M%S}-{participant_id}"
        )
        recording = SessionRecording(
            directory,
            {
                "participantId": participant_id,
                "pipeline": self._pipeline,
                "sampleRate": SAMPLING_RATE,
                "startedAt": now.isoformat(timespec="seconds"),
            },
        )
        log.info("Recording participant %s to %s", participant_id, directory)
        return recording
