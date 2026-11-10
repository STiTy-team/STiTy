import time
import wave
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np

from core.utils import logging
from core.utils.audio import SAMPLING_RATE, MultiChannelAudio, to_pcm_bytes
from core.utils.json import dumps, write_json

from core.pipeline import describe

from app.config import ServerConfig
from app.models.conversation import Conversation, Participant

log = logging.getLogger(__name__)

SAMPLE_WIDTH = 2
SILENCE = memoryview(bytes(1 << 20))


def _open_wav(path: Path) -> wave.Wave_write:
    audio = wave.open(str(path), "wb")
    audio.setnchannels(1)
    audio.setsampwidth(SAMPLE_WIDTH)
    audio.setframerate(SAMPLING_RATE)
    return audio


class SessionRecording:
    def __init__(self, directory: Path, meta: dict):
        directory.mkdir(parents=True, exist_ok=True)
        self.directory = directory
        self._meta = meta
        self._started = time.monotonic()
        self._frames = 0
        self._audio = _open_wav(directory / "audio.wav")
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


class RoomRecording:
    """One track per client plus the mix, all on the room's timeline.

    A client absent for a stretch gets silence there, so tracks line up sample for sample.
    """

    def __init__(self, directory: Path, meta: dict):
        (directory / "tracks").mkdir(parents=True, exist_ok=True)
        self.directory = directory
        self._meta = meta
        self._mix = _open_wav(directory / "mix.wav")
        self._tracks: dict[str, wave.Wave_write] = {}
        self._participants: dict[str, dict] = {}

    def write(self, audio: MultiChannelAudio, participants: Mapping[str, Participant]) -> None:
        for channel, samples in zip(audio.channels, audio.samples):
            if channel not in self._tracks:
                self._open_track(channel, participants.get(channel))
            self._pad(channel)
            self._tracks[channel].writeframes(to_pcm_bytes(samples))
        self._mix.writeframes(to_pcm_bytes(np.mean(audio.samples, axis=0)))

    def close(self) -> None:
        for channel, track in self._tracks.items():
            self._pad(channel)
            track.close()
        audio_seconds = self._mix.tell() / SAMPLING_RATE
        self._mix.close()
        write_json(
            self.directory / "room.json",
            {
                **self._meta,
                "endedAt": datetime.now().isoformat(timespec="seconds"),
                "audioSeconds": round(audio_seconds, 3),
                "participants": self._participants,
            },
        )

    def _open_track(self, channel: str, participant: Participant | None) -> None:
        self._tracks[channel] = _open_wav(self.directory / "tracks" / f"{channel}.wav")
        if participant is not None:
            self._participants[channel] = {
                "number": participant.number,
                "device": participant.device,
            }

    def _pad(self, channel: str) -> None:
        track = self._tracks[channel]
        missing = (self._mix.tell() - track.tell()) * SAMPLE_WIDTH
        while missing > 0:
            silence = SILENCE[:missing]
            track.writeframes(silence)
            missing -= len(silence)


class RoomRecorder:
    """Records a room while any client asks for it; a failure ends the recording, not the room."""

    def __init__(self, recordings: "RecordingService", conversation: Conversation):
        self._recordings = recordings
        self._conversation = conversation
        self._recording: RoomRecording | None = None
        self._failed = False

    def write(self, audio: MultiChannelAudio) -> None:
        if self._failed:
            return
        try:
            if not self._conversation.wants_recording:
                self.close()
            else:
                if self._recording is None:
                    self._recording = self._recordings.open_room(self._conversation.id)
                self._recording.write(audio, self._conversation.participants)
        except Exception:
            log.exception("Stopped recording room %s", self._conversation.id)
            self._failed = True
            self.close()

    def close(self) -> None:
        recording, self._recording = self._recording, None
        if recording is None:
            return
        try:
            recording.close()
        except Exception:
            log.exception("Failed to finish recording room %s", self._conversation.id)


class RecordingService:
    def __init__(self, config: ServerConfig):
        self._settings = config.recording
        self._pipeline = config.pipeline_name
        self._server_config = describe(config)

    def for_room(self, conversation: Conversation) -> RoomRecorder:
        return RoomRecorder(self, conversation)

    def open_room(self, conversation_id: str) -> RoomRecording:
        now = datetime.now()
        directory = self._directory(now, f"room-{conversation_id}")
        recording = RoomRecording(
            directory,
            {
                "conversationId": conversation_id,
                "pipeline": self._pipeline,
                "sampleRate": SAMPLING_RATE,
                "startedAt": now.isoformat(timespec="seconds"),
            },
        )
        log.info("Recording room %s to %s", conversation_id, directory)
        return recording

    def open(self, participant_id: str, device: dict | None = None) -> SessionRecording | None:
        if not self._settings.allowed:
            log.info("Refused to record participant %s: recording is not allowed", participant_id)
            return None
        now = datetime.now()
        directory = self._directory(now, participant_id)
        recording = SessionRecording(
            directory,
            {
                "participantId": participant_id,
                "device": device,
                "pipeline": self._pipeline,
                "serverConfig": self._server_config,
                "sampleRate": SAMPLING_RATE,
                "startedAt": now.isoformat(timespec="seconds"),
            },
        )
        log.info("Recording participant %s to %s", participant_id, directory)
        return recording

    def _directory(self, now: datetime, name: str) -> Path:
        day = self._settings.dir / now.strftime("%Y-%m-%d")
        return day / f"{now:%H%M%S}.{now.microsecond // 1000:03d}-{name}"
