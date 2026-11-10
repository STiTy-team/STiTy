"""The conversation server's session recordings on this machine: list, replay, audio, delete.

The server writes each recording to `server/logs/recordings/<date>/<time>-<participant>/`
as `audio.wav`, `events.jsonl` and `meta.json`. They never leave this machine.
"""

import io
import re
import shutil
import time
import urllib.parse
import wave
from pathlib import Path

from core.errors import DataError
from core.utils.audio import SAMPLING_RATE
from core.utils.json import read_json, read_jsonl
from core.utils.paths import get_project_root
from pydantic.alias_generators import to_snake

RECORDINGS_DIR = get_project_root() / "server" / "logs" / "recordings"
AUDIO_API = "/api/logs/audio"
WAV_HEADER_BYTES = 44
SAMPLE_WIDTH = 2
LIVE_WITHIN_SEC = 300
ID_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}/[A-Za-z0-9._-]+$")


class RecordingError(Exception):
    def __init__(self, message: str, code: int):
        super().__init__(message)
        self.message = message
        self.code = code


def resolve(recording_id: str) -> Path:
    if not ID_PATTERN.match(recording_id):
        raise RecordingError(f"bad recording id {recording_id!r}", 400)
    directory = RECORDINGS_DIR / recording_id
    if not (directory / "meta.json").is_file():
        raise RecordingError(f"no recording {recording_id!r}", 404)
    return directory


def _audio_seconds(directory: Path, meta: dict) -> float:
    if meta.get("audioSeconds") is not None:
        return float(meta["audioSeconds"])
    wav = directory / "audio.wav"
    if not wav.is_file():
        return 0.0
    return max(0, wav.stat().st_size - WAV_HEADER_BYTES) / SAMPLE_WIDTH / SAMPLING_RATE


def _state(directory: Path, meta: dict) -> str:
    """`done` once the server closed it; `live` while it is still being written; else `unfinished`."""
    if meta.get("endedAt"):
        return "done"
    touched = max((f.stat().st_mtime for f in directory.iterdir() if f.is_file()), default=0)
    return "live" if time.time() - touched < LIVE_WITHIN_SEC else "unfinished"


def _summary(directory: Path) -> dict:
    meta = read_json(directory / "meta.json")
    files = [f for f in directory.iterdir() if f.is_file()]
    return {
        "id": f"{directory.parent.name}/{directory.name}",
        "participant_id": meta.get("participantId") or "",
        "pipeline": meta.get("pipeline") or "",
        "started_at": meta.get("startedAt"),
        "ended_at": meta.get("endedAt"),
        "audio_seconds": round(_audio_seconds(directory, meta), 3),
        "bytes": sum(f.stat().st_size for f in files),
        "state": _state(directory, meta),
    }


def list_recordings() -> dict:
    found = []
    if RECORDINGS_DIR.is_dir():
        for directory in RECORDINGS_DIR.glob("*/*"):
            if not (directory / "meta.json").is_file():
                continue
            try:
                found.append(_summary(directory))
            except (OSError, DataError, ValueError) as e:
                print(f"skipping recording {directory}: {e}")
    found.sort(key=lambda r: r["id"], reverse=True)
    return {"dir": str(RECORDINGS_DIR), "recordings": found}


def _snake(message: dict) -> dict:
    return {to_snake(k): v for k, v in message.items()}


def _replay_events(lines: list[dict], target_lang: str) -> list[dict]:
    """The recorded WebSocket traffic in the replay's event shape.

    A `final` for the session's target language becomes `translated`, what the phone
    draws; finals for other targets the client asked for stay `final` and show only in
    the event stream, so one segment is not drawn once per target.
    """
    events = [{"type": "item_open", "t": 0.0, "audio": 0.0}]
    for line in lines:
        message = _snake(line.get("message") or {})
        kind = message.pop("type", "")
        if kind == "final" and message.get("target_lang") == target_lang:
            kind = "translated"
        events.append(
            {
                **message,
                "type": kind,
                "direction": line.get("direction"),
                "t": line.get("t"),
                "audio": line.get("audioSeconds"),
            }
        )
    return events


def replay(recording_id: str) -> dict:
    directory = resolve(recording_id)
    meta = read_json(directory / "meta.json")
    events_path = directory / "events.jsonl"
    lines = list(read_jsonl(events_path)) if events_path.is_file() else []
    messages = [line.get("message") or {} for line in lines]
    start = next((m for m in messages if m.get("type") == "start"), {})
    finals = [m for m in messages if m.get("type") == "final"]
    target_lang = start.get("targetLang") or next((m.get("targetLang") for m in finals), "") or ""
    src_lang = start.get("lang") if start.get("lang") not in (None, "", "auto") else ""
    src_lang = src_lang or next((m.get("language") for m in finals), "") or ""

    events = _replay_events(lines, target_lang)
    duration = _audio_seconds(directory, meta)
    item = {
        "id": recording_id,
        "events": events,
        "duration": max([duration] + [e.get("audio") or 0.0 for e in events]),
        "src_lang": src_lang,
        "target_lang": target_lang,
        "reference": "",
        "reference_translations": {},
    }
    if (directory / "audio.wav").is_file():
        item["audio_url"] = f"{AUDIO_API}?{urllib.parse.urlencode({'id': recording_id})}"
    return {
        "recording": _summary(directory),
        "server_config": meta.get("serverConfig") or {},
        "item": item,
    }


def audio_bytes(recording_id: str) -> bytes:
    """The recording's WAV. One still being written has a header that does not yet count
    its frames, so its samples are re-wrapped under a fresh header."""
    directory = resolve(recording_id)
    wav = directory / "audio.wav"
    if not wav.is_file():
        raise RecordingError(f"{recording_id} has no audio", 404)
    raw = wav.read_bytes()
    if read_json(directory / "meta.json").get("endedAt"):
        return raw
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(SAMPLE_WIDTH)
        out.setframerate(SAMPLING_RATE)
        out.writeframes(raw[WAV_HEADER_BYTES:])
    return buffer.getvalue()


def delete(recording_id: str) -> dict:
    directory = resolve(recording_id)
    if _state(directory, read_json(directory / "meta.json")) == "live":
        raise RecordingError(f"{recording_id} is still being recorded", 409)
    shutil.rmtree(directory)
    day = directory.parent
    if day.is_dir() and not any(day.iterdir()):
        day.rmdir()
    return {"deleted": recording_id}
