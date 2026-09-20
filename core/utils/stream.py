"""The run's machine-readable stream: one JSON object per line.

Three producers write here and nothing else does -- `logging` puts a human line in,
`timing` puts a measured duration in, and whatever drives a run puts the data it
collected in. A pipeline part never writes here; it returns its data and logs prose.

The clock lives here because every line is stamped with it. `t` is seconds since the
driver opened the item, `audio` is where in the recording that moment fell, and the
two together are what makes a run replayable faster than real time.
"""
import json
import time
from contextvars import ContextVar
from pathlib import Path

_ORIGIN: ContextVar[float | None] = ContextVar("stity_origin", default=None)
_AUDIO: ContextVar[float | None] = ContextVar("stity_audio", default=None)
_LABELS: ContextVar[dict] = ContextVar("stity_labels", default={})

_file = None


def attach(path) -> None:
    global _file
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    close()
    _file = open(path, "a", encoding="utf-8")


def close() -> None:
    global _file
    if _file is not None and not _file.closed:
        _file.close()
    _file = None


def bind(**labels: str) -> None:
    _LABELS.set({**_LABELS.get(), **labels})


def start_clock() -> None:
    _ORIGIN.set(time.perf_counter())


def stop_clock() -> None:
    _ORIGIN.set(None)


def elapsed() -> float | None:
    origin = _ORIGIN.get()
    return None if origin is None else round(time.perf_counter() - origin, 4)


def audio_position(seconds: float | None = ...) -> float | None:
    if seconds is not ...:
        _AUDIO.set(seconds)
    return _AUDIO.get()


def record(type: str, **fields) -> dict:
    line = {"t": elapsed(), "type": type}
    audio = _AUDIO.get()
    if audio is not None:
        line["audio"] = round(audio, 3)
    line.update({k: v for k, v in _LABELS.get().items() if v})
    line.update(fields)
    if _file is not None and not _file.closed:
        _file.write(json.dumps(line, ensure_ascii=False, default=str) + "\n")
        _file.flush()
    return line


def read(path):
    with open(path, encoding="utf-8") as f:
        for raw in f:
            if raw.strip():
                try:
                    yield json.loads(raw)
                except json.JSONDecodeError:
                    continue
