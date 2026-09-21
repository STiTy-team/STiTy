import json
import logging
import time
from contextvars import ContextVar
# re-exported so `from core.utils import logging` is a superset of the parts of the
# standard module app code touches, and nothing has to import both.
from logging import (  # noqa: F401
    CRITICAL, DEBUG, ERROR, INFO, WARNING, Formatter, Handler, LogRecord, getLogger,
)
from pathlib import Path

from rich.cells import cell_len
from rich.console import Console
from rich.markup import escape

EVENT_KEY = "stity_event"
STREAM_LEVEL = INFO

NOISY_LOGGERS = (
    "vllm",
    "torch",
    "transformers",
    "httpx",
    "httpcore",
    "openai",
    "filelock",
    "urllib3",
    "asyncio",
    "numba",
    "matplotlib",
)

_LABELS = {
    name: ContextVar(f"stity_log_{name}", default="")
    for name in ("run", "session", "item")
}
_ORIGIN: ContextVar[float | None] = ContextVar("stity_log_origin", default=None)
_events = logging.getLogger("stity.event")


def _check(labels: dict) -> dict:
    unknown = sorted(set(labels) - set(_LABELS))
    if unknown:
        raise ValueError(f"unknown log labels: {unknown} (known: {sorted(_LABELS)})")
    return labels


def bind(**labels: str) -> None:
    """Set the labels stamped onto every event. `""` clears one."""
    for name, value in _check(labels).items():
        _LABELS[name].set(value)


def _current() -> dict:
    return {name: var.get() for name, var in _LABELS.items()}


_AUDIO: ContextVar[float | None] = ContextVar("stity_audio", default=None)


def set_audio_position(seconds: float | None) -> None:
    _AUDIO.set(seconds)


def start_clock() -> None:
    _ORIGIN.set(time.perf_counter())


def stop_clock() -> None:
    _ORIGIN.set(None)


def _elapsed() -> float | None:
    """Seconds since start_clock(), which every entry point sets per item."""
    origin = _ORIGIN.get()
    return None if origin is None else round(time.perf_counter() - origin, 4)


_COLLECTORS: list = []


def emit(type: str, **fields) -> dict:
    event = {"t": _elapsed(), "type": type}
    audio = _AUDIO.get()
    if audio is not None:
        event["audio"] = round(audio, 3)
    labels = _current()
    for name in ("session", "item"):
        if labels[name]:
            event[name] = labels[name]
    event.update(fields)
    _events.info("", extra={EVENT_KEY: event})
    for collector in _COLLECTORS:
        collector.offer(event)
    return event


class JsonlFormatter(logging.Formatter):
    """`enrich` promotes structure out of a free-text line without this knowing its vocabulary."""

    def __init__(self, *, enrich=None):
        super().__init__()
        self._enrich = enrich

    def format(self, record: logging.LogRecord) -> str:
        event = getattr(record, EVENT_KEY, None)
        if event is None:
            event = {
                "t": _elapsed(),
                "type": "log",
                "level": record.levelname,
                "logger": record.name,
                "msg": record.getMessage(),
            }
            item = _current()["item"]
            if item:
                event["item"] = item
            if self._enrich is not None:
                event.update(self._enrich(record.getMessage()) or {})
            if record.exc_info:
                event["exc"] = self.formatException(record.exc_info)
        return json.dumps(event, ensure_ascii=False, default=str)


def _pad(text: str, width: int) -> str:
    """Pad to a display width. Korean and Japanese cells are two columns wide."""
    return text + " " * max(0, width - cell_len(text))


class ConsoleHandler(logging.Handler):
    """Events as aligned coloured columns, plain log lines as text."""

    STYLES = {"WARNING": "yellow", "ERROR": "bold red", "CRITICAL": "bold red"}

    def __init__(self, console: Console | None = None):
        super().__init__()
        self.console = console or Console(stderr=True)

    def emit(self, record: logging.LogRecord) -> None:
        event = getattr(record, EVENT_KEY, None)
        if event is None:
            style = self.STYLES.get(record.levelname, "dim")
            self.console.print(
                f"[{style}]{record.levelname}[/] " f"{escape(record.getMessage())}",
                highlight=False,
            )
            return
        t = event.get("t")
        stamp = f"{t:7.3f}" if isinstance(t, (int, float)) else "      -"
        detail = str(event.get("output") or event.get("msg") or "")
        if cell_len(detail) > 80:
            detail = detail[:77] + "\u2026"
        self.console.print(
            f"[dim]{stamp}[/]  [cyan]{_pad(event['type'], 14)}[/] "
            f"[magenta]{_pad(event.get('item') or event.get('session') or '', 24)}[/] "
            f"{escape(detail)}",
            highlight=False,
        )


def _level(value) -> int:
    if isinstance(value, int):
        return value
    return getattr(logging, str(value).upper(), logging.INFO)


def configure(*, console_level="INFO") -> None:
    """Console output and the noisy-logger clamp. Entry points call this once, first thing.

    Separate from attach_stream() because this needs nothing but a level, while the
    run's stream needs a path that is not known until its config has been read.
    """
    console = ConsoleHandler()
    console.setLevel(_level(console_level))
    root = logging.getLogger()
    root.setLevel(console.level)
    root.addHandler(console)
    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


def attach_stream(path, *, enrich=None) -> Handler:
    """Add a run's append-only JSONL stream. Its level is fixed at STREAM_LEVEL:
    a quieter console must not mean a thinner record."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, mode="a", encoding="utf-8")
    handler.setFormatter(JsonlFormatter(enrich=enrich))
    handler.setLevel(STREAM_LEVEL)
    root = logging.getLogger()
    root.setLevel(min(root.level, STREAM_LEVEL))
    root.addHandler(handler)
    return handler


class collect:

    def __init__(self, *types: str):
        self.types = set(types)
        self.events: list[dict] = []

    def __enter__(self) -> list:
        _COLLECTORS.append(self)
        return self.events

    def __exit__(self, *exc) -> None:
        _COLLECTORS.remove(self)

    def offer(self, event: dict) -> None:
        if not self.types or event.get("type") in self.types:
            self.events.append(dict(event))


def read_stream(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
