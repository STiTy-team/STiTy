"""Human lines. `log.debug`, `log.info`, `log.warning`, `log.error` -- nothing else.

A line starts with its tag in brackets and reads as a sentence:

    log.info("[COMMIT-SKIP] reason=%s text=%r", reason, shown)

The tag is lifted out into a field of its own on the way to the stream, so it can be
filtered on without anyone parsing prose. Numbers that get plotted or scored do not
belong in the sentence -- those are data, and data is returned, not logged.
"""
import logging
import re
# re-exported so `from core.utils import logging` is a superset of the parts of the
# standard module app code touches, and nothing has to import both.
from logging import (  # noqa: F401
    CRITICAL, DEBUG, ERROR, INFO, WARNING, getLogger,
)

from rich.console import Console
from rich.markup import escape

from core.utils import stream

STREAM_LEVEL = INFO

TAG = re.compile(r"^\[([A-Z][A-Z0-9_-]*)\]\s*")

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

LEVEL_STYLES = {"WARNING": "yellow", "ERROR": "bold red", "CRITICAL": "bold red"}


def split_tag(message: str) -> tuple[str, str]:
    found = TAG.match(message)
    return ("", message) if found is None else (found.group(1), message[found.end():])


class RecordHandler(logging.Handler):
    """Every line also lands in the run's stream, tag split off from the prose."""

    def emit(self, record: logging.LogRecord) -> None:
        tag, message = split_tag(record.getMessage())
        fields = {"level": record.levelname, "logger": record.name, "msg": message}
        if tag:
            fields["tag"] = tag
        if record.exc_info:
            fields["exc"] = self.format(record) if self.formatter else None
        stream.record("log", **fields)


class ConsoleHandler(logging.Handler):

    def __init__(self, console: Console | None = None):
        super().__init__()
        self.console = console or Console(stderr=True)

    def emit(self, record: logging.LogRecord) -> None:
        tag, message = split_tag(record.getMessage())
        style = LEVEL_STYLES.get(record.levelname, "dim")
        at = stream.elapsed()
        stamp = f"{at:7.3f}" if at is not None else "      -"
        head = f"[cyan]{escape(tag)}[/] " if tag else ""
        self.console.print(f"[dim]{stamp}[/]  [{style}]{record.levelname[0]}[/] "
                           f"{head}{escape(message)}", highlight=False)


def _level(value) -> int:
    if isinstance(value, int):
        return value
    return getattr(logging, str(value).upper(), INFO)


def configure(*, console_level="INFO") -> None:
    """Entry points call this once, first thing.

    The console follows the level asked for; the stream is pinned at INFO, because a
    quieter console must not mean a thinner record.
    """
    console = ConsoleHandler()
    console.setLevel(_level(console_level))
    to_stream = RecordHandler()
    to_stream.setLevel(STREAM_LEVEL)
    root = getLogger()
    root.setLevel(min(console.level, STREAM_LEVEL))
    root.addHandler(console)
    root.addHandler(to_stream)
    for name in NOISY_LOGGERS:
        getLogger(name).setLevel(WARNING)
