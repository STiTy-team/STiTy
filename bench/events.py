"""Event sourcing over the standard logging module.

One append-only JSONL stream is the run's source of truth; results.json and the
replay file are projections of it. Three consequences are the reason for doing it
this way rather than collecting events in memory:

  - A killed run keeps everything written so far. Writing at the end loses the
    whole record, including what was spent.
  - Selecting the top-k items needs no running bookkeeping. Write all of it and
    pick afterwards.
  - Server log lines land in the same stream on the same clock. That recovers the
    commits the server drops before send_message ever runs -- [SILENCE-DROP],
    [HALLUC-DROP], [EMPTY-DROP], [TAIL-DROP], [AST-LATE] -- which a socket-shaped
    sink cannot see at all.

`t` is monotonic seconds since the current item's first audio chunk, matching
FSLStreamingHandler.stream_start_perf. Every server-reported duration is measured
from that origin, so the fsl/laal cross-checks only hold on this clock. Wall clock
is recorded once per item instead.
"""
import json
import logging
import re
import time
from pathlib import Path

EVENT_KEY = "bench_event"
EVENT_LOGGER = "bench.event"

# Server log lines worth promoting out of free text. Each marks a commit the
# server decided not to send, or an attribution warning -- all invisible to a sink.
SERVER_TAGS = (
    "SILENCE-DROP", "HEADER-ONLY-DROP", "HALLUC-DROP", "EMPTY-DROP", "TAIL-DROP",
    "AST-LATE", "AST-OVERLAP", "AST-DRAIN", "VAD-SLOT-SWITCH", "DOT-SLOT-SWITCH",
    "SEG-SLOT-SWITCH", "CAP-FREEZE", "TRANS-ALERT",
)
_TAG_RE = re.compile(r"\[(" + "|".join(SERVER_TAGS) + r")\]\s*(.*)", re.DOTALL)

_CAMEL_RE = re.compile(r"(?<!^)(?=[A-Z])")


def camel_to_snake(name: str) -> str:
    if "_" in name or name.islower():
        return name.lower()
    return _CAMEL_RE.sub("_", name).lower()


def normalize_keys(payload: dict) -> dict:
    """camelCase -> snake_case, recursively.

    The three handler layers mix conventions (commitReason, decisionAudioSec,
    fsl_sec, segmentId). Both existing clients hand-rolled a field-by-field
    mapping; converting generically covers every field without an enumeration to
    keep in sync.
    """
    out = {}
    for key, value in payload.items():
        new_key = camel_to_snake(key)
        if isinstance(value, dict):
            value = normalize_keys(value)
        elif isinstance(value, list):
            value = [normalize_keys(v) if isinstance(v, dict) else v for v in value]
        out[new_key] = value
    return out


class Clock:
    """Per-item monotonic origin."""

    def __init__(self):
        self._origin: float | None = None

    def start(self) -> None:
        self._origin = time.perf_counter()

    def stop(self) -> None:
        self._origin = None

    def elapsed(self) -> float | None:
        if self._origin is None:
            return None
        return round(time.perf_counter() - self._origin, 4)


class Recorder:
    """Holds the clock and the current item/session labels."""

    def __init__(self):
        self.clock = Clock()
        self.run = ""
        self.session = ""
        self.item = ""
        self._logger = logging.getLogger(EVENT_LOGGER)

    def bind(self, *, run: str | None = None, session: str | None = None,
             item: str | None = None) -> None:
        if run is not None:
            self.run = run
        if session is not None:
            self.session = session
        if item is not None:
            self.item = item

    def emit(self, type: str, **fields) -> dict:
        event = {"t": self.clock.elapsed(), "type": type}
        if self.session:
            event["session"] = self.session
        if self.item:
            event["item"] = self.item
        event.update(fields)
        self._logger.info("", extra={EVENT_KEY: event})
        return event

    def emit_server_message(self, payload: dict) -> dict:
        """A message the handler tried to send. Normalized, then typed.

        A `final` carries both a transcription and a translation; it is emitted as
        one event with both, plus the timing fields the metrics need. Splitting it
        into two events would duplicate every timing field or lose it.
        """
        norm = normalize_keys(payload)
        msg_type = norm.pop("type", "message")
        return self.emit(msg_type, **norm)


_recorder = Recorder()


def recorder() -> Recorder:
    return _recorder


def emit(type: str, **fields) -> dict:
    return _recorder.emit(type, **fields)


class EventSink:
    """Stands in for the websocket.

    The handler's only egress is `await self.websocket.send(json_string)`; the two
    other socket touchpoints (remote_address, async-for) live in handle(), which
    the driver never calls. ASTStreamingHandler wraps whatever it is given in
    _StartSniffingWS, whose __getattr__ forwards here, so this must not define
    `state` or `_sniff`.
    """

    remote_address = ("in-process", 0)

    def __init__(self, rec: Recorder | None = None):
        self._rec = rec or _recorder
        self.n_messages = 0
        self.finals: list[dict] = []
        self.partials: int = 0
        # Trailing silence stops as soon as VAD reports nothing left, so short
        # clips do not pay the whole wait.
        self.vad_dones: list[dict] = []

    async def send(self, raw: str) -> None:
        self.n_messages += 1
        try:
            payload = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            self._rec.emit("sink_error", raw=str(raw)[:200])
            return
        event = self._rec.emit_server_message(payload)
        kind = event.get("type")
        if kind == "final":
            self.finals.append(event)
        elif kind == "partial":
            self.partials += 1
        elif kind == "vad_done":
            self.vad_dones.append(event)

    def reset_item(self) -> None:
        self.finals = []
        self.vad_dones = []


class JsonlFormatter(logging.Formatter):
    """One JSON object per line, for bench events and server log lines alike.

    A bench event carries its own clock reading and labels. A plain log line has
    neither, so they come from the recorder -- which is injected rather than
    taken from the module global, so a test (or a second run in one process) can
    use its own.
    """

    def __init__(self, rec: "Recorder | None" = None):
        super().__init__()
        self._rec = rec

    def format(self, record: logging.LogRecord) -> str:
        event = getattr(record, EVENT_KEY, None)
        if event is not None:
            return json.dumps(event, ensure_ascii=False, default=str)

        rec = self._rec or _recorder
        message = record.getMessage()
        entry = {
            "t": rec.clock.elapsed(),
            "type": "log",
            "level": record.levelname,
            "logger": record.name,
            "msg": message,
        }
        if rec.item:
            entry["item"] = rec.item
        match = _TAG_RE.search(message)
        if match:
            entry["type"] = "server_note"
            entry["tag"] = match.group(1)
            entry["msg"] = match.group(2).strip()
        if record.exc_info:
            entry["exc"] = self.formatException(record.exc_info)
        return json.dumps(entry, ensure_ascii=False, default=str)


class ConsoleFormatter(logging.Formatter):
    """Readable console lines; bench events collapse to one line."""

    def format(self, record: logging.LogRecord) -> str:
        event = getattr(record, EVENT_KEY, None)
        if event is None:
            return f"{record.levelname}\t{record.getMessage()}"
        t = event.get("t")
        stamp = f"{t:7.3f}" if isinstance(t, (int, float)) else "      -"
        label = event.get("item") or event.get("session") or ""
        detail = event.get("output") or event.get("msg") or ""
        if isinstance(detail, str) and len(detail) > 80:
            detail = detail[:77] + "..."
        return f"{stamp}  {event['type']:<14} {label:<24} {detail}"


def configure(path: Path, *, console_level: str = "INFO",
              file_level: int = logging.INFO,
              rec: "Recorder | None" = None) -> logging.Handler:
    """Attach the JSONL stream to the root logger.

    Called before the ASR server module is imported, because that module's
    _configure_logging does root.handlers.clear() and would drop this handler.
    The driver never calls it, but the ordering is pinned by a test.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(path, mode="a", encoding="utf-8")
    file_handler.setFormatter(JsonlFormatter(rec))
    file_handler.setLevel(file_level)

    console = logging.StreamHandler()
    console.setFormatter(ConsoleFormatter())
    console.setLevel(getattr(logging, console_level, logging.INFO))

    root = logging.getLogger()
    root.setLevel(min(file_level, console.level))
    root.addHandler(file_handler)
    root.addHandler(console)
    # vLLM and friends are extremely chatty at INFO; they would bury the stream.
    for noisy in ("vllm", "torch", "transformers", "httpx", "httpcore", "openai",
                  "filelock", "urllib3", "asyncio", "numba", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return file_handler


def read_stream(path: Path):
    """Reads the JSONL stream back. The projections in report.py go through here."""
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue
