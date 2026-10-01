"""A streaming ASR that runs in another process, reached over WebSocket.

Voxtral, Nemotron and WhisperLiveKit each need their own env (their dependencies
cannot share one), so each runs as a server and this part talks to it. The
server only transcribes. When to commit is decided here, by the same rules
the Qwen parts use, so that the only thing that differs between runs is the
recognizer:

- VAD silence (`flush("vad")`) commits everything shown and not yet committed.
- With `commit: punct`, a sentence-final mark in *confirmed* text commits up to it.
  Voxtral only ever appends, so all its text is confirmed. WLK
  (`buffer_transcription`) and Nemotron (`pending`) also send a tail that may
  still change; it is shown as a partial and committed on VAD silence, but
  never triggers a dot commit.

Protocols, as the servers actually speak them (see evaluation/backends/ on the
multi-asr-backends and wlk branches):

  voxtral   vLLM /v1/realtime. session.update{model} -> commit (starts generation)
            -> append{b64 pcm16} xN -> commit{final:true}. transcription.delta
            appends text, transcription.done ends. No language field exists.
  nemotron  evaluation/backends/server.py. hello -> start{lang} -> ready ->
            pcm16 bytes -> final{original, pending} / partial{text} -> finish ->
            finish_done. `original` is only ever new text; `pending` is the word
            the server holds until the next word starts (our patched copy).
  wlk       WhisperLiveKit /asr?language=xx with --pcm-input. pcm16 bytes ->
            {lines[], buffer_transcription} snapshots -> b"" -> ready_to_stop.
"""
import asyncio
import base64
import json
import re

from core.errors import ConfigError
from core.utils import audio as audio_mod
from core.utils import langs, logging, timing

from . import transcribers
from ..registry import Partial, Speech, Transcribed
from .base import Transcriber

log = logging.getLogger(__name__)

PROTOCOLS = ("voxtral", "nemotron", "wlk")
CONNECT_TIMEOUT_SEC = 60.0
END_TIMEOUT_SEC = 90.0
SENTENCE_END = re.compile(r"[.?!。？！]$")


def _protocol(value: str) -> str:
    if value not in PROTOCOLS:
        raise ConfigError(f"protocol {value!r} (available: {list(PROTOCOLS)})")
    return value


def _lines_text(lines) -> str:
    # speaker -2 marks silence in WLK and carries no text.
    return " ".join(t for t in ((ln.get("text") or "").strip()
                                for ln in lines or [] if ln.get("speaker") != -2) if t)


@transcribers.register("remote-stream")
class RemoteStreamTranscriber(Transcriber):
    SETTINGS = {
        "protocol": ("protocol", _protocol),
        "url": ("url", str),
        "model": ("model", str),
    }

    @classmethod
    def validate(cls, options: dict, *, kind: str = "transcription") -> dict:
        out = super().validate(options, kind=kind)
        for key in ("protocol", "url"):
            if key not in out:
                raise ConfigError(f"pipeline.{kind}: remote-stream needs {key!r}")
        if out["protocol"] == "voxtral" and "model" not in out:
            raise ConfigError(f"pipeline.{kind}: voxtral needs 'model'")
        return out

    async def load(self) -> None:
        # Fail before the run starts, not on the first item, when the server is down.
        ws = await self._connect()
        await ws.close()

    def start(self, languages: list[str] | None = None, **_) -> None:
        self.language = langs.norm_code((languages or [""])[0]) or ""
        self.ws = None
        self.reader = None
        self.confirmed = ""       # text the server will not revise
        self.tail = ""            # text the server may still revise (WLK, Nemotron)
        self.committed_words = 0
        self.fed_samples = 0
        self.partial_seq = 0
        self.partial_text = None
        self.done = asyncio.Event()
        self.out: list = []

    # -- connection ---------------------------------------------------------

    async def _connect(self):
        import websockets

        protocol, url = self.settings["protocol"], self.settings["url"]
        if protocol == "voxtral":
            url = f"{url}?model={self.settings['model']}"
        elif protocol == "wlk" and getattr(self, "language", ""):
            url = f"{url}?language={self.language}"
        return await asyncio.wait_for(
            websockets.connect(url, max_size=None, ping_interval=None,
                               open_timeout=CONNECT_TIMEOUT_SEC),
            timeout=CONNECT_TIMEOUT_SEC)

    async def _open(self) -> None:
        self.ws = await self._connect()
        protocol = self.settings["protocol"]
        if protocol == "voxtral":
            await self.ws.send(json.dumps({"type": "session.update",
                                           "model": self.settings["model"]}))
            await self.ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
        elif protocol == "nemotron":
            await self._expect("hello")
            await self.ws.send(json.dumps({"type": "start", "lang": self.language or "auto"}))
            await self._expect("ready")
        self.reader = asyncio.create_task(self._read())

    async def _expect(self, kind: str) -> None:
        while True:
            msg = json.loads(await asyncio.wait_for(self.ws.recv(), CONNECT_TIMEOUT_SEC))
            if msg.get("type") == kind:
                return

    async def _read(self) -> None:
        protocol = self.settings["protocol"]
        try:
            async for raw in self.ws:
                if isinstance(raw, bytes):
                    continue
                msg = json.loads(raw)
                kind = msg.get("type")
                if protocol == "voxtral":
                    if kind == "transcription.delta":
                        self.confirmed += msg.get("delta") or ""
                    elif kind == "transcription.done":
                        self.done.set()
                        return
                    elif kind == "error":
                        log.warning("[REMOTE-ERROR] %s", msg)
                elif protocol == "nemotron":
                    # The server holds the last word back until the next one starts;
                    # it arrives as `pending` and counts as the revisable tail.
                    if kind == "final":
                        self.confirmed = f"{self.confirmed} {msg.get('original') or ''}".strip()
                        self.tail = (msg.get("pending") or "").strip()
                    elif kind == "partial":
                        self.tail = (msg.get("text") or "").strip()
                    elif kind == "finish_done":
                        self.done.set()
                        return
                else:
                    if kind == "ready_to_stop":
                        self.done.set()
                        return
                    if "lines" in msg:
                        confirmed = _lines_text(msg.get("lines"))
                        if not confirmed.startswith(self.confirmed):
                            log.info("[REVISED] confirmed was=%r now=%r",
                                     self.confirmed, confirmed)
                        self.confirmed = confirmed
                        self.tail = (msg.get("buffer_transcription") or "").strip()
                self._offer_partial()
        except Exception as e:  # noqa: BLE001 - the session reports it at finish
            log.warning("[REMOTE-READ-FAILED] %s: %s", type(e).__name__, e)
        finally:
            self.done.set()

    # -- audio in -------------------------------------------------------------

    async def transcribe(self, audio: bytes) -> list:
        if self.ws is None:
            await self._open()
        self.fed_samples += len(audio) // 2
        if self.settings["protocol"] == "voxtral":
            await self.ws.send(json.dumps({"type": "input_audio_buffer.append",
                                           "audio": base64.b64encode(audio).decode()}))
        else:
            await self.ws.send(audio)
        if self.cfg.stity.commit.enable_dot_commit:
            self._commit_to_last_sentence_end()
        self._offer_partial()
        return self._drain()

    async def flush(self, reason: str, speech: Speech | None = None) -> list:
        self._commit_all(reason)
        return self._drain()

    async def finish(self, reason: str = "finish", speech: Speech | None = None) -> list:
        if self.ws is None:
            return self._drain()
        protocol = self.settings["protocol"]
        try:
            if protocol == "voxtral":
                await self.ws.send(json.dumps({"type": "input_audio_buffer.commit",
                                               "final": True}))
            elif protocol == "nemotron":
                await self.ws.send(json.dumps({"type": "finish"}))
            else:
                await self.ws.send(b"")
            await asyncio.wait_for(self.done.wait(), END_TIMEOUT_SEC)
        except asyncio.TimeoutError:
            log.warning("[REMOTE-END-TIMEOUT] %s gave no end-of-stream in %.0fs",
                        protocol, END_TIMEOUT_SEC)
        self._commit_all(reason, include_tail=True)
        await self._close_session()
        return self._drain()

    async def close(self) -> None:
        await self._close_session()

    async def _close_session(self) -> None:
        if self.reader is not None:
            self.reader.cancel()
            self.reader = None
        if self.ws is not None:
            try:
                if self.settings["protocol"] == "nemotron":
                    await self.ws.send(json.dumps({"type": "stop"}))
                await self.ws.close()
            except Exception:  # noqa: BLE001 - the item is already scored
                pass
            self.ws = None

    # -- commits ---------------------------------------------------------------

    def _words(self, include_tail: bool = True) -> list[str]:
        text = f"{self.confirmed} {self.tail}" if include_tail else self.confirmed
        return text.split()

    def _commit_to_last_sentence_end(self) -> None:
        confirmed = self._words(include_tail=False)
        # The last confirmed word may still be growing (Voxtral deltas split words),
        # so a dot on it counts only once the next word has started.
        settled = confirmed[:-1] if self.settings["protocol"] == "voxtral" else confirmed
        end = None
        for i in range(len(settled) - 1, self.committed_words - 1, -1):
            if SENTENCE_END.search(settled[i]):
                end = i + 1
                break
        if end is not None:
            self._commit(confirmed[self.committed_words:end], "dot")
            self.committed_words = end

    def _commit_all(self, reason: str, include_tail: bool = True) -> None:
        words = self._words(include_tail=include_tail)
        self._commit(words[self.committed_words:], reason)
        self.committed_words = max(self.committed_words, len(words))

    def _commit(self, words: list[str], reason: str) -> None:
        text = " ".join(words).strip()
        if not text:
            return
        self.out.append(Transcribed(
            original=text,
            language=self.language,
            commit_reason=reason,
            decision_audio_sec=round(self.fed_samples / audio_mod.SAMPLING_RATE, 3),
            committed_elapsed_sec=round(timing.elapsed() or 0.0, 4),
        ))
        self._offer_partial()

    def _offer_partial(self) -> None:
        text = " ".join(self._words()[self.committed_words:])
        if text == (self.partial_text or ""):
            return
        self.partial_text = text
        self.partial_seq += 1
        self.out.append(Partial(text=text, language=self.language, seq=self.partial_seq))

    def _drain(self) -> list:
        out, self.out = self.out, []
        return out
