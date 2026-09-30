"""A stand-in for Qwen3ASRModel's streaming API that replays a written decode script.

The commit logic in `qwen-seg`, `qwen-seg:v2` and `qwen-la` only reads what the model
wrote (`state.text`, `state.language`) and when (callbacks). A script says what each
decode returns as a function of the audio fed so far, so a worklog bug can be replayed
without a GPU, and v1 and v2 can be fed exactly the same decodes.
"""
from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np

from qwen_asr.inference.qwen3_asr import _cut_repeat_hallucination
from qwen_asr.inference.sentence_boundary import count_dot_commit_boundaries
from qwen_asr.inference.utils import parse_asr_output

from core.config import COMMIT_MODES

SR = 16000


@dataclass
class FakeState:
    chunk_size_sec: float
    chunk_size_samples: int
    unfixed_chunk_num: int = 2
    unfixed_token_num: int = 5
    chunk_id: int = 0
    buffer: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float32))
    audio_accum: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float32))
    context: str = ""
    language: str = ""
    text: str = ""
    _raw_decoded: str = ""
    allowed_languages: list | None = None
    hallucination_detected: bool = False
    committed_token_len: int = 0
    _last_nonempty_text: str = ""


class FakeTokenizer:
    def encode(self, text: str) -> list[int]:
        return list(range(len(text.split())))


class ScriptedModel:
    """`script(t, state, final)` returns the raw decode at audio time `t` (seconds)."""

    def __init__(self, script):
        self.script = script
        self.clock = 0
        self.processor = SimpleNamespace(tokenizer=FakeTokenizer())
        self.contexts: list[str] = []

    def init_streaming_state(self, context: str = "", chunk_size_sec: float = 2.0,
                             unfixed_chunk_num: int = 2, unfixed_token_num: int = 5,
                             allowed_languages=None, language=None) -> FakeState:
        self.contexts.append(context)
        return FakeState(chunk_size_sec=chunk_size_sec,
                         chunk_size_samples=int(round(chunk_size_sec * SR)),
                         unfixed_chunk_num=unfixed_chunk_num,
                         unfixed_token_num=unfixed_token_num,
                         context=context, allowed_languages=allowed_languages)

    async def streaming_transcribe(self, pcm, state, on_seg=None, on_dot=None,
                                   on_partial=None, lora_request=None):
        x = np.asarray(pcm, dtype=np.float32).reshape(-1)
        self.clock += x.shape[0]
        state.buffer = np.concatenate([state.buffer, x])
        while state.buffer.shape[0] >= state.chunk_size_samples:
            chunk = state.buffer[:state.chunk_size_samples]
            state.buffer = state.buffer[state.chunk_size_samples:]
            state.audio_accum = np.concatenate([state.audio_accum, chunk])
            at = (self.clock - state.buffer.shape[0]) / SR
            raw = self.script(round(at, 3), state, False)
            lang, text = parse_asr_output(raw)
            state.language = lang
            state.text = text
            cut = _cut_repeat_hallucination(text)
            if cut != text:
                state.text = cut
                state.chunk_id += 1
                state.hallucination_detected = True
                break
            if on_partial and "<asr_text>" in raw:
                await on_partial(state)
            if on_seg:
                for _ in range(text.count("<SEG>")):
                    await on_seg(state)
            if on_dot:
                for _ in range(count_dot_commit_boundaries(text)):
                    await on_dot(state)
            state._raw_decoded = raw
            if text.replace("<SEG>", "").strip():
                state._last_nonempty_text = text
            state.chunk_id += 1
        return state

    async def finish_streaming_transcribe(self, state, lora_request=None):
        if state.buffer is None or state.buffer.shape[0] == 0:
            return state
        tail = state.buffer
        state.buffer = np.zeros(0, np.float32)
        state.audio_accum = np.concatenate([state.audio_accum, tail])
        raw = self.script(round(self.clock / SR, 3), state, True)
        state._raw_decoded = raw
        state.language, state.text = parse_asr_output(raw)
        state.chunk_id += 1
        return state


def timeline(entries: list[tuple[float, str]], *, final: dict | None = None):
    """A script from `(from_time, raw)` pairs: the raw with the latest `from_time <= t`."""
    entries = sorted(entries)
    final = final or {}

    def script(t, state, is_final):
        if is_final:
            for at in sorted(final, reverse=True):
                if t >= at:
                    return final[at]
        chosen = ""
        for at, raw in entries:
            if t >= at:
                chosen = raw
        return chosen

    return script


def fake_cfg(mode: str = "seg", gpu: float = 0.3):
    table = COMMIT_MODES[mode]
    commit = SimpleNamespace(mode=mode, **table)
    return SimpleNamespace(stity=SimpleNamespace(commit=commit, gpu_memory_utilization=gpu,
                                                 resolved={"pipeline": {}}))


def make(cls, script, *, cfg=None, **options):
    settings = cls.validate({"model": "Org/fake", **options}, kind="transcription")
    part = cls(settings, cfg=cfg or fake_cfg())
    part.model = ScriptedModel(script)
    part.glossary = None
    return part


class FakeVad:
    def __init__(self, spans):
        self.spans = [list(s) for s in spans]


def chunks(seconds: float, *, step: float = 0.2, level: float = 0.01):
    n = int(round(step * SR))
    for _ in range(int(round(seconds / step))):
        yield (np.full(n, level, dtype=np.float32) * 32767).astype("<i2").tobytes()


async def feed(part, seconds: float, *, languages=("en",), vad=None, flush_at=None,
               speech=None) -> list:
    """Feed `seconds` of audio in 200 ms chunks; flush with `speech` after chunk `flush_at`."""
    part.start(languages=list(languages), vad=vad)
    out, heard = [], 0.0
    for chunk in chunks(seconds):
        heard = round(heard + 0.2, 3)
        out += await part.transcribe(chunk)
        if flush_at is not None and abs(heard - flush_at) < 1e-6:
            out += await part.flush("vad", speech)
    out += await part.finish()
    return out


def commits(records) -> list[str]:
    return [r.original for r in records if type(r).__name__ == "Transcribed"]


def partials(records) -> list[str]:
    return [r.text for r in records if type(r).__name__ == "Partial"]
