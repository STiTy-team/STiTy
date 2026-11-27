"""Engines that stand in for vLLM when the zero transcriber runs without a GPU.

Both engines read the slot's position from the audio itself: every sample the scenarios
feed is its own time code (in 10 ms steps), so `window_of` tells which stretch of the
stream a decode was asked about.

- ScriptEngine plays a scripted talk. It hears a word only when the whole word is inside
  the slot, writes <SEG> once SEG_LAG_SEC of the pause after a sentence is inside too, and
  continues after the committed text it is given, as the real model does.
- ReplayEngine answers with what the real model said in a bench run, looked up by the
  slot position and the committed prefix. A call the recording does not have is a miss.
"""

from dataclasses import dataclass, field

import numpy as np

from core.components.transcription.zero_qwen3 import ASR_TEXT, SEG, ZeroQwen3Transcription
from core.utils import audio as audio_mod

SR = audio_mod.SAMPLING_RATE
STEP = SR // 100  # samples per 10 ms time code
SEG_LAG_SEC = 0.3
# The real model still hears a word whose first 150 ms fell before the slot.
ONSET_TOLERANCE_SEC = 0.15
# A slot shorter than this that holds only the start of a word makes the model guess it.
GUESS_MAX_SLOT_SEC = 0.5
LANGUAGE_NAMES = {"ko": "Korean", "en": "English"}
SILENT_HEADER = "language None" + ASR_TEXT
PROMPT = "<|im_start|>assistant\n"


@dataclass(frozen=True)
class Word:
    text: str
    start: float
    end: float
    ends_sentence: bool = False
    lang: str = "ko"


def sentence(
    first: int, count: int, start: float, word_sec=0.4, gap_sec=0.1, closed=True, lang="ko"
) -> list[Word]:
    """`count` words from `first` (단어<n>, or word<n> in English); the last closes the sentence if `closed`."""
    stem = "단어" if lang == "ko" else "word"
    words, t = [], start
    for i in range(count):
        last = closed and i == count - 1
        words.append(Word(f"{stem}{first + i}" + ("." if last else ""), t, t + word_sec, last, lang))
        t += word_sec + gap_sec
    return words


def time_coded_pcm(first: int, last: int) -> bytes:
    """Samples `first`..`last` of the stream, each holding its own time code."""
    return (np.arange(first, last) // STEP).astype(np.int16).tobytes()


def window_of(audio: np.ndarray) -> tuple[int, int]:
    """The samples of the stream a slot holds, read back from the time codes."""
    if audio.size == 0:
        return 0, 0
    first = int(round(audio[0] * 32768)) * STEP
    return first, first + audio.size


@dataclass
class Decode:
    window: tuple[int, int]  # samples
    committed: str
    output: str
    heard: list[str] = field(default_factory=list)
    missed: bool = False

    @property
    def seconds(self) -> tuple[float, float]:
        return self.window[0] / SR, self.window[1] / SR


@dataclass
class Output:
    text: str

    @property
    def outputs(self):
        return [self]


class Engine:
    def __init__(self) -> None:
        self.decodes: list[Decode] = []

    def answer(self, window: tuple[int, int], committed: str) -> Decode:
        raise NotImplementedError

    async def generate(self, request, params, request_id):
        committed = request["prompt"].removeprefix(PROMPT)
        decode = self.answer(window_of(request["multi_modal_data"]["audio"][0]), committed)
        self.decodes.append(decode)
        yield Output(decode.output)


class ScriptEngine(Engine):
    """`empty_at`: decode ends (seconds) where the model returns nothing, as it sometimes does
    right after a prefix that ends in <SEG>. `guess`: a short slot holding only a word's start
    makes the model write a guess (~<word>) instead of nothing."""

    def __init__(self, words: list[Word], *, empty_at=(), guess=False) -> None:
        super().__init__()
        self.words = words
        self.empty_at = {round(t * SR) for t in empty_at}
        self.guess = guess

    def answer(self, window, committed):
        start, end = window[0] / SR, window[1] / SR
        if window[1] in self.empty_at:
            return Decode(window, committed, "")
        heard = [w for w in self.words if w.start >= start - ONSET_TOLERANCE_SEC and w.end <= end]
        if not heard and self.guess and end - start < GUESS_MAX_SLOT_SEC:
            started = [w for w in self.words if start <= w.start < end < w.end]
            heard = [Word("~" + w.text.rstrip("."), w.start, w.end, False, w.lang) for w in started[:1]]
        full = SILENT_HEADER
        if heard:
            # Like the real model, the slot's language is named once, by its first words.
            body = ""
            for w in heard:
                body += (" " if body and not body.endswith(SEG) else "") + w.text
                if w.ends_sentence and end >= w.end + SEG_LAG_SEC:
                    body += SEG
            full = f"language {LANGUAGE_NAMES[heard[0].lang]}{ASR_TEXT}" + body
        # A prefix the talk does not start with would mean the scenario fell out of step.
        missed = not full.startswith(committed)
        output = "" if missed else full[len(committed):]
        return Decode(window, committed, output, [w.text for w in heard], missed)


class ReplayEngine(Engine):
    def __init__(self, recorded: list[dict]) -> None:
        super().__init__()
        self.recorded = {
            (r["slot_start"], r["slot_samples"], r["committed"]): r["output"] for r in recorded
        }

    def answer(self, window, committed):
        key = (window[0], window[1] - window[0], committed)
        output = self.recorded.get(key)
        return Decode(window, committed, output or "", missed=output is None)


def transcriber(engine: Engine, *, offset: int = 0) -> ZeroQwen3Transcription:
    asr = ZeroQwen3Transcription({"model_path": "fake"}, cfg=None)
    asr.engine, asr.prompt, asr.params = engine, PROMPT, None
    asr.start(languages=["ko", "en"])
    asr._heard = offset  # a replay cut starts mid-stream, right after a flush
    return asr


async def feed(asr, steps: list[tuple[int, bool]], *, start: int = 0, finish: bool = True) -> list:
    """Feeds the stream up to each `(end sample, flush after)` step, as bench does.

    A step that VAD closed is followed by flush("vad"); the end of the stream by finish(),
    or by a flush when `finish` is false (a cut that ends at a VAD flush).
    """
    committed, at = [], start
    for end, flush in steps:
        if end > at:
            committed += await asr.transcribe(time_coded_pcm(at, end))
            at = end
        if flush:
            committed += await asr.flush("vad")
    if finish:
        committed += await asr.finish()
    return committed


def even_steps(until: float, *, every: float = 0.1, vad_ends=()) -> list[tuple[int, bool]]:
    """Pieces of `every` seconds up to `until`, with a VAD flush at each of `vad_ends` and at the end."""
    stop, size = round(until * SR), round(every * SR)
    flushes = {round(t * SR) for t in vad_ends} | {stop}
    ends = sorted(set(range(size, stop, size)) | flushes)
    return [(end, end in flushes) for end in ends]
