import asyncio
import hashlib
import re
import unicodedata
import uuid

import numpy as np

from core.utils import audio as audio_mod
from core.utils import cache, langs, stream, timing

from . import transcribers
from ..registry import Speech, Transcribed
from .base import Transcriber
from .qwen3 import MAX_MODEL_LEN, resolve_model

SEG = "<SEG>"
ASR_TEXT = "<asr_text>"
CHUNK_SAMPLES = 2 * audio_mod.SAMPLING_RATE

# seg_confirm 에서 SEG 뒤에 이만큼 단어가 붙어 나오면 그 SEG 를 확정된 것으로 본다.
# 끝 SEG 를 들고 있다가 다음 디코딩에서 말이 붙은 경우도 같은 기준으로 확정된다.
SEG_MIN_TAIL_WORDS = 1

# 슬롯 오디오 길이가 이 길이를 넘을 경우에는 강제로 커밋하고 비우기
MAX_SLOT_SAMPLES = 30 * audio_mod.SAMPLING_RATE # Whisper 창 길이 30초 기준


async def load_model(path: str, gpu_memory_utilization: float) -> tuple:
    from qwen_asr.core.transformers_backend import Qwen3ASRProcessor
    from qwen_asr.core.vllm_backend import Qwen3ASRForConditionalGeneration
    from vllm import ModelRegistry, SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine

    ModelRegistry.register_model("Qwen3ASRForConditionalGeneration", Qwen3ASRForConditionalGeneration)
    engine = AsyncLLMEngine.from_engine_args(
        AsyncEngineArgs(
            model=path, gpu_memory_utilization=gpu_memory_utilization, max_model_len=MAX_MODEL_LEN
        )
    )
    processor = Qwen3ASRProcessor.from_pretrained(path, fix_mistral_regex=True)
    prompt = processor.apply_chat_template(
        [
            {"role": "system", "content": ""},
            {"role": "user", "content": [{"type": "audio", "audio": ""}]},
        ],
        add_generation_prompt=True,
        tokenize=False,
    )
    return engine, prompt, SamplingParams(temperature=0.0, max_tokens=128)


@transcribers.register("zero-qwen3")
class ZeroQwen3Transcription(Transcriber):
    SETTINGS = {
        "model": ("model_path", resolve_model),
        "record_decodes": ("record_decodes", bool),
        "seg_confirm": ("seg_confirm", bool),
        "seg_min_tail_words": ("seg_min_tail_words", int),
        "aligner": ("aligner", str),
    }

    async def load(self) -> None:
        path = self.settings["model_path"]
        gpu = self.cfg.stity.gpu_memory_utilization
        key = ("zero-qwen3", path, gpu)
        self.engine, self.prompt, self.params = await cache.load(key, lambda: load_model(path, gpu))
        self.aligner = None
        if self.settings.get("aligner"):
            name = self.settings["aligner"]
            self.aligner = await cache.load(("zero-aligner", name), lambda: load_aligner(name))

    def start(self, languages: list[str] | None = None, **_) -> None:
        self._heard = 0
        self._language = ""
        self._reset()

    async def transcribe(self, audio: bytes) -> list:
        samples = audio_mod.from_pcm_bytes(audio)
        self._heard += len(samples)
        self._audio = np.concatenate([self._audio, samples])
        self._undecoded += len(samples)
        if self._undecoded < CHUNK_SAMPLES:
            return []

        await self._decode()

        # SEG 가 안 끝나 슬롯이 30초를 넘을 경우에는 전부 커밋하고 슬롯 비우기
        if len(self._audio) > MAX_SLOT_SAMPLES:
            return await self.flush("limit")

        if self.settings.get("seg_confirm"):
            return await self._commit_confirmed()

        before, seg, tail = self._decoded.rpartition(SEG)
        out = self._commit(before + seg, "seg")

        # SEG 로 딱 끝났다 -> 오디오/커밋 텍스트를 버리고 새 슬롯 시작하기
        # TODO: 여기에서 문제 발생하는거일수도??
        if seg and not tail.strip():
            self._reset()

        return out

    async def _commit_confirmed(self) -> list:
        """Commits a <SEG> only once the model has stood by it.

        - A <SEG> followed by at least seg_min_tail_words words was written after hearing
          what comes next: commit through it now. With an aligner the slot restarts there,
          keeping the audio from the middle of the pause at that <SEG>; without one the slot
          goes on.
        - A <SEG> at the very end may be a breath: hold it. If the next decode still ends
          there with nothing new, commit and reset; if words follow, the rule above takes it
          (with the default of one word, as soon as anything follows).
        - A decode that adds nothing tells nothing and changes nothing.
        VAD end and the 30 s limit still flush everything.
        """
        pending = self._decoded[len(self._committed) :]
        if not pending.strip():
            return []
        need = self.settings.get("seg_min_tail_words", SEG_MIN_TAIL_WORDS)
        cut = None
        for found in re.finditer(re.escape(SEG), pending):
            if len(_words(pending[found.end() :])) >= need:
                cut = found.end()
        out = []
        if cut is not None:
            through = self._committed + pending[:cut]
            at = await self._pause_at(through) if self.aligner is not None else None
            out = self._commit(through, "seg")
            pending = pending[cut:]
            if at is not None:
                carry = self._audio[at:]
                self._reset()
                self._audio, self._undecoded = carry, len(carry)
                return out
        if not pending.rstrip().endswith(SEG):
            self._held = None
            return out
        if self._held is not None and _words(self._held) == _words(pending):
            out += self._commit(self._committed + pending, "seg")
            self._reset()
            return out
        self._held = pending
        return out

    async def _pause_at(self, through: str) -> int | None:
        """The sample in the slot halfway between the last word of `through` and the next one.

        The whole decoded slot is aligned so the next word's start is known; aligned units
        are mapped back by counting the characters the aligner keeps.
        """
        header, _, body = self._decoded.rpartition(ASR_TEXT)
        language = "Korean" if "Korean" in header or not header else header.split()[-1]
        text = " ".join(body.replace(SEG, " ").split())
        before = _kept_chars(through.rpartition(ASR_TEXT)[2].replace(SEG, " "))
        if not text or not before:
            return None
        audio = self._audio
        result = await asyncio.to_thread(self.aligner.align, (audio, audio_mod.SAMPLING_RATE), text, language)
        units, done = list(result[0]), 0
        for i, unit in enumerate(units):
            done += _kept_chars(unit.text)
            if done == before:
                if i + 1 >= len(units):
                    return None
                at = (unit.end_time + units[i + 1].start_time) / 2
                sample = int(at * audio_mod.SAMPLING_RATE)
                if self.settings.get("record_decodes"):
                    stream.record("align", slot_start=self._heard - len(audio), cut=sample,
                                  last=unit.text, next=units[i + 1].text,
                                  last_end=unit.end_time, next_start=units[i + 1].start_time)
                return sample if 0 < sample < len(audio) else None
            if done > before:
                return None
        return None

    async def flush(self, reason: str, speech: Speech | None = None) -> list:
        out = await self.finish(reason, speech)
        self._reset()
        return out

    async def finish(self, reason: str = "finish", speech: Speech | None = None) -> list:
        if self._undecoded:
            await self._decode()
        return self._commit(self._decoded, reason)

    def _reset(self) -> None:
        self._audio = np.zeros(0, dtype=np.float32)
        self._undecoded = 0
        self._decoded = ""
        self._committed = ""
        self._held = None

    async def _decode(self) -> None:
        # Re-decodes all audio since the last reset with the committed text as a fixed prefix,
        # so the model continues after it and never rewrites what was already emitted.
        request = {
            "prompt": self.prompt + self._committed,
            "multi_modal_data": {"audio": [self._audio]},
        }
        async for out in self.engine.generate(request, self.params, request_id=str(uuid.uuid4())):
            pass
        text = out.outputs[0].text
        if self.settings.get("record_decodes"):
            self._record_decode(text)
        self._decoded = self._committed + text
        self._undecoded = 0

    def _record_decode(self, text: str) -> None:
        # One "decode" event per model call: what the slot held, the prefix it was given and
        # the raw reply (headers and <SEG> included), so a run can be replayed without a GPU.
        stream.record(
            "decode",
            slot_start=self._heard - len(self._audio),
            slot_samples=len(self._audio),
            audio_sha1=hashlib.sha1(self._audio.tobytes()).hexdigest()[:16],
            committed=self._committed,
            output=text,
        )

    def _commit(self, through: str, reason: str) -> list:
        piece = through[len(self._committed) :]
        self._committed = through

        # Only the first piece of a stream carries the "language X<asr_text>" header.
        header, tag, text = piece.rpartition(ASR_TEXT)
        if tag:
            self._language = langs.norm_code(header.removeprefix("language ")) or self._language

        text = " ".join(text.replace(SEG, " ").split())
        if not text:
            return []
        return [
            Transcribed(
                original=text,
                language=self._language,
                commit_reason=reason,
                decision_audio_sec=round(self._heard / audio_mod.SAMPLING_RATE, 3),
                committed_elapsed_sec=round(timing.elapsed() or 0.0, 4),
            )
        ]


def _kept_chars(text: str) -> int:
    """How many characters of `text` the forced aligner keeps (letters, digits, apostrophes)."""
    return sum(1 for ch in text if ch == "'" or unicodedata.category(ch)[0] in "LN")


async def load_aligner(name: str):
    import torch
    from qwen_asr import Qwen3ForcedAligner

    return Qwen3ForcedAligner.from_pretrained(name, dtype=torch.bfloat16, device_map="cuda")


def _words(text: str) -> list[str]:
    """The words of a piece of decoded text, without language headers, <SEG> and punctuation."""
    text = re.sub(r"language \S*" + re.escape(ASR_TEXT), " ", text).replace(SEG, " ")
    return [w for w in (re.sub(r"[^\w]", "", t.lower()) for t in text.split()) if w]
