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

    def prompt(context: str = "") -> str:
        # Qwen3-ASR reads earlier text, if any, from the system message.
        return processor.apply_chat_template(
            [
                {"role": "system", "content": context},
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
        "context_sentences": ("context_sentences", int),
        "carry_sentence": ("carry_sentence", bool),
        "force_allowed_language": ("force_allowed_language", bool),
        "lid_timeline": ("lid_timeline", dict),
        "lid_switch_steps": ("lid_switch_steps", int),
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
        self.lid = None
        if self.settings.get("lid_timeline") is not None:
            from core.components.langid.audio.component import AudioLanguageDetector

            # Only the timeline is used: it is fed every piece and read at each decode.
            options = {"mode": "timeline", "aligner": False, **self.settings["lid_timeline"]}
            self.lid = AudioLanguageDetector(options, cfg=self.cfg)
            await self.lid.load()

    def start(self, languages: list[str] | None = None, **_) -> None:
        self._allowed = [code for code in (langs.norm_code(x) for x in languages or []) if code]
        if getattr(self, "lid", None) is not None:
            self.lid.start(languages=self._allowed)
        self._heard = 0
        self._language = ""
        self._recent = []
        self._reset()

    async def transcribe(self, audio: bytes) -> list:
        samples = audio_mod.from_pcm_bytes(audio)
        if getattr(self, "lid", None) is not None:
            self.lid.hear(audio)
        self._heard += len(samples)
        self._audio = np.concatenate([self._audio, samples])
        self._undecoded += len(samples)
        if self._undecoded < CHUNK_SAMPLES:
            return []

        await self._decode()

        # SEG 가 안 끝나 슬롯이 30초를 넘을 경우에는 전부 커밋하고 슬롯 비우기
        if len(self._audio) > MAX_SLOT_SAMPLES:
            return await self.flush("limit")

        if getattr(self, "lid", None) is not None and self.aligner is not None:
            switched = await self._commit_at_switch()
            if switched is not None:
                return switched

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
            found = await self._pause_at(through) if self.aligner is not None else None
            header = re.match(r"language \S*" + re.escape(ASR_TEXT), self._decoded)
            out = self._commit(through, "seg")
            pending = pending[cut:]
            if found is not None:
                at, sentence_at = found
                last = [p for p in through.split(SEG) if p.strip()][-1].rpartition(ASR_TEXT)[2]
                if self.settings.get("carry_sentence") and sentence_at is not None and header:
                    # Keep the sentence just committed, as audio and as already-written text,
                    # so the new slot starts with both kinds of context and does not write it again.
                    carry = self._audio[sentence_at:]
                    self._reset()
                    self._committed = self._decoded = f"{header.group()}{' '.join(last.split())} {SEG}"
                else:
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

    def _switch_at(self) -> tuple[float, str, str] | None:
        """Where, inside the slot, the audio language timeline says the language changes: the
        time (seconds into the slot) halfway between the last step of the old language and the
        first of the new, when lid_switch_steps steps in a row are in the new one."""
        from core.components.langid.audio.decide import smooth

        lid, need = self.lid, self.settings.get("lid_switch_steps", 2)
        start = (self._heard - len(self._audio)) / audio_mod.SAMPLING_RATE
        steps = [(t, sc, q) for (t, sc), q in zip(lid.steps, lid.silent) if t >= start]
        if len(lid.candidates) < 2 or len(steps) < need + 1:
            return None
        posts = smooth([sc for _, sc, _ in steps], lid.candidates, lid.settings.get("p_stay", 0.8))
        conf = lid.settings.get("conf", 0.5)
        labels = [
            None if q or sc is None or max(p.values()) < conf else max(p, key=p.get)
            for (_, sc, q), p in zip(steps, posts)
        ]
        known = [(t, lab) for (t, _, _), lab in zip(steps, labels) if lab]
        if not known:
            return None
        old = known[0][1]
        for i in range(1, len(known) - need + 1):
            if all(lab != old and lab == known[i][1] for _, lab in known[i : i + need]):
                return (known[i - 1][0] + known[i][0]) / 2 - start, old, known[i][1]
        return None

    async def _commit_at_switch(self) -> list | None:
        """With lid_timeline: when the language changes inside the slot, commit what was said
        before the change and restart the slot there.

        The text comes from the latest decode written in the old language - the decode that
        heard the new language often rewrites the slot in it and drops the old sentence. That
        text is aligned to the audio it was decoded from; the words ending before the change are
        committed (reason "lang") and the slot restarts halfway to the next word.
        """
        found = self._switch_at()
        if found is None:
            return None
        at, old, new = found
        for decoded, length in ((self._decoded, len(self._audio)), self._prev):
            named = re.match(r"language (\S+?)" + re.escape(ASR_TEXT), decoded or "")
            if named and langs.norm_code(named.group(1)) == old and decoded.startswith(self._committed):
                break
        else:
            return None
        header, body = named.group(), decoded[named.end() :]
        text = " ".join(body.replace(SEG, " ").split())
        if not text:
            return None
        audio = self._audio[:length]
        result = await asyncio.to_thread(
            self.aligner.align, (audio, audio_mod.SAMPLING_RATE), text, named.group(1)
        )
        units = list(result[0])
        kept = [u for u in units if u.end_time <= at + 0.15]
        if not kept:
            return None
        k = len(kept) - 1
        cut = (units[k].end_time + units[k + 1].start_time) / 2 if k + 1 < len(units) else at
        through = header + _prefix_kept(body, sum(_kept_chars(u.text) for u in kept))
        if len(through) <= len(self._committed):
            return None
        sample = int(min(cut, at + 0.3) * audio_mod.SAMPLING_RATE)
        if self.settings.get("record_decodes"):
            stream.record("lang_switch", slot_start=self._heard - len(self._audio), at=at, old=old,
                          new=new, cut=sample, last=units[k].text)
        out = self._commit(through, "lang")
        carry = self._audio[sample:]
        self._reset()
        self._audio, self._undecoded = carry, len(carry)
        return out

    async def _pause_at(self, through: str) -> tuple[int, int | None] | None:
        """The sample in the slot halfway between the last word of `through` and the next one,
        and where the last sentence of `through` starts (halfway from the word before it).

        The whole decoded slot is aligned so the next word's start is known; aligned units
        are mapped back by counting the characters the aligner keeps.
        """
        header, _, body = self._decoded.rpartition(ASR_TEXT)
        language = "Korean" if "Korean" in header or not header else header.split()[-1]
        text = " ".join(body.replace(SEG, " ").split())
        before = _kept_chars(through.rpartition(ASR_TEXT)[2].replace(SEG, " "))
        if not text or not before:
            return None
        pieces = [p for p in through.split(SEG) if p.strip()]
        lead = before - _kept_chars(pieces[-1].rpartition(ASR_TEXT)[2]) if pieces else None
        audio = self._audio
        result = await asyncio.to_thread(self.aligner.align, (audio, audio_mod.SAMPLING_RATE), text, language)
        units, done, sentence_at = list(result[0]), 0, None
        for i, unit in enumerate(units):
            if done == lead:
                start = (units[i - 1].end_time + unit.start_time) / 2 if i else 0.0
                sentence_at = int(start * audio_mod.SAMPLING_RATE)
            done += _kept_chars(unit.text)
            if done == before:
                if i + 1 >= len(units):
                    return None
                at = (unit.end_time + units[i + 1].start_time) / 2
                sample = int(at * audio_mod.SAMPLING_RATE)
                if self.settings.get("record_decodes"):
                    stream.record("align", slot_start=self._heard - len(audio), cut=sample,
                                  sentence_start=sentence_at, last=unit.text, next=units[i + 1].text,
                                  last_end=unit.end_time, next_start=units[i + 1].start_time)
                return (sample, sentence_at) if 0 < sample < len(audio) else None
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
        self._decoded_len = 0
        self._prev = ("", 0)
        self._committed = ""
        self._held = None
        # A new slot starts with the last committed sentences as context (context_sentences).
        keep = self.settings.get("context_sentences", 0)
        self._context = " ".join(self._recent[-keep:]) if keep else ""

    async def _decode(self) -> None:
        # Re-decodes all audio since the last reset with the committed text as a fixed prefix,
        # so the model continues after it and never rewrites what was already emitted.
        request = {
            "prompt": self.prompt(self._context) + self._committed,
            "multi_modal_data": {"audio": [self._audio]},
        }
        async for out in self.engine.generate(request, self.params, request_id=str(uuid.uuid4())):
            pass
        text = out.outputs[0].text
        forced = self._allowed_header(text)
        if forced:
            # The model named a language outside the session's (laughter as Chinese, "음" as
            # "嗯"): decode again with the header fixed to the last committed language.
            request["prompt"] += forced
            async for out in self.engine.generate(request, self.params, request_id=str(uuid.uuid4())):
                pass
            text = forced + out.outputs[0].text
        if self.settings.get("record_decodes"):
            self._record_decode(text)
        self._prev = (self._decoded, self._decoded_len)
        self._decoded = self._committed + text
        self._decoded_len = len(self._audio)
        self._undecoded = 0

    def _allowed_header(self, text: str) -> str:
        """With force_allowed_language, the header to force when a slot's first decode names a
        language the session does not have; "" otherwise."""
        if not self.settings.get("force_allowed_language") or self._committed or not self._allowed:
            return ""
        named = re.match(r"language (\S+?)" + re.escape(ASR_TEXT), text)
        code = langs.norm_code(named.group(1)) if named else ""
        if not named or named.group(1) == "None" or code in self._allowed:
            return ""
        keep = self._language if self._language in self._allowed else self._allowed[0]
        return f"language {langs.CODE_TO_NAME[keep]}{ASR_TEXT}"

    def _record_decode(self, text: str) -> None:
        # One "decode" event per model call: what the slot held, the prefix it was given and
        # the raw reply (headers and <SEG> included), so a run can be replayed without a GPU.
        stream.record(
            "decode",
            slot_start=self._heard - len(self._audio),
            slot_samples=len(self._audio),
            audio_sha1=hashlib.sha1(self._audio.tobytes()).hexdigest()[:16],
            committed=self._committed,
            context=self._context,
            output=text,
        )

    def _commit(self, through: str, reason: str) -> list:
        piece = through[len(self._committed) :]
        self._committed = through

        # Only the first piece of a stream carries the "language X<asr_text>" header.
        header, tag, text = piece.rpartition(ASR_TEXT)
        if tag:
            self._language = langs.norm_code(header.removeprefix("language ")) or self._language

        # Each <SEG>-closed piece counts as one sentence for the next slot's context.
        self._recent += [" ".join(p.split()) for p in text.split(SEG) if p.strip()]
        del self._recent[:-8]
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


def _prefix_kept(body: str, n: int) -> str:
    """The start of decoded `body` that holds its first `n` aligner-kept characters, with the
    punctuation right after them; <SEG> marks are not counted."""
    count, i = 0, 0
    while i < len(body):
        if body.startswith(SEG, i):
            i += len(SEG)
            continue
        ch = body[i]
        i += 1
        if ch == "'" or unicodedata.category(ch)[0] in "LN":
            count += 1
            if count == n:
                while i < len(body) and not body.startswith(SEG, i) and not body[i].isspace() \
                        and not (body[i] == "'" or unicodedata.category(body[i])[0] in "LN"):
                    i += 1
                return body[:i]
    return body


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
