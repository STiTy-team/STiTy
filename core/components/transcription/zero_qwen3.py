import uuid

import numpy as np

from core.utils import audio as audio_mod
from core.utils import cache, langs, timing

from . import transcribers
from ..registry import Speech, Transcribed
from .base import Transcriber
from .qwen3 import MAX_MODEL_LEN, resolve_model

SEG = "<SEG>"
ASR_TEXT = "<asr_text>"
CHUNK_SAMPLES = 2 * audio_mod.SAMPLING_RATE


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
    SETTINGS = {"model": ("model_path", resolve_model)}

    async def load(self) -> None:
        path = self.settings["model_path"]
        gpu = self.cfg.stity.gpu_memory_utilization
        key = ("zero-qwen3", path, gpu)
        self.engine, self.prompt, self.params = await cache.load(key, lambda: load_model(path, gpu))

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
        before, seg, _ = self._decoded.rpartition(SEG)
        return self._commit(before + seg, "seg")

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

    async def _decode(self) -> None:
        # Re-decodes all audio since the last reset with the committed text as a fixed prefix,
        # so the model continues after it and never rewrites what was already emitted.
        request = {
            "prompt": self.prompt + self._committed,
            "multi_modal_data": {"audio": [self._audio]},
        }
        async for out in self.engine.generate(request, self.params, request_id=str(uuid.uuid4())):
            pass
        self._decoded = self._committed + out.outputs[0].text
        self._undecoded = 0

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
