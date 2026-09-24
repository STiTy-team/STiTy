import time
from pathlib import Path

from core.errors import ConfigError
from core.utils import audio as audio_mod
from core.utils import langs
from core.utils import cache
from core.utils import logging
from core.utils import timing

from . import transcribers
from ..registry import Partial, Speech, Transcribed
from .base import Transcriber

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL = "Qwen/Qwen3-ASR-1.7B"
MAX_MODEL_LEN = 4096
KEEP_AFTER_SPEECH_SEC = 0.1
PARTIAL_MIN_INTERVAL_SEC = 0.12


def _last_word_boundary(decoded: str, already: str) -> int:
    shared = 0
    for shared, (left, right) in enumerate(zip(decoded, already)):
        if left != right:
            break
    else:
        shared = min(len(decoded), len(already))
    return decoded.rfind(" ", 0, shared) + 1


def resolve_model(raw: str) -> str:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        candidate = PROJECT_ROOT / raw
        if candidate.exists():
            return str(candidate)
    if path.exists():
        return str(path)
    if "/" not in raw:
        raise ConfigError(
            f"transcription model {raw!r} is neither a directory that exists nor a "
            f"HuggingFace id (those look like 'org/name'). "
            f"The default is {DEFAULT_MODEL!r}."
        )
    return raw


@transcribers.register('qwen3')
class Qwen3Transcription(Transcriber):
    SETTINGS = {
        "model": ("model_path", resolve_model),
        "chunk_size_sec": ("chunk_size_sec", float),
        "max_new_tokens": ("max_new_tokens", int),
        "beam_size": ("beam_size", int),
        "enforce_eager": ("enforce_eager", bool),
        "gpu_memory_utilization": ("gpu_memory_utilization", float),
        "unfixed_chunk_num": ("unfixed_chunk_num", int),
        "unfixed_token_num": ("unfixed_token_num", int),
        "restrict_languages": ("restrict_languages", bool),
    }

    model = None
    state = None

    @classmethod
    def validate(cls, options: dict, *, kind: str = "transcription") -> dict:
        out = super().validate(options, kind=kind)
        out.setdefault("model_path", DEFAULT_MODEL)
        out.setdefault("restrict_languages", True)
        return out

    async def load(self) -> None:
        self.model = await cache.load(self._cache_key(), self._build_model)
        self._log_gpu("after load")

    def _gpu_memory_utilization(self) -> float:
        return float(self.settings.get("gpu_memory_utilization")
                     or self.cfg.stity.gpu_memory_utilization)

    def _cache_key(self) -> tuple:
        kw = self.settings
        return ("qwen3-asr",
                kw["model_path"],
                self._gpu_memory_utilization(),
                int(kw.get("max_new_tokens", 128)),
                int(kw.get("beam_size", 1)),
                bool(kw.get("enforce_eager", False)))

    async def _build_model(self):
        from qwen_asr import Qwen3ASRModel
        from qwen_asr.inference.utils import warmup_streaming
        from vllm import SamplingParams

        kw = self.settings
        max_new_tokens = int(kw.get("max_new_tokens", 128))
        log.info("[LOAD] model %s", kw["model_path"])
        model = Qwen3ASRModel.LLM(
            model=kw["model_path"],
            gpu_memory_utilization=self._gpu_memory_utilization(),
            max_new_tokens=max_new_tokens,
            max_model_len=MAX_MODEL_LEN,
            enforce_eager=bool(kw.get("enforce_eager", False)),
        )
        beam_size = int(kw.get("beam_size", 1))
        params = dict(temperature=0.0, max_tokens=max_new_tokens,
                      skip_special_tokens=True)
        if beam_size > 1:
            try:
                model.sampling_params = SamplingParams(
                    use_beam_search=True, best_of=beam_size, **params)
            except TypeError:
                raise ConfigError(
                    f"beam_size={beam_size} but this vLLM build has no beam search. "
                    f"Use beam_size 1, or install a build that supports it -- "
                    f"falling back to greedy would score a decoder nobody chose."
                ) from None
        else:
            model.sampling_params = SamplingParams(**params)
        await warmup_streaming(model)
        return model

    def start(self, language: str | None = None, target_lang: str | None = None,
              **_) -> None:
        self._languages = (language, target_lang)
        self._fed_samples = 0
        self._partial_seq = 0
        self._out: list = []
        self._start_stream()

    def _drain(self) -> list:
        out, self._out = self._out, []
        return out

    def _allowed_language_names(self) -> list[str] | None:
        if not self.settings.get("restrict_languages", True):
            return None
        names = [langs.CODE_TO_NAME.get(langs.norm_code(code or ""))
                 for code in getattr(self, "_languages", ())]
        return list(dict.fromkeys(name for name in names if name)) or None

    def _start_stream(self) -> None:
        kw = self.settings
        self.state = self.model.init_streaming_state(
            chunk_size_sec=float(kw.get("chunk_size_sec", 2.0)),
            unfixed_chunk_num=int(kw.get("unfixed_chunk_num", 2)),
            unfixed_token_num=int(kw.get("unfixed_token_num", 5)),
            allowed_languages=self._allowed_language_names(),
        )
        self._emitted = ""
        self._partial_text = None
        self._partial_at = 0.0

    async def transcribe(self, audio: bytes) -> list:
        self._fed_samples += len(audio) // 2
        await self.model.streaming_transcribe(
            audio_mod.from_pcm_bytes(audio), self.state,
            on_seg=self._trigger("seg"), on_dot=self._trigger("dot"),
            on_partial=self._partial_callback(),
        )
        if self.cfg.stity.commit.always_commit:
            self._take("always")
        self._offer_partial(force=True)
        return self._drain()

    async def flush(self, reason: str, speech: Speech | None = None) -> list:
        out = await self.finish(reason, speech)
        self._start_stream()
        return out

    async def finish(self, reason: str = "finish",
                     speech: Speech | None = None) -> list:
        if speech is not None:
            self._drop_trailing_silence(speech)
        await self.model.finish_streaming_transcribe(self.state)
        self._take(reason)
        return self._drain()

    def _drop_trailing_silence(self, speech: Speech) -> None:
        cut = int(max(0.0, speech.silence_waited_out_sec - KEEP_AFTER_SPEECH_SEC)
                  * audio_mod.SAMPLING_RATE)
        if cut <= 0:
            return
        state = self.state
        buffer = getattr(state, "buffer", None)
        accum = getattr(state, "audio_accum", None)
        buffered = 0 if buffer is None else buffer.shape[0]
        accumulated = 0 if accum is None else accum.shape[0]
        would_leave_nothing_to_decode = buffered + accumulated <= cut
        if would_leave_nothing_to_decode:
            return
        from_buffer = min(cut, buffered)
        if from_buffer:
            state.buffer = buffer[: buffered - from_buffer]
        if cut > from_buffer:
            state.audio_accum = accum[: accumulated - (cut - from_buffer)]
        log.debug("[AUDIO-TRIM] cause=tail-silence removed_sec=%.2f",
                  cut / audio_mod.SAMPLING_RATE)

    def _trigger(self, reason: str):
        commit = self.cfg.stity.commit
        if reason == "seg" and (commit.hide_seg or commit.always_commit):
            return None
        if reason == "dot" and not commit.enable_dot_commit:
            return None

        async def callback(_state) -> None:
            self._take(reason)

        return callback

    def _partial_callback(self):
        async def callback(_state) -> None:
            self._offer_partial()

        return callback

    def _offer_partial(self, *, force: bool = False) -> None:
        now = time.perf_counter()
        if not force and now - self._partial_at < PARTIAL_MIN_INTERVAL_SEC:
            return
        decoded = (self.state.text or "").replace("<SEG>", "")
        tail = decoded[len(self._emitted):] if decoded.startswith(self._emitted) \
            else decoded
        text = tail.strip()
        if not text and not force:
            return
        previous = self._partial_text
        if not force and previous and text != previous and previous.startswith(text):
            return
        self._partial_at = now
        if text == (previous or ""):
            return
        self._partial_text = text
        self._partial_seq += 1
        self._out.append(Partial(
            text=text,
            language=langs.norm_code(self.state.language or ""),
            seq=self._partial_seq))

    def _take(self, reason: str) -> None:
        decoded = (self.state.text or "").replace("<SEG>", "")
        already = self._emitted
        self._emitted = decoded
        if decoded.startswith(already):
            new = decoded[len(already):].strip()
        elif already.startswith(decoded):
            log.info("[RETRACTED] was=%r", already[len(decoded):])
            new = ""
        else:
            resume_at = _last_word_boundary(decoded, already)
            log.info("[REVISED] was=%r now=%r",
                     already[resume_at:], decoded[resume_at:])
            new = decoded[resume_at:].strip()
        if not new:
            return
        self._out.append(Transcribed(
            original=new,
            language=langs.norm_code(self.state.language or ""),
            commit_reason=reason,
            decision_audio_sec=round(self._fed_samples / audio_mod.SAMPLING_RATE, 3),
            recv_elapsed_sec=round(timing.elapsed() or 0.0, 4),
        ))
        self._offer_partial(force=True)

    def _log_gpu(self, when: str) -> None:
        try:
            import torch

            if not torch.cuda.is_available():
                return
            free, total = torch.cuda.mem_get_info()
            free_gib, total_gib = round(free / 2**30, 2), round(total / 2**30, 2)
            log.info("[GPU] %s: %.2f GiB free of %.2f GiB",
                     when, free_gib, total_gib)
        except Exception:  # noqa: BLE001 - diagnostics only
            pass
