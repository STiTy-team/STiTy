"""Transcription backends.

`qwen3` is the only one: init_model calls Qwen3ASRModel.LLM unconditionally, so
the transformers backend is unreachable and offering it in the config would be a
lie.

Per-model settings live under the component, not at the top level, because they
belong to the model: a checkpoint with a different chunk size or token budget
carries those values with it. Anything listed in OPTIONS is forwarded into
StreamingConfig; anything else raises. Silently dropping an unrecognised option is
the failure this guards against -- writing `max_new_tokens: 256` under the model
and having it do nothing is worse than an error, because the run still produces a
number.
"""
from .. import paths, registry
from ..errors import BenchConfigError

BASELINE = "Qwen/Qwen3-ASR-1.7B"

ALIASES = {
    ("baseline", None): BASELINE,
    ("baseline(1.0.0)", None): BASELINE,
    ("finetuned", "en"): "models/Qwen3-ASR-1.7B-en-silence-c80-merged",
    ("finetuned", "ko"): "models/Qwen3-ASR-1.7B-ko-silence-v4c900-merged",
    ("finetuned-seg", "en"): "models/Qwen3-ASR-1.7B-en-dailytalk-seg",
}

# option name -> (StreamingConfig field, caster)
OPTIONS = {
    "chunk_size_sec": ("chunk_size_sec", float),
    "max_new_tokens": ("max_new_tokens", int),
    "beam_size": ("beam_size", int),
    "enforce_eager": ("enforce_eager", bool),
    "gpu_memory_utilization": ("gpu_memory_utilization", float),
    "unfixed_chunk_num": ("unfixed_chunk_num", int),
    "unfixed_token_num": ("unfixed_token_num", int),
    "adapter_en": ("adapter_en", str),
    "adapter_ko": ("adapter_ko", str),
    "max_lora_rank": ("max_lora_rank", int),
    "no_lora": ("no_lora", bool),
}

RESERVED = {"model"}


def resolve_alias(value: str, src_lang: str) -> str:
    """A path or HF id passes through; a known alias resolves; anything else raises."""
    if "/" in value or value.startswith("models"):
        return value
    if (value, None) in ALIASES:
        return ALIASES[(value, None)]
    if (value, src_lang) in ALIASES:
        return ALIASES[(value, src_lang)]
    known = sorted({alias for alias, _ in ALIASES})
    langs_for = sorted({lang for alias, lang in ALIASES if alias == value and lang})
    if langs_for:
        raise BenchConfigError(
            f"transcription model alias {value!r} has no weights for language "
            f"{src_lang!r} (it exists for {langs_for}). Give the weights path "
            f"explicitly rather than letting it fall back to the base model."
        )
    raise BenchConfigError(
        f"unknown transcription model {value!r} (aliases: {known}; or give a path "
        f"or a HuggingFace id)"
    )


def config_contribution(name: str, options: dict, src_lang: str) -> dict:
    """StreamingConfig fields this model contributes. Raises on unknown options.

    Callable without instantiating anything, so config parsing (and --dry-run)
    validates per-model settings without importing torch.
    """
    registry.get("transcription", name)  # raises with the available list
    unknown = sorted(set(options) - RESERVED - set(OPTIONS))
    if unknown:
        raise BenchConfigError(
            f"stity.transcription: unknown option(s) {unknown} for {name!r} "
            f"(accepted: {sorted(RESERVED | set(OPTIONS))})"
        )
    out = {"model_path": paths.resolve_model_path(
        resolve_alias(options.get("model", BASELINE), src_lang))}
    for key, (field, cast) in OPTIONS.items():
        if key in options and options[key] is not None:
            out[field] = cast(options[key])
    return out


class TranscriptionBackend:
    def __init__(self, options: dict, ctx):
        self.options = options
        self.ctx = ctx


@registry.register("transcription", "qwen3")
class Qwen3Transcription(TranscriptionBackend):
    def __init__(self, options: dict, ctx):
        super().__init__(options, ctx)
        self.alias = options.get("model", BASELINE)
        self.kwargs = config_contribution("qwen3", options, ctx.src_lang)
        self.model_path = self.kwargs["model_path"]

    def streaming_config_kwargs(self) -> dict:
        return dict(self.kwargs)

    def provenance(self) -> dict:
        return {"backend": "qwen3", "alias": self.alias, "model_path": self.model_path,
                "options": {k: v for k, v in self.options.items() if k not in RESERVED}}


def build(name: str, options: dict, ctx) -> TranscriptionBackend:
    return registry.get("transcription", name)(options, ctx)
