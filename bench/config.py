import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from . import langs
from .errors import BenchConfigError

COMMIT_MODES = {
    "seg": dict(always_commit=False, enable_dot_commit=False, dot_commit_confirm=False,
                hide_seg=False),
    "punct": dict(always_commit=False, enable_dot_commit=True, dot_commit_confirm=True,
                  hide_seg=True),
    "static": dict(always_commit=True, enable_dot_commit=False, dot_commit_confirm=False,
                   hide_seg=True),
}

KNOWN_METRICS = ("wer", "cer", "fsl", "laal", "bleu", "commit", "routing")

LATENCY_METRICS = frozenset({"fsl", "laal"})

LOCAL_TRANSLATION_GPU_CEILING = 0.6


def _as_component(raw: Any, *, field_name: str) -> tuple[str, dict]:
    if isinstance(raw, str):
        return raw, {}
    if isinstance(raw, dict):
        spec = dict(raw)
        name = spec.pop("name", None)
        if not isinstance(name, str) or not name:
            raise BenchConfigError(f"{field_name}: mapping form needs a 'name' key")
        return name, spec
    raise BenchConfigError(f"{field_name}: expected a name or a mapping, got {type(raw).__name__}")


def _require(d: dict, key: str, *, where: str):
    if key not in d or d[key] is None:
        raise BenchConfigError(f"{where}: {key!r} is required")
    return d[key]


def _reject_unknown(d: dict, allowed: set[str], *, where: str) -> None:
    extra = sorted(set(d) - allowed)
    if extra:
        raise BenchConfigError(f"{where}: unknown key(s) {extra} (allowed: {sorted(allowed)})")


@dataclass(frozen=True)
class DatasetCfg:
    name: str
    limit: int | None = None

    @staticmethod
    def parse(raw: Any) -> "DatasetCfg":
        name, spec = _as_component(raw, field_name="dataset")
        _reject_unknown(spec, {"limit"}, where="dataset")
        limit = spec.get("limit")
        if limit is not None:
            if not isinstance(limit, int) or limit <= 0:
                raise BenchConfigError(f"dataset.limit must be a positive integer, got {limit!r}")
        return DatasetCfg(name=name, limit=limit)


@dataclass(frozen=True)
class LanguagesCfg:
    mode: str
    map: dict[str, str]
    lang: str
    target: str
    restrict: bool

    @staticmethod
    def parse(raw: Any) -> "LanguagesCfg":
        if not isinstance(raw, dict):
            raise BenchConfigError("languages: expected a mapping")
        _reject_unknown(raw, {"map", "lang", "target", "restrict"}, where="languages")
        restrict = bool(raw.get("restrict", True))
        has_map = raw.get("map") is not None
        has_pair = raw.get("lang") is not None or raw.get("target") is not None
        if has_map and has_pair:
            raise BenchConfigError(
                "languages: 'map' and the 'lang'/'target' pair are mutually exclusive. "
                "They produce different ASR allowed_languages sets, so their scores are "
                "not comparable. Pick one."
            )
        if has_map:
            pairs = raw["map"]
            if not isinstance(pairs, dict) or not pairs:
                raise BenchConfigError("languages.map: expected a non-empty mapping")
            resolved: dict[str, str] = {}
            for src, dst in pairs.items():
                s = langs.require_code(src, field="languages.map key")
                d = langs.require_code(dst, field=f"languages.map[{src}]")
                if s == d:
                    raise BenchConfigError(
                        f"languages.map[{src}]={dst}: identity entries are dropped by the "
                        "server (parse_lang_map), so this would silently do nothing"
                    )
                resolved[s] = d
            return LanguagesCfg("map", resolved, "", "", restrict)
        if not has_pair:
            raise BenchConfigError(
                "languages: give either 'map' (multi-language) or 'lang' + 'target' (two-language)"
            )
        lang = langs.require_code(_require(raw, "lang", where="languages"), field="languages.lang")
        target = langs.require_code(_require(raw, "target", where="languages"),
                                    field="languages.target")
        if lang == target:
            raise BenchConfigError("languages: 'lang' and 'target' must differ")
        return LanguagesCfg("pair", {}, lang, target, restrict)

    @property
    def source_langs(self) -> list[str]:
        if self.mode == "map":
            return sorted(self.map)
        return [self.lang, self.target]

    @property
    def target_langs(self) -> list[str]:
        if self.mode == "map":
            return sorted(set(self.map.values()))
        return [self.target, self.lang]

    def expected_target(self, src_lang: str) -> str:
        """The target the server should pick for an utterance in src_lang.

        Mirrors _correct_and_translate: langMap wins, else the pair rule.
        """
        code = langs.norm_code(src_lang)
        if self.mode == "map":
            return self.map.get(code, "")
        if not code:
            return self.target
        if code == self.lang:
            return self.target
        return self.lang

    def covers(self, dataset_langs: list[str]) -> list[str]:
        known = set(self.source_langs)
        return [c for c in dataset_langs if c not in known]

    def targets_for(self, dataset_langs: list[str]) -> list[str]:
        """Target languages this run can actually produce, given what is in the data.

        Not the same as `target_langs`, which lists both ends of a pair because a
        two-way conversation can travel in either direction. A dataset whose audio
        is all English never triggers the ko->en leg, so demanding an English
        reference for an en->ko run would block a perfectly scorable setup.
        """
        out = []
        for code in dataset_langs:
            target = self.expected_target(code)
            if target and target not in out:
                out.append(target)
        return sorted(out)


@dataclass(frozen=True)
class CommitCfg:
    mode: str
    always_commit: bool
    enable_dot_commit: bool
    dot_commit_confirm: bool
    dot_commit_stall_chunks: int
    hide_seg: bool
    rep_dedup: bool

    @staticmethod
    def parse(raw: Any) -> "CommitCfg":
        mode, spec = _as_component(raw, field_name="stity.commit")
        if mode not in COMMIT_MODES:
            raise BenchConfigError(
                f"stity.commit: unknown mode {mode!r} (available: {sorted(COMMIT_MODES)})"
            )
        _reject_unknown(spec, {"stall_chunks", "hide_seg", "rep_dedup"}, where="stity.commit")
        table = COMMIT_MODES[mode]
        return CommitCfg(
            mode=mode,
            always_commit=table["always_commit"],
            enable_dot_commit=table["enable_dot_commit"],
            dot_commit_confirm=table["dot_commit_confirm"],
            dot_commit_stall_chunks=int(spec.get("stall_chunks", 1)),
            hide_seg=bool(spec.get("hide_seg", table["hide_seg"])),
            rep_dedup=bool(spec.get("rep_dedup", True)),
        )


@dataclass(frozen=True)
class VadCfg:
    enabled: bool
    min_silence_ms: int
    threshold: float
    speech_pad_ms: int

    @staticmethod
    def parse(raw: Any) -> "VadCfg":
        if raw is None:
            raw = {}
        if isinstance(raw, bool):
            raw = {"enabled": raw}
        if isinstance(raw, str):
            if raw not in ("silero", "none"):
                raise BenchConfigError(
                    f"stity.vad: expected 'silero', 'none', or a mapping, got {raw!r}"
                )
            raw = {"enabled": raw == "silero"}
        if not isinstance(raw, dict):
            raise BenchConfigError("stity.vad: expected a mapping")
        _reject_unknown(raw, {"enabled", "min_silence_ms", "threshold", "speech_pad_ms"},
                        where="stity.vad")
        return VadCfg(
            enabled=bool(raw.get("enabled", True)),
            min_silence_ms=int(raw.get("min_silence_ms", 800)),
            threshold=float(raw.get("threshold", 0.5)),
            speech_pad_ms=int(raw.get("speech_pad_ms", 160)),
        )


@dataclass(frozen=True)
class ComponentCfg:
    name: str
    options: dict = field(default_factory=dict)

    def fingerprint(self) -> str:
        if not self.options:
            return self.name
        inner = ":".join(f"{k}={self.options[k]}" for k in sorted(self.options))
        return f"{self.name}:{inner}"


@dataclass(frozen=True)
class StityCfg:
    """Per-model settings live under their component, not here.

    A checkpoint's chunk size or token budget belongs to that checkpoint, so the
    master config passes them through to the model that owns them rather than
    flattening everything into one namespace. gpu_memory_utilization stays at this
    level because it is a whole-process resource decision that the local
    translator also has to fit inside -- a model may still override it.
    """
    transcription: ComponentCfg
    translation: ComponentCfg
    commit: CommitCfg
    vad: VadCfg
    gpu_memory_utilization: float
    transcription_kwargs: dict
    translation_kwargs: dict

    @staticmethod
    def parse(raw: Any, *, src_lang: str) -> "StityCfg":
        from .components import transcription as transcription_mod
        from .components import translation as translation_mod

        if not isinstance(raw, dict):
            raise BenchConfigError("stity: expected a mapping")
        _reject_unknown(raw, {"transcription", "translation", "commit", "vad",
                              "gpu_memory_utilization"}, where="stity")
        t_name, t_opts = _as_component(_require(raw, "transcription", where="stity"),
                                       field_name="stity.transcription")
        x_name, x_opts = _as_component(_require(raw, "translation", where="stity"),
                                       field_name="stity.translation")

        gpu = raw.get("gpu_memory_utilization",
                      t_opts.get("gpu_memory_utilization"))
        if gpu is None:
            raise BenchConfigError(
                "stity.gpu_memory_utilization is required and has no default. vLLM's own "
                "default (0.8) means 'reserve everything spare', which has killed a "
                "co-tenant job on this machine. Use 0.5 when sharing the card."
            )
        gpu = float(gpu)
        if not 0.0 < gpu <= 0.95:
            raise BenchConfigError(
                f"stity.gpu_memory_utilization must be in (0, 0.95], got {gpu}")
        if x_name == "local" and gpu > LOCAL_TRANSLATION_GPU_CEILING:
            raise BenchConfigError(
                f"stity.gpu_memory_utilization={gpu} with translation 'local': the local "
                f"translator adds about 7.2 GiB in the same process, so keep this at or "
                f"below {LOCAL_TRANSLATION_GPU_CEILING}, or run the translator as a "
                f"separate process and use translation 'remote'."
            )

        # Validated here so a mistyped per-model option fails at --dry-run rather
        # than being dropped and producing a plausible number.
        t_kwargs = transcription_mod.config_contribution(t_name, t_opts, src_lang)
        x_kwargs = translation_mod.config_contribution(x_name, x_opts)
        t_kwargs.setdefault("gpu_memory_utilization", gpu)

        return StityCfg(
            transcription=ComponentCfg(t_name, t_opts),
            translation=ComponentCfg(x_name, x_opts),
            commit=CommitCfg.parse(_require(raw, "commit", where="stity")),
            vad=VadCfg.parse(raw.get("vad")),
            gpu_memory_utilization=float(t_kwargs["gpu_memory_utilization"]),
            transcription_kwargs=t_kwargs,
            translation_kwargs=x_kwargs,
        )


@dataclass(frozen=True)
class PacingCfg:
    chunk_size_ms: int
    send_interval_ms: int
    trailing_silence_ms: int | None

    @staticmethod
    def parse(raw: Any) -> "PacingCfg":
        raw = raw or {}
        if not isinstance(raw, dict):
            raise BenchConfigError("pacing: expected a mapping")
        _reject_unknown(raw, {"chunk_size_ms", "send_interval_ms", "trailing_silence_ms"},
                        where="pacing")
        return PacingCfg(
            chunk_size_ms=int(raw.get("chunk_size_ms", 200)),
            send_interval_ms=int(raw.get("send_interval_ms", 200)),
            trailing_silence_ms=(int(raw["trailing_silence_ms"])
                                 if raw.get("trailing_silence_ms") is not None else None),
        )

    @property
    def realtime(self) -> bool:
        return self.send_interval_ms > 0


@dataclass(frozen=True)
class LogsCfg:
    top_k: int
    rank_by: str
    order: str

    @staticmethod
    def parse(raw: Any, *, metrics: list[str]) -> "LogsCfg":
        raw = raw or {}
        if not isinstance(raw, dict):
            raise BenchConfigError("logs: expected a mapping")
        _reject_unknown(raw, {"top_k", "rank_by", "order"}, where="logs")
        rank_by = raw.get("rank_by") or metrics[0]
        if rank_by not in metrics:
            raise BenchConfigError(
                f"logs.rank_by={rank_by!r} is not in metrics {metrics}"
            )
        order = raw.get("order", "worst")
        if order not in ("worst", "best"):
            raise BenchConfigError(f"logs.order must be 'worst' or 'best', got {order!r}")
        return LogsCfg(top_k=int(raw.get("top_k", 10)), rank_by=rank_by, order=order)


@dataclass(frozen=True)
class BenchConfig:
    name: str
    dataset: DatasetCfg
    languages: LanguagesCfg
    stity: StityCfg
    metrics: list[str]
    pacing: PacingCfg
    logs: LogsCfg
    raw: dict
    overrides: dict

    def streaming_config_kwargs(self) -> dict:
        """Every StreamingConfig field, explicitly.

        Built here rather than in the driver so this module never imports the ASR
        server (which pulls in torch and vLLM); tests/test_config_resolve.py checks
        the key set against the real dataclass. Nothing is left to a default,
        because a default that moves with the checkpoint is exactly the landmine
        this framework exists to defuse.

        Order: framework defaults, then the run's commit/vad/language resolution,
        then each component's own contribution -- so a per-model setting wins.
        """
        st = self.stity
        kwargs = dict(
            model_path="Qwen/Qwen3-ASR-1.7B",
            gpu_memory_utilization=st.gpu_memory_utilization,
            max_new_tokens=128,
            adapter_en="",
            adapter_ko="",
            no_lora=True,
            max_lora_rank=16,
            chunk_size_sec=2.0,
            unfixed_chunk_num=2,
            unfixed_token_num=5,
            beam_size=1,
            enforce_eager=False,
            no_vad=not st.vad.enabled,
            enable_dot_commit=st.commit.enable_dot_commit,
            always_commit=st.commit.always_commit,
            dot_commit_confirm=st.commit.dot_commit_confirm,
            dot_commit_stall_chunks=st.commit.dot_commit_stall_chunks,
            rep_dedup=st.commit.rep_dedup,
            restrict_languages=self.languages.restrict,
            enable_correction=False,
            correction_model="gpt-5.4-mini",
            api_key=None,
            enable_gpt_translation=False,
            translation_model="gpt-5.4-mini",
            context_window=5,
            google_context=False,
            local_translation_context=0,
            no_translation=False,
            record_audio=False,
            host="in-process",
            port=0,
            no_idle_shutdown=True,
            idle_shutdown_sec=0,
            close_timeout=None,
        )
        kwargs.update(st.transcription_kwargs)
        kwargs.update(st.translation_kwargs)
        return kwargs

    def resolved(self) -> dict:
        return {
            "dataset": asdict(self.dataset),
            "languages": asdict(self.languages),
            "stity": {
                "transcription": {"name": self.stity.transcription.name,
                                  **self.stity.transcription.options},
                "translation": {"name": self.stity.translation.name,
                                **self.stity.translation.options},
                "commit": asdict(self.stity.commit),
                "vad": asdict(self.stity.vad),
                "gpu_memory_utilization": self.stity.gpu_memory_utilization,
                "transcription_kwargs": dict(self.stity.transcription_kwargs),
                "translation_kwargs": dict(self.stity.translation_kwargs),
            },
            "metrics": list(self.metrics),
            "pacing": asdict(self.pacing),
            "logs": asdict(self.logs),
            "streaming_config": self.streaming_config_kwargs(),
        }

    def fingerprint(self) -> str:
        payload = json.dumps(self.resolved(), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _apply_overrides(raw: dict, overrides: dict) -> dict:
    out = dict(raw)
    for key, value in overrides.items():
        if value is None:
            continue
        if key == "limit":
            ds = dict(out.get("dataset") or {})
            if isinstance(out.get("dataset"), str):
                ds = {"name": out["dataset"]}
            ds["limit"] = value
            out["dataset"] = ds
        elif key == "name":
            out["name"] = value
        elif key == "top_k":
            lg = dict(out.get("logs") or {})
            lg["top_k"] = value
            out["logs"] = lg
        else:
            raise BenchConfigError(f"unsupported override {key!r}")
    return out


def parse(raw: dict, *, overrides: dict | None = None) -> BenchConfig:
    if not isinstance(raw, dict):
        raise BenchConfigError("config root: expected a mapping")
    overrides = {k: v for k, v in (overrides or {}).items() if v is not None}
    merged = _apply_overrides(raw, overrides)
    _reject_unknown(merged, {"name", "dataset", "languages", "stity", "metrics", "pacing", "logs"},
                    where="config root")

    name = _require(merged, "name", where="config root")
    if not isinstance(name, str) or not name.strip():
        raise BenchConfigError("name: expected a non-empty string")

    metrics_raw = merged.get("metrics") or ["wer"]
    if not isinstance(metrics_raw, list) or not metrics_raw:
        raise BenchConfigError("metrics: expected a non-empty list")
    unknown = [m for m in metrics_raw if m not in KNOWN_METRICS]
    if unknown:
        raise BenchConfigError(f"metrics: unknown {unknown} (available: {list(KNOWN_METRICS)})")
    metrics = list(dict.fromkeys(metrics_raw))

    pacing = PacingCfg.parse(merged.get("pacing"))
    latency_requested = sorted(LATENCY_METRICS & set(metrics))
    if latency_requested and not pacing.realtime:
        raise BenchConfigError(
            f"metrics {latency_requested} need real-time pacing, but "
            f"pacing.send_interval_ms=0. Latency is meaningless when audio is pushed "
            f"faster than real time."
        )

    languages = LanguagesCfg.parse(_require(merged, "languages", where="config root"))

    return BenchConfig(
        name=name.strip(),
        dataset=DatasetCfg.parse(_require(merged, "dataset", where="config root")),
        languages=languages,
        stity=StityCfg.parse(_require(merged, "stity", where="config root"),
                             src_lang=languages.source_langs[0]),
        metrics=metrics,
        pacing=pacing,
        logs=LogsCfg.parse(merged.get("logs"), metrics=metrics),
        raw=raw,
        overrides=overrides,
    )


def load(path: str | Path, *, overrides: dict | None = None) -> BenchConfig:
    p = Path(path)
    if not p.is_file():
        raise BenchConfigError(f"config file not found: {p}")
    with open(p, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return parse(raw, overrides=overrides)
