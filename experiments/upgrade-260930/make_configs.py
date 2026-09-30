"""Write the pipeline configs of the 2026-09-30 upgrade experiments.

    python experiments/upgrade-260930/make_configs.py

Every variant is its base config with ONE change, so a run compares against its base
and nothing else moved. The table below is the source of truth; the YAML files under
configs/pipelines/ are generated from it and committed so bench can read them. Change
the table and re-run rather than editing the YAML by hand.
"""
import copy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "configs" / "pipelines"

MODELS = {
    "ko": "Doo12/Qwen3-ASR-1.7B-ko-silence-v4c900-merged",
    "en": "Doo12/Qwen3-ASR-1.7B-en-silence-c80-merged",
}


def base(lang: str) -> dict:
    return {
        "pipeline": {
            "name": "cascade:v2",
            "transcription": {
                "name": "qwen-seg:v2",
                "model": MODELS[lang],
                "chunk_size_sec": 2.0,
                "max_new_tokens": 128,
                "dot_commit_confirm": True,
                "dot_commit_stall_chunks": 1,
                "rep_dedup": True,
            },
            "translation": {
                "name": "qwen3.5",
                "url": "http://127.0.0.1:8100/v1",
                "model": "Qwen/Qwen3.5-4B",
                "gpu_memory_utilization": 0.52,
                "context_turns": 1,
            },
            "vad": {"name": "silero", "min_silence_ms": 800},
        },
        "commit": "seg",
        "gpu_memory_utilization": 0.30,
    }


def asr(**flags):
    return lambda c: c["pipeline"]["transcription"].update(flags)


def pipeline(**options):
    return lambda c: c["pipeline"].update(options)


def vad(spec: dict):
    return lambda c: c["pipeline"].__setitem__("vad", spec)


def part(kind: str, spec: dict):
    return lambda c: c["pipeline"].__setitem__(kind, spec)


def mt(**options):
    def apply(c):
        c["pipeline"]["translation"]["name"] = "qwen3.5:v2"
        c["pipeline"]["translation"].update(options)
    return apply


def la(**options):
    def apply(c):
        t = c["pipeline"]["transcription"]
        for key in ("dot_commit_confirm", "dot_commit_stall_chunks", "rep_dedup"):
            t.pop(key)
        t["name"] = "qwen-la"
        t.update(options)
    return apply


ALL_ASR_FIXES = dict(
    no_speech_since_vad_reset=True, keep_held_fragment=True, strict_boundary_dedup=True,
    carry_uncommitted_audio=True, resync_cursor=True, loop_guard=True,
    carry_on_loop_reset=True, drop_language_lists=True, clean_partials=True,
    dedup_dot_carry=True)

CANNED_EN = [
    "I want to buy a new car.", "I'm sorry, sir.", "I'm afraid I can't help you.",
    "I'm sorry, sir. I'm afraid I can't help you.", "I'm sorry, I'm not sure what you mean.",
    "How may I help you?", "I'm afraid we're out of that size.",
]
FILTER_BASE = {"name": "commit-rules", "fillers": True, "foreign_script": True, "repeats": 2}
STYLE_KO = {"ko": "Write natural spoken Korean in polite 해요체, and address seniors with "
                  "their title (for example 부장님) rather than 씨."}

VARIANTS = {
    "off": ("Parity check: v2 components with every change off; must match the v1 config", []),
    "nospeech": ("A1 no-speech gate measured from the last VAD reset", [asr(no_speech_since_vad_reset=True)]),
    "fragment": ("A2 held comma fragment is not remembered until it is emitted", [asr(keep_held_fragment=True)]),
    "boundary": ("A3 seg-boundary dedup only drops exact tail repeats", [asr(strict_boundary_dedup=True)]),
    "carry": ("A4 header reset carries all uncommitted audio and flush decodes it", [asr(carry_uncommitted_audio=True)]),
    "resync": ("A5/N5 committed cursor re-anchored by text after a rewrite", [asr(resync_cursor=True)]),
    "loopguard": ("A5/N6 loop detection on normalized tokens, reset keeps the last chunk", [asr(loop_guard=True, carry_on_loop_reset=True)]),
    "langlist": ("A8 commits made only of language names are dropped", [asr(drop_language_lists=True)]),
    "partials": ("A9 language headers stripped from partials", [asr(clean_partials=True)]),
    "dotcarry": ("N7 dot-switch carry deduplicated against the last commit for any trigger", [asr(dedup_dot_carry=True)]),
    "tailkeep": ("N8 keep 0.3 s (not 0.1 s) of trailing silence before the flush decode", [asr(tail_keep_sec=0.3)]),
    "asrfix": ("All qwen-seg:v2 fixes on (A1-A9, N5-N7)", [asr(**ALL_ASR_FIXES)]),
    "split": ("A10 chunk split at the VAD end sample before the flush", [pipeline(split_at_vad=True)]),
    "label": ("B7/T1/N2 language label per commit from its script", [part("labeler", {"name": "script"})]),
    "filter": ("A5/A7 filler-only, foreign-script and repeated commits dropped", [part("filter", FILTER_BASE)]),
    "filter-canned": ("A6 filter plus the en finetune's canned phrases", [part("filter", {**FILTER_BASE, "canned_phrases": CANNED_EN})]),
    "open": ("M1 language header not restricted to the dataset's languages", [asr(restrict_languages=False)]),
    "chunk1.0": ("Chunk 1.0 s instead of 2.0 s", [asr(chunk_size_sec=1.0)]),
    "chunk1.5": ("Chunk 1.5 s instead of 2.0 s", [asr(chunk_size_sec=1.5)]),
    "vad250": ("Silero min silence 250 ms", [vad({"name": "silero", "min_silence_ms": 250})]),
    "vad400": ("Silero min silence 400 ms", [vad({"name": "silero", "min_silence_ms": 400})]),
    "vad550": ("Silero min silence 550 ms", [vad({"name": "silero", "min_silence_ms": 550})]),
    "vadthr35": ("Silero threshold 0.35", [vad({"name": "silero", "min_silence_ms": 800, "threshold": 0.35})]),
    "vadthr65": ("Silero threshold 0.65", [vad({"name": "silero", "min_silence_ms": 800, "threshold": 0.65})]),
    "vad2fixed": ("Parity check: silero:v2 with the fixed tuner; must match silero", [vad({"name": "silero:v2", "min_silence_ms": 800, "tuner": "fixed"})]),
    "vad-ramp": ("silero:v2 ramp: min silence 800 -> 300 ms over 8 s, 100 ms after 15 s", [vad({"name": "silero:v2", "min_silence_ms": 800, "tuner": {"name": "ramp", "end_ms": 300, "ramp_sec": 8, "max_segment_sec": 15}})]),
    "vad-adapt": ("silero:v2 adaptive: min silence from the speaker's own pauses", [vad({"name": "silero:v2", "min_silence_ms": 800, "tuner": {"name": "adaptive", "pauses": True}})]),
    "vad-adapt-q75": ("silero:v2 adaptive: 75th percentile pause x1.0, floor 200 ms", [vad({"name": "silero:v2", "min_silence_ms": 800, "tuner": {"name": "adaptive", "pauses": True, "pause_quantile": 0.75, "pause_factor": 1.0, "floor_ms": 200}})]),
    "vad-ramp-fast": ("silero:v2 ramp: 800 -> 250 ms over 5 s, 100 ms after 10 s", [vad({"name": "silero:v2", "min_silence_ms": 800, "tuner": {"name": "ramp", "end_ms": 250, "ramp_sec": 5, "max_segment_sec": 10}})]),
    "vad-noise": ("silero:v2 adaptive: threshold raised over the noise floor", [vad({"name": "silero:v2", "min_silence_ms": 800, "tuner": {"name": "adaptive", "pauses": False, "noise_floor": True}})]),
    "vad-adapt-noise": ("silero:v2 adaptive: pauses and noise floor together", [vad({"name": "silero:v2", "min_silence_ms": 800, "tuner": {"name": "adaptive", "pauses": True, "noise_floor": True}})]),
    "vad-turn": ("silero:v2 smart-turn: end when Smart Turn v3.2 says the turn is over, else 1.5 s", [vad({"name": "silero:v2", "min_silence_ms": 800, "tuner": {"name": "smart-turn", "probe_ms": 250, "max_silence_ms": 1500}})]),
    "denoise-vad": ("Spectral denoiser feeds VAD only; ASR hears raw audio", [part("enhancement", {"name": "spectral", "vad_input": "enhanced", "asr_input": "raw"})]),
    "denoise-mix": ("Spectral denoiser: VAD enhanced, ASR 0.2 raw + 0.8 enhanced", [part("enhancement", {"name": "spectral", "vad_input": "enhanced", "asr_input": "mix", "mix_weight": 0.2})]),
    "bias": ("ASR context biasing with the northstar glossary", [asr(bias_glossary="northstar-260928")]),
    "bias-recent": ("ASR context: the last 3 commits of the conversation", [asr(bias_recent_commits=3)]),
    "mt-v2": ("Parity check: qwen3.5:v2 with every option off; must match qwen3.5", [mt()]),
    "mt-leak": ("Translator retries once when the output leaks another script", [mt(leak_retry=True)]),
    "mt-meta": ("Translator retries when it answers the instruction instead of translating", [mt(meta_guard=True)]),
    "mt-style": ("Translator told the Korean register (해요체, titles)", [mt(style=STYLE_KO)]),
    "mt-glossary": ("Translator given the northstar glossary pairs found in the line", [mt(glossary="northstar-260928")]),
    "mt-cont": ("Fragments translated as a continuation of the sentence on screen", [mt(continuation=True)]),
    "mt-ctx0": ("No earlier turn in the translator prompt", [mt(context_turns=0)]),
    "mt-ctx3": ("Three earlier turns in the translator prompt", [mt(context_turns=3)]),
    "la": ("LocalAgreement-2 commits at sentence boundaries (qwen-la) instead of SEG", [la()]),
    "la-clause": ("qwen-la committing at clause boundaries too", [la(boundary="clause")]),
    "la-chunk1.0": ("qwen-la with 1.0 s chunks", [la(chunk_size_sec=1.0)]),
    "all": ("Every fix expected to help: asrfix + split + label + filter + vad-ramp + mt leak/meta/style",
            [asr(**ALL_ASR_FIXES), pipeline(split_at_vad=True), part("labeler", {"name": "script"}),
             part("filter", FILTER_BASE),
             vad({"name": "silero:v2", "min_silence_ms": 800, "tuner": {"name": "ramp", "end_ms": 300, "ramp_sec": 8, "max_segment_sec": 15}}),
             mt(leak_retry=True, meta_guard=True, style=STYLE_KO)]),
}

EN_ONLY = {"fragment", "boundary", "langlist", "filter-canned"}
KO_ONLY = {"loopguard", "dotcarry", "mt-style"}


def name_of(lang: str, variant: str) -> str:
    asr_part = "qwen-la" if variant == "la" or variant.startswith("la-") else "qwen-seg-v2"
    mt_part = "qwen3.5-v2-4b" if variant.startswith("mt-") or variant == "all" else "qwen3.5-4b"
    suffix = variant.removeprefix("mt-").replace("v2", "off") if variant.startswith("mt-") \
        else variant
    return f"asr.{asr_part}-{lang}+mt.{mt_part}_{suffix}"


def build(lang: str, variant: str) -> dict:
    description, changes = VARIANTS[variant]
    config = copy.deepcopy(base(lang))
    for change in changes:
        change(config)
    tags = ["upgrade-260930", "cascade", f"{lang}-asr", variant]
    return {"meta": {"description": f"{lang} finetune. {description}", "tags": tags}, **config}


def main() -> None:
    written = []
    for lang in MODELS:
        for variant in VARIANTS:
            if (variant in EN_ONLY and lang != "en") or (variant in KO_ONLY and lang != "ko"):
                continue
            path = OUT / f"{name_of(lang, variant)}.yml"
            path.write_text(yaml.safe_dump(build(lang, variant), allow_unicode=True,
                                           sort_keys=False, width=100), encoding="utf-8")
            written.append(path.name)
    print(f"wrote {len(written)} configs to {OUT}")


if __name__ == "__main__":
    main()
