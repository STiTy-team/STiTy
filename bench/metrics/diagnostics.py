"""Numbers that say what went wrong, next to the scores that say how much.

None of these replace WER/BLEU/COMET. They exist because two of those were found to
hide failures (WORKLOG M2: COMET rewards Korean passed through to English readers) and
because the upgrade experiments each target one mechanism, which needs its own counter:

- `passthrough_share`: translations equal to their source while the source is written
  in another script than the target -- mislabelled text that was never translated (T1).
- `wrong_script_share`: translations whose letters are mostly not the target's script.
- `leak_share`: translations with more than 15% of their letters in a foreign script
  (Qwen3.5 writing Chinese into Korean).
- `translation_failures`: commits that needed a translation and got an empty one (T2).
- `commit_gap_p90_sec` / `commit_gap_max_sec`: audio time between consecutive commits
  of one item; the latency tail the SEG policy is known for.
- `speech_segment_*_sec`: VAD segment lengths (N1: segments too long in conversation).
- `commits_per_min`: commits per minute of audio, i.e. translation calls per minute.
- `term_recall` / `term_translation_recall`: glossary terms said in the reference that
  appear in the transcript / in the translation (only when the dataset names `terms`).
"""
from statistics import median

import numpy as np

from core.utils import langs, scripts

from .common import succeeded, transcribed, translated

LEAK_SHARE = 0.15


def diagnostics(rows: list[dict], target: str, glossary=None) -> dict:
    rows = succeeded(rows)
    values = {
        **_translation_health(rows, target),
        **_commit_gaps(rows),
        **_speech_segments(rows),
    }
    minutes = sum(float(r.get("audio_sec") or 0) for r in rows) / 60
    commits = sum(len(transcribed(r["records"])) for r in rows)
    values["commits_per_min"] = commits / minutes if minutes else None
    if glossary is not None:
        values.update(_term_recall(rows, target, glossary))
    return {key: value for key, value in values.items() if value is not None}


def _needs_translation(record: dict, target: str) -> bool:
    return bool((record.get("original") or "").strip()) and record.get("language") != target \
        and (record.get("target_lang") or target) == target


def _translation_health(rows: list[dict], target: str) -> dict:
    records = [r for row in rows for r in translated(row["records"])
               if (r.get("target_lang") or target) == target and (r.get("original") or "").strip()]
    if not records:
        return {}
    other_script = [r for r in records if not scripts.matches_language(r["original"], target)]
    passthrough = [r for r in other_script if (r.get("translation") or "") == r["original"]]
    produced = [r for r in records if (r.get("translation") or "").strip()]
    wrong = [r for r in produced if not scripts.matches_language(r["translation"], target)]
    leaks = [r for r in produced if scripts.foreign_share(r["translation"], target) > LEAK_SHARE]
    needed = [r for r in records if _needs_translation(r, target)]
    failed = [r for r in needed if not (r.get("translation") or "").strip()]
    return {
        "passthrough_share": len(passthrough) / len(other_script) if other_script else None,
        "wrong_script_share": len(wrong) / len(produced) if produced else None,
        "leak_share": len(leaks) / len(produced) if produced else None,
        "translation_failures": len(failed),
    }


def _commit_gaps(rows: list[dict]) -> dict:
    gaps = []
    for row in rows:
        times = sorted(r["decision_audio_sec"] for r in transcribed(row["records"])
                       if r.get("decision_audio_sec") is not None)
        gaps += [b - a for a, b in zip(times, times[1:])]
    if not gaps:
        return {}
    return {"commit_gap_p90_sec": float(np.percentile(gaps, 90)),
            "commit_gap_max_sec": float(max(gaps))}


def _speech_segments(rows: list[dict]) -> dict:
    lengths = [r["ended_at"] - r["started_at"] for row in rows for r in row["records"]
               if r.get("type") == "speech" and r.get("ended_at") is not None]
    if not lengths:
        return {}
    return {"speech_segment_p50_sec": float(median(lengths)),
            "speech_segment_p90_sec": float(np.percentile(lengths, 90)),
            "speech_segment_max_sec": float(max(lengths))}


def _term_recall(rows: list[dict], target: str, glossary) -> dict:
    heard = said = carried = expected = 0
    for row in rows:
        sentences = row.get("reference_segmentation") or [
            {"transcript": row.get("reference") or "", "src_lang": row["src_lang"]}]
        hypothesis = (row.get("transcription_output") or "").lower()
        translation = " ".join(r.get("translation") or "" for r in translated(row["records"])
                               if (r.get("target_lang") or target) == target).lower()
        for sentence in sentences:
            raw_lang = sentence.get("src_lang") or row["src_lang"]
            lang = langs.norm_code(raw_lang) or raw_lang
            for entry in glossary.forms_in(sentence.get("transcript") or "", lang):
                said += 1
                heard += entry[lang].lower() in hypothesis
                if lang != target and target in entry:
                    expected += 1
                    carried += entry[target].lower() in translation
    return {"term_recall": heard / said if said else None,
            "term_translation_recall": carried / expected if expected else None}
