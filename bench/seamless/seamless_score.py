"""Score a SeamlessStreaming run with bench's own metric code.

- WER/CER: bench.metrics.transcription on the ASR-mode transcripts.
- YAAL: bench's omnisteval inputs, built from the S2S text segments. A segment's
  emission is the audio fed when it came out (computation-unaware, as bench does).
- ASR-COMET: the synthesized speech is transcribed back with Qwen/Qwen3-ASR-1.7B
  (as the teammate's S2S evaluation did), then written as bench comet_inputs.jsonl
  (src = reference transcript, mt = round-trip text, ref = reference translation).
  `python -m bench.metrics.comet --run-dir` adds COMET to summary.json.

Run from the STiTy-bench root:
    PYTHONPATH=. bench/.venv/bin/python <this> --export DIR --seamless DIR --run-dir DIR
"""

import argparse
import json
import math
from pathlib import Path

from omnisteval import Instance, YAALScorer

from bench.metrics.transcription import cer, wer
from core.utils import langs


def read(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]


def roundtrip(recs: list[dict], lang: str) -> dict[str, str]:
    import asyncio
    import inspect

    import torch
    from qwen_asr import Qwen3ASRModel

    # Same model and call as the teammate's s2s_roundtrip_score.py.
    asr = Qwen3ASRModel.from_pretrained(
        "Qwen/Qwen3-ASR-1.7B",
        dtype=torch.bfloat16,
        device_map="cuda:0",
        max_inference_batch_size=8,
        max_new_tokens=256,
    )
    name = {"en": "English", "ko": "Korean"}[lang]
    out = {}
    for r in recs:
        ret = asr.transcribe(audio=r["synth_wav"], language=name, return_time_stamps=False)
        results = asyncio.run(ret) if inspect.isawaitable(ret) else ret
        out[r["utt_id"]] = results[0].text.strip()
    return out


def yaal_instance(segments, source_ms: float, reference: str, target: str) -> Instance:
    unit = langs.laal_unit(target)
    separator = " " if unit == "word" else ""
    emitted, emitted_ca = [], []
    for text, fed_sec, wall_sec in segments:
        words = (text or "").split()
        n = len(words) if unit == "word" else len("".join(words))
        emitted += [min(fed_sec * 1000.0, source_ms)] * n
        emitted_ca += [wall_sec * 1000.0] * n
    return Instance(
        reference=separator.join(reference.split()),
        latency_unit=unit,
        source_length=source_ms,
        emission_cu=emitted,
        emission_ca=emitted_ca,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True)
    ap.add_argument("--seamless", required=True)
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()

    items = {u["utt_id"]: u for u in read(Path(args.export) / "manifest.jsonl")}
    s2s = {r["utt_id"]: r for r in read(Path(args.seamless) / "s2s.jsonl")}
    asr = {r["utt_id"]: r for r in read(Path(args.seamless) / "asr.jsonl")}
    src = next(iter(items.values()))["src_lang"]
    tgt = next(iter(items.values()))["tgt_lang"]
    pair = f"{src}-{tgt}"

    run = Path(args.run_dir)
    run.mkdir(parents=True, exist_ok=True)
    rt_path = run / "roundtrip.jsonl"
    ok_s2s = [r for r in s2s.values() if r.get("ok")]
    if rt_path.exists():
        rt = {r["utt_id"]: r["text"] for r in read(rt_path)}
    else:
        rt = roundtrip(ok_s2s, tgt)
        rt_path.write_text(
            "".join(
                json.dumps({"utt_id": k, "text": v}, ensure_ascii=False) + "\n"
                for k, v in rt.items()
            ),
            encoding="utf-8",
        )

    rows, comet_inputs, per_item = [], [], []
    scorer = YAALScorer(computation_aware=False)
    scorer_ca = YAALScorer(computation_aware=True)
    instances = []
    for uid, u in items.items():
        a, s = asr.get(uid, {}), s2s.get(uid, {})
        rows.append(
            {
                "src_lang": src,
                "reference": u["src_text"],
                "transcription_output": a.get("text", "") if a.get("ok") else "",
            }
        )
        row = {
            "id": uid,
            "asr_ok": bool(a.get("ok")),
            "s2s_ok": bool(s.get("ok")),
            "duration_sec": u["duration_sec"],
            "asr_text": a.get("text"),
            "mt_text": s.get("text"),
            "roundtrip_text": rt.get(uid),
            "reference": u["src_text"],
            "reference_translation": u["tgt_text"],
        }
        if s.get("ok"):
            inst = yaal_instance(s["segments"], u["duration_sec"] * 1000.0, u["tgt_text"], tgt)
            instances.append(inst)
            y = scorer([inst])
            row["yaal_ms"] = None if math.isnan(y) else y
            comet_inputs.append(
                {"pair": pair, "src": u["src_text"], "mt": rt.get(uid, ""), "ref": u["tgt_text"]}
            )
        per_item.append(row)

    yaal_ms = scorer(instances)
    yaal_ca_ms = scorer_ca(instances)
    metrics = {
        "wer": wer(rows),
        "cer": cer(rows),
        "yaal_ms": None if math.isnan(yaal_ms) else yaal_ms,
        "yaal_ca_ms": None if math.isnan(yaal_ca_ms) else yaal_ca_ms,
    }
    counts = {
        "items": len(items),
        "asr_errored": sum(1 for r in per_item if not r["asr_ok"]),
        "s2s_errored": sum(1 for r in per_item if not r["s2s_ok"]),
        "asr_empty": sum(1 for r in per_item if r["asr_ok"] and not (r["asr_text"] or "").strip()),
        "yaal_items": sum(1 for r in per_item if r.get("yaal_ms") is not None),
    }
    (run / "items.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in per_item), encoding="utf-8"
    )
    (run / "comet_inputs.jsonl").write_text(
        "".join(json.dumps(c, ensure_ascii=False) + "\n" for c in comet_inputs), encoding="utf-8"
    )
    (run / "summary.json").write_text(
        json.dumps(
            {
                "name": f"seamless-streaming/{pair}",
                "metrics": metrics,
                "unavailable": {
                    "token_emission_ms": "S2S has no source-transcript timing to align"
                },
                "counts": counts,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"metrics": metrics, "counts": counts}, ensure_ascii=False))


if __name__ == "__main__":
    main()
