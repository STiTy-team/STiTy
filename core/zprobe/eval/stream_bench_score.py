"""Score a stream_bench.py run with bench's own latency code, the way bench/seamless/seamless_score.py scores
SeamlessStreaming: YAAL from the commit segments (fed_sec = computation-unaware, wall_sec = computation-aware),
and comet_inputs.jsonl + summary.json so `python -m bench.metrics.comet --run-dir` adds COMET.

There is no source transcript, so WER/CER and Token Emission are left out (as for SeamlessStreaming's S2S run).

Run from the STiTy-bench root with the bench env:
    PYTHONPATH=.:bench/seamless bench/.venv/bin/python <this> --export DIR --run X.jsonl --name NAME --run-dir DIR"""
import argparse, json, math
from pathlib import Path

from omnisteval import YAALScorer
from omnisteval.scoring import LAALScorer
from seamless_score import yaal_instance


def read(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True); ap.add_argument("--run", required=True)
    ap.add_argument("--name", required=True); ap.add_argument("--run-dir", required=True)
    a = ap.parse_args()
    items = {u["utt_id"]: u for u in read(Path(a.export) / "manifest.jsonl")}
    runs = {r["utt_id"]: r for r in read(a.run)}
    u0 = next(iter(items.values())); pair = f"{u0['src_lang']}-{u0['tgt_lang']}"
    cu, ca = YAALScorer(computation_aware=False), YAALScorer(computation_aware=True)
    lcu, lca = LAALScorer(computation_aware=False), LAALScorer(computation_aware=True)
    instances, per_item, comet_inputs = [], [], []
    for uid, u in items.items():
        r = runs.get(uid)
        row = {"id": uid, "ok": bool(r), "duration_sec": u["duration_sec"], "mt_text": r and r["text"],
               "reference": u["src_text"], "reference_translation": u["tgt_text"]}
        if r:
            inst = yaal_instance(r["segments"], u["duration_sec"] * 1000.0, u["tgt_text"], u["tgt_lang"])
            instances.append(inst)
            y, yc = cu([inst]), ca([inst])
            row["yaal_ms"] = None if math.isnan(y) else y
            row["yaal_ca_ms"] = None if math.isnan(yc) else yc
            row["segments"] = len(r["segments"]); row["decodes"] = r.get("decodes")
            comet_inputs.append({"pair": pair, "src": u["src_text"], "mt": r["text"], "ref": u["tgt_text"]})
        per_item.append(row)
    y, yc = cu(instances), ca(instances)
    la, lac = lcu(instances), lca(instances)
    ok = [r for r in per_item if r["ok"]]
    nan = lambda v: None if math.isnan(v) else v
    metrics = {"yaal_ms": nan(y), "yaal_ca_ms": nan(yc), "laal_ms": nan(la), "laal_ca_ms": nan(lac),
               "segments_per_item": sum(r["segments"] for r in ok) / len(ok),
               "decodes_per_item": sum(r["decodes"] or 0 for r in ok) / len(ok)}
    counts = {"items": len(items), "errored": len(items) - len(ok),
              "empty": sum(1 for r in ok if not (r["mt_text"] or "").strip()),
              "yaal_items": sum(1 for r in ok if r.get("yaal_ca_ms") is not None)}
    run = Path(a.run_dir); run.mkdir(parents=True, exist_ok=True)
    old = run / "summary.json"
    if old.exists():  # rescoring latency keeps the COMET that bench.metrics.comet already added
        prev = json.loads(old.read_text(encoding="utf-8")).get("metrics", {})
        metrics.update({k: v for k, v in prev.items() if k.startswith("comet")})
    (run / "items.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in per_item), encoding="utf-8")
    (run / "comet_inputs.jsonl").write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in comet_inputs), encoding="utf-8")
    (run / "summary.json").write_text(json.dumps({
        "name": a.name, "metrics": metrics,
        "unavailable": {"wer": "direct speech translation, no source transcript",
                        "token_emission_ms": "no source-transcript timing to align"},
        "counts": counts}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"metrics": metrics, "counts": counts}, ensure_ascii=False))


if __name__ == "__main__":
    main()
