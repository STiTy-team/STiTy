"""run14 의 v0..v5 프롬프트로 test 100문장을 분절해 bleu_eval 이 읽는 prompt_eval JSON 을 만든다.
GPU 불필요(LLM 분절만). v3(best) 는 최종 test 평가 캐시라 호출 0."""
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, "/home/mobility/STiTy")
from core.meaning_segmentator.autoseg.runtime import data, pipeline as P
from core.meaning_segmentator.autoseg.infra.gateway import Gateway
from core.meaning_segmentator.autoseg.paths import RUNS_DIR

ap = argparse.ArgumentParser()
ap.add_argument("--versions", nargs="+", type=int, default=[3, 0, 1, 2, 4, 5])
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--budget", type=float, default=8.0)
a = ap.parse_args()

run = RUNS_DIR / "en2x/en-multi/run14"
cfg = json.loads((run / "config.json").read_text())
measured = json.loads((run / "measured_profile.json").read_text())
spaced, trailing_punct = data.profile_settings(measured)
sents = data.read_split(run / "data" / "test.json")
if a.limit: sents = sents[: a.limit]
t_grid = sorted(cfg["final_t_grid"]); min_gap = int(cfg["min_gap"]); t_floor = int(cfg["t_floor"])
gw = Gateway(provider=cfg.get("provider") or "letsur", model=cfg["model"], budget=a.budget)
cache = P.JsonCache(run / "cache" / "segment.json")
need = lambda t: P.coverage_need(t, t_floor, spaced, min_gap)
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
try:
    for v in a.versions:
        prompt = (run / f"iter_{v:02d}" / "prompt.txt").read_text(encoding="utf-8")
        legacy = P.check_tag_convention(prompt)
        if legacy: raise SystemExit(legacy)
        c0 = gw.usage.snapshot()["cost"]
        segs, first = P.segment_batch(
            gw, prompt, [s.text for s in sents], cache=cache, workers=int(cfg.get("workers", 16)),
            validate_fn=lambda t, o: P.validate("", t, o, spaced, True, need(t)),
            normalize_fn=lambda t, o: P.normalize_tags(o, spaced, trailing_punct, min_gap=min_gap),
            need_fn=need, reasoning_effort=cfg.get("seg_reasoning_effort"),
            batch_size=int(cfg.get("batch_size", 6)))
        rows = []
        for s, seg in zip(sents, segs):
            viol = P.validate(s.id, s.text, seg, spaced, True, need(s.text))
            by_T = {}
            for T in t_grid:
                cut, missing = P.truncate(seg, T, spaced, min_gap)
                pieces = P.split_segments(cut) or [s.text]
                by_T[str(T)] = {"seg_text": cut, "k": len(pieces), "missing_boundaries": missing,
                                "pieces_src": pieces}
            rows.append({"id": s.id, "text": s.text, "seg_text": seg, "valid": not viol,
                         "full_trans": None, "by_T": by_T})
        out = RUNS_DIR / "en2x/en-multi" / f"run14_v{v}" / "prompt_eval" / f"v{v}_test.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"prompt_file": f"run14/iter_{v:02d}/prompt.txt", "split": "test",
                                   "t_grid": t_grid, "min_gap": min_gap, "t_floor": t_floor,
                                   "src_spaced": spaced, "tag_convention": "score", "rows": rows},
                                  ensure_ascii=False, indent=1))
        cache.flush()
        log(f"v{v}: {len(rows)}문장 1차통과 {sum(first)}/{len(first)} 유효 {sum(r['valid'] for r in rows)} "
            f"비용 +${gw.usage.snapshot()['cost'] - c0:.3f} → {out.relative_to(RUNS_DIR)}")
finally:
    cache.flush(); gw.close()
    log(f"total cost ${gw.usage.snapshot()['cost']:.3f}")
