"""Lookahead ("invention") rate of a bench run: the share of output pieces that already say something the model
could not have heard yet.

Definition. Only pieces with type == "translated" and target_lang == --target are used. A piece is judged only when
its decision_audio_sec falls inside a reference segment whose src_lang != target (a cross-language segment);
otherwise it is "unjudged". For a judged piece at time t in segment S:
  heard  = alignment words with end <= t lying inside S [offset, offset+duration]
  known  = content(translation into target of " ".join(heard), target)
  full   = content(translation into target of S.transcript, target)
  piece_content = content(piece.translation, target)
  violation = bool((piece_content - known) & full)
Translation is Hy-MT2 (hy_word_prefixes.load_model/translate), cached by (segment index, number of heard words).
--no-hy: known/full use the SOURCE words themselves (crude same-script fallback, only to exercise the pipeline).

usage: PYTHONPATH=. python lookahead_rate.py --run-dir bench/runs/<..>/<run> [--target ko] [--no-hy] [--batch 32]
Prints "LOOKAHEAD run=.. pieces=.. judged=.. violations=.. rate=.. unjudged=.." and writes <run_dir>/lookahead.json."""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "labels"))


def heard_words(alignment, t, seg):
    """Source words with end <= t that lie inside seg [offset, offset+duration]."""
    lo, hi = seg["offset"], seg["offset"] + seg["duration"]
    eps = 1e-6
    return [w["word"] for w in alignment if w["end"] <= t + eps and w["start"] >= lo - eps and w["end"] <= hi + eps]


def judge(piece_content, known, full):
    """True = violation: the piece has content that is not yet known but is in the full translation."""
    return bool((set(piece_content) - set(known)) & set(full))


def segment_at(segments, t):
    """Index of the segment with offset <= t < offset+duration, else None."""
    for i, s in enumerate(segments):
        if s["offset"] <= t < s["offset"] + s["duration"]:
            return i
    return None


def collect(items, target):
    """Per item: list of (piece record, seg index or None, judged?). Judged = inside a cross-language segment."""
    out = []
    for it in items:
        segs = it.get("reference_segmentation") or []
        rows = []
        for r in it.get("records", []):
            if r.get("type") != "translated" or r.get("target_lang") != target:
                continue
            t = r.get("decision_audio_sec")
            si = segment_at(segs, t) if t is not None else None
            judged = si is not None and segs[si]["src_lang"] != target
            rows.append((r, si if judged else None, judged))
        out.append((it, rows))
    return out


def run(items, target, no_hy=False, batch=32):
    from build_prefix import content
    data = collect(items, target)
    need = {}  # (item idx, seg idx, nheard) -> (text, src_lang)
    prep = []
    for ii, (it, rows) in enumerate(data):
        segs, al = it.get("reference_segmentation") or [], it.get("reference_alignment") or []
        for r, si, judged in rows:
            if not judged:
                prep.append(None)
                continue
            hw = heard_words(al, r["decision_audio_sec"], segs[si])
            prep.append((ii, si, hw))
            need[(ii, si, len(hw))] = (" ".join(hw), segs[si]["src_lang"])
            need[(ii, si, -1)] = (segs[si]["transcript"], segs[si]["src_lang"])
    if no_hy:
        tr = {k: v[0] for k, v in need.items()}
    else:
        from hy_word_prefixes import load_model, translate
        torch, trn, tok, model = load_model()
        keys = list(need)
        jobs = [(need[k][0], need[k][1], target) if need[k][0] else ("", need[k][1], target) for k in keys]
        res = translate(jobs, batch, torch, trn, tok, model)
        tr = {k: ("" if not need[k][0] else v) for k, v in zip(keys, res)}
    scenes, viol, pieces, judged_n, k = [], [], 0, 0, 0
    for ii, (it, rows) in enumerate(data):
        sj = sv = 0
        for r, si, judged in rows:
            pieces += 1
            p = prep[k]
            k += 1
            if p is None:
                continue
            _, si, hw = p
            known = content(tr[(ii, si, len(hw))], target) if hw else set()
            full = content(tr[(ii, si, -1)], target)
            pc = content(r.get("translation", ""), target)
            judged_n += 1
            sj += 1
            if judge(pc, known, full):
                sv += 1
                viol.append({"item": it.get("id"), "decision_audio_sec": r["decision_audio_sec"],
                             "translation": r.get("translation", ""), "extra": sorted((pc - known) & full)})
        scenes.append({"id": it.get("id"), "pieces": len(rows), "judged": sj, "violations": sv})
    return pieces, judged_n, viol, scenes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--target", default="ko")
    ap.add_argument("--no-hy", action="store_true")
    ap.add_argument("--batch", type=int, default=32)
    a = ap.parse_args()
    items = [json.loads(l) for l in open(os.path.join(a.run_dir, "items.jsonl"), encoding="utf-8") if l.strip()]
    pieces, judged, viol, scenes = run(items, a.target, a.no_hy, a.batch)
    mode = "no-hy" if a.no_hy else "hy"
    name = os.path.basename(os.path.normpath(a.run_dir))
    rate = len(viol) / judged if judged else 0.0
    print(f"LOOKAHEAD run={name} pieces={pieces} judged={judged} violations={len(viol)} rate={rate:.3f} "
          f"unjudged={pieces - judged} mode={mode}")
    print("per-scene: " + " ".join(f"{s['id']}:{s['violations']}/{s['judged']}" for s in scenes))
    with open(os.path.join(a.run_dir, "lookahead.json"), "w", encoding="utf-8") as f:
        json.dump({"run": name, "target": a.target, "mode": mode, "pieces": pieces, "judged": judged,
                   "violations": len(viol), "rate": rate, "unjudged": pieces - judged, "scenes": scenes,
                   "violating_pieces": viol}, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
