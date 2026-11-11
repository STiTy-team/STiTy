"""Lookahead ("invention") rate of a bench run: the share of output pieces that already say something the model
could not have heard yet.

Definition. Only pieces with type == "translated" and target_lang == --target are used. A piece is judged only when
its decision_audio_sec falls inside a reference segment whose src_lang != target (a cross-language segment);
otherwise it is "unjudged". For a judged piece at time t in segment S:
  heard  = alignment words with end + PAD <= t lying inside S [offset, offset+duration] (PAD = 0.2 s, the
           same alignment margin build_prefix14 uses for training rows)
  known  = union over j = 1..len(heard) of content(translation of the first j heard words, target) - the
           cumulative set schedule_gate/hy_word_prefixes.known_sets uses, so a stem that a shorter prefix
           translation already had stays known even if a longer prefix is translated differently
  full   = content(translation into target of S.transcript, target)
  piece_content = content(piece.translation, target)
  violation = bool((piece_content - known) & full)
Translation is Hy-MT2 (hy_word_prefixes.load_model/translate), cached by (segment index, prefix length).
--no-hy: known/full use the SOURCE words themselves (crude same-script fallback, only to exercise the pipeline).

Window: a piece belongs to the segment with offset <= t < min(next segment offset, end + grace); --grace (default 2.0 s)
covers pieces decided just after a segment ends (heard = all its words). Longer gaps stay unjudged; --grace 0 is strict.
(The brief's `lag` parameter of heard_words is intentionally not implemented.)

usage: PYTHONPATH=. python lookahead_rate.py --run-dir bench/runs/<..>/<run> [--target ko] [--no-hy] [--batch 32] [--grace 2.0]
Prints "LOOKAHEAD run=.. pieces=.. judged=.. violations=.. rate=.. unjudged=.." and writes <run_dir>/lookahead.json."""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "labels"))

PAD = 0.2  # = build_prefix14.PAD: a word counts as heard only from its end + PAD


def heard_words(alignment, t, seg, pad=0.0):
    """Source words with end + pad <= t that lie inside seg [offset, offset+duration]. run() passes PAD."""
    lo, hi = seg["offset"], seg["offset"] + seg["duration"]
    eps = 1e-6
    return [w["word"] for w in alignment
            if w["end"] + pad <= t + eps and w["start"] >= lo - eps and w["end"] <= hi + eps]


def cumulative_known(prefix_translations, target, content):
    """Union of content stems over the translations of prefixes 1..n (same as hy_word_prefixes.known_sets[-1])."""
    acc = set()
    for t in prefix_translations:
        acc |= content(t or "", target)
    return acc


def judge(piece_content, known, full):
    """True = violation: the piece has content that is not yet known but is in the full translation."""
    return bool((set(piece_content) - set(known)) & set(full))


def segment_at(segments, t, grace=0.0):
    """Index of the segment with offset <= t < min(next segment offset, end + grace), else None.
    A piece decided slightly after a segment's end (within grace, before the next segment starts) still belongs to it."""
    for i, s in enumerate(segments):
        nxt = segments[i + 1]["offset"] if i + 1 < len(segments) else float("inf")
        if s["offset"] <= t < min(nxt, s["offset"] + s["duration"] + grace):
            return i
    return None


def collect(items, target, grace=0.0):
    """Per item: list of (piece record, seg index or None, judged?). Judged = inside a cross-language segment."""
    out = []
    for it in items:
        segs = it.get("reference_segmentation") or []
        rows = []
        for r in it.get("records", []):
            if r.get("type") != "translated" or r.get("target_lang") != target:
                continue
            t = r.get("decision_audio_sec")
            si = segment_at(segs, t, grace) if t is not None else None
            judged = si is not None and segs[si]["src_lang"] != target
            rows.append((r, si if judged else None, judged))
        out.append((it, rows))
    return out


def run(items, target, no_hy=False, batch=32, grace=0.0):
    from build_prefix import content
    data = collect(items, target, grace)
    need = {}  # (item idx, seg idx, nheard) -> (text, src_lang)
    prep = []
    for ii, (it, rows) in enumerate(data):
        segs, al = it.get("reference_segmentation") or [], it.get("reference_alignment") or []
        for r, si, judged in rows:
            if not judged:
                prep.append(None)
                continue
            hw = heard_words(al, r["decision_audio_sec"], segs[si], PAD)
            prep.append((ii, si, hw))
            for j in range(1, len(hw) + 1):
                need[(ii, si, j)] = (" ".join(hw[:j]), segs[si]["src_lang"])
            need[(ii, si, -1)] = (segs[si]["transcript"], segs[si]["src_lang"])
    if no_hy:
        tr = {k: v[0] for k, v in need.items()}
    else:
        from hy_word_prefixes import load_model, translate
        torch, trn, tok, model = load_model()
        keys = list(need)
        keys = [k for k in keys if need[k][0].strip()]  # empty text is never sent to Hy
        res = translate([(need[k][0], need[k][1], target) for k in keys], batch, torch, trn, tok, model)
        tr = dict(zip(keys, res))
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
            known = cumulative_known([tr.get((ii, si, j), "") for j in range(1, len(hw) + 1)], target, content)
            full = content(tr.get((ii, si, -1), ""), target)
            pc = content(r.get("translation", ""), target)
            judged_n += 1
            sj += 1
            if judge(pc, known, full):
                sv += 1
                viol.append({"item": it.get("id"), "decision_audio_sec": r["decision_audio_sec"],
                             "translation": r.get("translation", ""), "extra": sorted((pc - known) & full)})
        scenes.append({"id": it.get("id"), "pieces": len(rows), "judged": sj, "violations": sv})
    return pieces, judged_n, viol, scenes


def run_dir(path, target="ko", no_hy=False, batch=32, grace=2.0):
    items = [json.loads(l) for l in open(os.path.join(path, "items.jsonl"), encoding="utf-8") if l.strip()]
    pieces, judged, viol, scenes = run(items, target, no_hy, batch, grace)
    mode = "no-hy" if no_hy else "hy"
    name = os.path.basename(os.path.normpath(path))
    rate = len(viol) / judged if judged else 0.0
    print(f"LOOKAHEAD run={name} pieces={pieces} judged={judged} violations={len(viol)} rate={rate:.3f} "
          f"unjudged={pieces - judged} mode={mode} grace={grace}")
    print("per-scene: " + " ".join(f"{s['id']}:{s['violations']}/{s['judged']}" for s in scenes))
    res = {"run": name, "target": target, "mode": mode, "grace": grace, "pieces": pieces, "judged": judged,
           "violations": len(viol), "rate": rate, "unjudged": pieces - judged, "scenes": scenes,
           "violating_pieces": viol}
    with open(os.path.join(path, "lookahead.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--target", default="ko")
    ap.add_argument("--no-hy", action="store_true")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--grace", type=float, default=2.0,
                    help="seconds after a segment's end (before the next segment) still judged against it; 0 = strict")
    a = ap.parse_args()
    run_dir(a.run_dir, a.target, a.no_hy, a.batch, a.grace)


if __name__ == "__main__":
    main()
