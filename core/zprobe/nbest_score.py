"""Score nbest.py output: oracle upper bound and reference-free rerankers.
Every unique candidate gets COMET-DA (with the FLEURS reference), CometKiwi (no reference) and chrF++. Reports:
  greedy / beam top-1      what the adapter outputs today
  oracle@k                 best candidate per sentence by COMET, over the first k beams, k samples, or everything
  oracle by chrF++         the same oracle chosen with a different metric, so the gain is not just COMET noise picked twice
  kiwi rerank              pick the candidate CometKiwi likes best (deployable: no reference)
  MBR                      pick the candidate with the highest mean COMET-DA against the other candidates (deployable)
Paired bootstrap 95% CI against greedy for every row.

usage: python core/zprobe/nbest_score.py --nbest N.jsonl --out report.md [--cache scores.jsonl]"""
import argparse, json, os, random

import sacrebleu


def boot(a, b, n=2000, seed=0):
    rng = random.Random(seed); d = [y - x for x, y in zip(a, b)]; m = len(d)
    s = sorted(sum(d[rng.randrange(m)] for _ in range(m)) / m for _ in range(n))
    return sum(d) / m, s[int(0.025 * n)], s[int(0.975 * n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nbest", required=True); ap.add_argument("--out", required=True); ap.add_argument("--cache", default="")
    ap.add_argument("--batch", type=int, default=64)
    a = ap.parse_args()
    from comet import download_model, load_from_checkpoint
    recs = [json.loads(l) for l in open(a.nbest, encoding="utf-8")]
    for r in recs:
        r["cands"] = list(dict.fromkeys([r["greedy"]] + r["beam"] + r["sample"]))
    cache = a.cache or os.path.splitext(a.nbest)[0] + "_scores.jsonl"
    S = {}
    if os.path.exists(cache):
        for l in open(cache, encoding="utf-8"):
            x = json.loads(l); S[(x["id"], x["mt"])] = x
    todo = [(r, c) for r in recs for c in r["cands"] if (r["id"], c) not in S]
    if todo:
        da = load_from_checkpoint(download_model("Unbabel/wmt22-comet-da"))
        kiwi = load_from_checkpoint(download_model("Unbabel/wmt22-cometkiwi-da"))
        o1 = da.predict([{"src": r["ref_src"], "mt": c, "ref": r["ref_tgt"]} for r, c in todo], batch_size=a.batch, gpus=1).scores
        o2 = kiwi.predict([{"src": r["ref_src"], "mt": c} for r, c in todo], batch_size=a.batch, gpus=1).scores
        with open(cache, "a", encoding="utf-8") as f:
            for (r, c), x, y in zip(todo, o1, o2):
                rec = {"id": r["id"], "mt": c, "comet": x, "kiwi": y,
                       "chrf": sacrebleu.sentence_chrf(c, [r["ref_tgt"]], word_order=2).score}
                S[(r["id"], c)] = rec; f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    mbr_cache = os.path.splitext(cache)[0] + "_mbr.jsonl"
    M = {}
    if os.path.exists(mbr_cache):
        for l in open(mbr_cache, encoding="utf-8"):
            x = json.loads(l); M[(x["id"], x["mt"])] = x["mbr"]
    pairs = [(r, h, p) for r in recs if any((r["id"], h) not in M for h in r["cands"])
             for h in r["cands"] for p in r["cands"] if h != p]
    if pairs:
        da = load_from_checkpoint(download_model("Unbabel/wmt22-comet-da"))
        sc = da.predict([{"src": r["ref_src"], "mt": h, "ref": p} for r, h, p in pairs], batch_size=a.batch, gpus=1).scores
        acc = {}
        for (r, h, p), s in zip(pairs, sc):
            acc.setdefault((r["id"], h), []).append(s)
        with open(mbr_cache, "a", encoding="utf-8") as f:
            for k, v in acc.items():
                M[k] = sum(v) / len(v); f.write(json.dumps({"id": k[0], "mt": k[1], "mbr": M[k]}, ensure_ascii=False) + "\n")

    def pick(r, pool, key):
        return max(pool, key=lambda c: key(r, c))

    def comet(r, c): return S[(r["id"], c)]["comet"]
    def chrf(r, c): return S[(r["id"], c)]["chrf"]
    def kiwi_(r, c): return S[(r["id"], c)]["kiwi"]
    def mbr(r, c): return M.get((r["id"], c), -1.0) if len(r["cands"]) > 1 else 0.0

    rows = {"greedy": lambda r: r["greedy"], "beam top-1": lambda r: r["beam"][0]}
    for k in (2, 4, 8):
        rows[f"oracle beam@{k}"] = lambda r, k=k: pick(r, r["beam"][:k], comet)
    for k in (2, 4, 8):
        rows[f"oracle sample@{k}"] = lambda r, k=k: pick(r, r["sample"][:k], comet)
    rows["oracle all"] = lambda r: pick(r, r["cands"], comet)
    rows["oracle all by chrF++"] = lambda r: pick(r, r["cands"], chrf)
    rows["kiwi rerank (all)"] = lambda r: pick(r, r["cands"], kiwi_)
    rows["MBR (all)"] = lambda r: pick(r, r["cands"], mbr)
    rows["kiwi rerank (beam 8)"] = lambda r: pick(r, r["beam"], kiwi_)
    rows["MBR (beam 8)"] = lambda r: pick(r, [c for c in r["beam"]], mbr)

    g = [comet(r, r["greedy"]) for r in recs]; gc = [chrf(r, r["greedy"]) for r in recs]
    n_c = sum(len(r["cands"]) for r in recs) / len(recs)
    lines = [f"# n-best upper bound ({os.path.basename(a.nbest)}, N={len(recs)}, unique candidates / sentence {n_c:.1f})", "",
             "| selection | COMET | Δ vs greedy [95%] | chrF++ | Δ chrF++ |", "|---|---|---|---|---|"]
    for name, f in rows.items():
        sel = [f(r) for r in recs]
        c = [comet(r, s) for r, s in zip(recs, sel)]; h = [chrf(r, s) for r, s in zip(recs, sel)]
        d, lo, hi = boot(g, c); dc = (sum(h) - sum(gc)) / len(recs)
        lines.append(f"| {name} | {sum(c) / len(c):.3f} | {d:+.3f} [{lo:+.3f}, {hi:+.3f}] | {sum(h) / len(h):.1f} | {dc:+.1f} |")
    spread = [max(comet(r, c) for c in r["cands"]) - min(comet(r, c) for c in r["cands"]) for r in recs]
    worst = sorted(range(len(recs)), key=lambda i: g[i])[: len(recs) // 10]
    gain_w = sum(comet(recs[i], pick(recs[i], recs[i]["cands"], comet)) - g[i] for i in worst) / len(worst)
    lines += ["", f"- per-sentence COMET spread (max − min over candidates): mean {sum(spread) / len(spread):.3f}",
              f"- oracle gain on the worst 10% greedy sentences: {gain_w:+.3f}"]
    open(a.out, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
