"""Cheap O/X cut decision: a linear read-out of Qwen3-ASR hidden states at each word end, trained on per-cut COMET labels
(cut_search.py rows + comet_eval.py scores), instead of asking an LLM or relying on the SEG model's own <SEG>.

features : the SEG ASR model (PROBE_MODEL, no adapter) reads the gold English transcript (teacher forced, full audio);
           at the last token of word k we keep layers --layers. -> npz {key "id:k": [L, D]}
fit      : label of cut k = COMET(cut at k) − COMET(no cut). Ridge regression on the loss and logistic O/X (O = loss ≥ --ok)
           are trained on --train sentences; on --test sentences each model picks its best-scored cut, which is compared
           with the SEG model's cut, a random cut (mean over cuts), the oracle cut and the worst cut.

usage: PROBE_MODEL=<seg model> PYTHONPATH=Qwen3-ASR python core/zprobe/cut_probe.py features --rows dev.jsonl --split dev --out F.npz
       python core/zprobe/cut_probe.py fit --train dev:F_dev.npz:scores/dev.jsonl test:F_test.npz:scores/test.jsonl --test_ids ..."""
import argparse, collections, csv, json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


def features(a):
    import librosa, torch
    import probe_z as pz
    m = pz.load_model()
    thinker = m.model.thinker
    tok = m.processor.tokenizer
    dev = next(thinker.parameters()).device
    taps = pz.Taps(thinker)
    layers = [int(x) for x in a.layers.split(",")]
    rows_tsv = {r[0]: r for r in csv.reader(open(pz.FLEURS / "en_us" / f"{a.split}.tsv", encoding="utf-8"), delimiter="\t",
                                            quoting=csv.QUOTE_NONE) if len(r) >= 3}
    asr_tok = tok.convert_tokens_to_ids("<asr_text>")
    want = collections.defaultdict(set)
    for r in map(json.loads, open(a.rows, encoding="utf-8")):
        if r["k"]:
            want[r["id"]].add(r["k"])
    feats = {}
    for n, (uid, ks) in enumerate(want.items()):
        gold = rows_tsv[uid][2].strip(); words = gold.split()
        wav, _ = librosa.load(str(pz.FLEURS / "en_us" / "audio" / a.split / rows_tsv[uid][1]), sr=16000)
        prompt = m._build_text_prompt(context="", force_language="English") + gold
        inputs = m.processor(text=[prompt], audio=[wav], return_tensors="pt", padding=True).to(dev).to(torch.bfloat16)
        with torch.no_grad():
            thinker(**inputs)
        ids = inputs["input_ids"][0]
        start = int((ids == asr_tok).nonzero()[-1]) + 1
        out_ids = ids[start:].tolist()
        # character end of each word in the transcript, then the token that completes it
        ends, c = [], 0
        for w in words:
            c = gold.index(w, c) + len(w); ends.append(c)
        tok_end, j = {}, 0
        for t in range(len(out_ids)):
            L = len(tok.decode(out_ids[: t + 1]))
            while j < len(ends) and L >= ends[j]:
                tok_end[j + 1] = start + t; j += 1
        H = torch.stack([taps.h[l] for l in layers], 0)[:, 0]  # [layers, T, D]
        for k in ks:
            if k in tok_end:
                feats[f"{uid}:{k}"] = H[:, tok_end[k]].float().cpu().numpy()
        if (n + 1) % 50 == 0:
            print(f"{n + 1}/{len(want)}", flush=True)
    np.savez_compressed(a.out, keys=np.array(list(feats)), X=np.stack(list(feats.values())), layers=np.array(layers))
    print(f"saved {len(feats)} cut features, layers {layers}")


def load(spec):
    name, fpath, rpath, spath = spec.split(":")
    z = np.load(fpath)
    F = {k: x for k, x in zip(z["keys"], z["X"])}
    rows = [json.loads(l) for l in open(rpath, encoding="utf-8")]
    sc = [json.loads(l)["comet"] for l in open(spath, encoding="utf-8")]
    whole = {r["id"]: s for r, s in zip(rows, sc) if r["k"] == 0}
    out = []
    for r, s in zip(rows, sc):
        if r["k"] and r["key"] in F and r["id"] in whole:
            out.append({"set": name, "id": r["id"], "k": r["k"], "is_seg": r["is_seg"], "loss": s - whole[r["id"]], "x": F[r["key"]]})
    return out, list(z["layers"])


def fit(a):
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.preprocessing import StandardScaler
    allr, layers = [], None
    for spec in a.data:
        rs, layers = load(spec); allr += rs
    test_ids = {l.strip() for l in open(a.test_ids)} if a.test_ids else set()
    tr = [r for r in allr if r["id"] not in test_ids]; te = [r for r in allr if r["id"] in test_ids]
    print(f"train cuts {len(tr)} ({len({r['id'] for r in tr})} sentences), test cuts {len(te)} ({len({r['id'] for r in te})} sentences)")
    by = collections.defaultdict(list)
    for r in te:
        by[r["id"]].append(r)
    f = lambda v: sum(v) / len(v)
    base = {"random cut": f([f([r["loss"] for r in rs]) for rs in by.values()]),
            "oracle cut": f([max(r["loss"] for r in rs) for rs in by.values()]),
            "worst cut": f([min(r["loss"] for r in rs) for rs in by.values()])}
    seg = [next((r["loss"] for r in rs if r["is_seg"]), None) for rs in by.values()]
    seg = [x for x in seg if x is not None]
    print(f"\n| cut chooser | COMET loss vs no cut |\n|---|---|")
    print(f"| SEG model (n={len(seg)}) | {f(seg):+.3f} |")
    for k, v in base.items():
        print(f"| {k} | {v:+.3f} |")
    for li, L in enumerate(layers):
        Xtr = np.stack([r["x"][li] for r in tr]); Xte = np.stack([r["x"][li] for r in te])
        sc = StandardScaler().fit(Xtr); Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
        ytr = np.array([r["loss"] for r in tr])
        for name, model, score in [
            ("ridge", Ridge(alpha=a.alpha).fit(Xtr, ytr), lambda mdl, X: mdl.predict(X)),
            ("O/X", LogisticRegression(C=a.C, max_iter=2000).fit(Xtr, ytr >= a.ok), lambda mdl, X: mdl.predict_proba(X)[:, 1])]:
            pred = score(model, Xte)
            for r, p in zip(te, pred):
                r["p"] = p
            pick = [max(rs, key=lambda r: r["p"])["loss"] for rs in by.values()]
            print(f"| L{L} {name} | {f(pick):+.3f} |")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["features", "fit"])
    ap.add_argument("--rows"); ap.add_argument("--split"); ap.add_argument("--out"); ap.add_argument("--layers", default="9,15,21,24")
    ap.add_argument("--data", nargs="*", default=[], help="name:features.npz:rows.jsonl:scores.jsonl")
    ap.add_argument("--test_ids"); ap.add_argument("--alpha", type=float, default=100.0); ap.add_argument("--C", type=float, default=0.01)
    ap.add_argument("--ok", type=float, default=-0.01)
    a = ap.parse_args()
    features(a) if a.mode == "features" else fit(a)
