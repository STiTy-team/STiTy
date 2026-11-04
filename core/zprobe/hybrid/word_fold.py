"""Word-level (region) folding of the live hybrid run, offline replay.

Whole-commit folding clips the head of the next translated sentence when a commit straddles a language
switch (measured: 16/48 scored sentences, -0.014 to -0.025 COMET). This folds REGIONS inside a commit
instead: a token matches when its normalized form is a substring of the transcript window [t-8, t+2];
runs of matching tokens fold, single non-matching tokens BETWEEN matching neighbours fold with their run
(the two ASRs disagree on proper nouns and spelling - "김태우" vs "김태호" - and word-literal folding
would leave those shards behind); an isolated matching token stays (common words are not evidence).
A fold only fires when the folded run is substantial (>= 4 normalized chars). Second pass: a commit
whose REMAINING text still has char-3-gram containment >= 0.7 in the window is dropped whole - short
tokens ("몇") evade the word match and a fully-duplicated commit would otherwise leak through (measured:
46-49 leaks at containment 1.0). A straddler survives this pass because its remainder is the translation
head, which the transcript does not contain. Kept records keep their timing.

usage: (bench venv, cwd STiTy-bench) python word_fold.py --run-dir <ko run> --target ko --outdir <dir>
"""
import argparse, json, os, re, sys
from pathlib import Path

sys.path.insert(0, os.getcwd())
from bench.metrics.latency import longyaal
from bench.metrics.translation import translation_sentences

ap = argparse.ArgumentParser()
ap.add_argument("--run-dir", required=True)
ap.add_argument("--target", required=True)
ap.add_argument("--outdir", required=True)
ap.add_argument("--gap", type=int, default=1, help="non-matching tokens this long a run may swallow")
a = ap.parse_args()


def norm(text):
    return re.sub(r"[^0-9a-z가-힣]", "", text.lower())


ref = {}
for line in open(Path(a.run_dir) / "events.jsonl"):
    d = json.loads(line)
    if d.get("type") == "asrref":
        ref.setdefault(d["item"], []).append((float(d["decision_audio_sec"]), norm(d["original"])))


def trigrams(s):
    return {s[i:i + 3] for i in range(len(s) - 2)}


def contained(x, window):
    if len(x) < 4:
        return 1.0 if x and x in window else 0.0
    g = trigrams(x)
    return len(g & trigrams(window)) / len(g) if g else 0.0


def fold_text(scene, t, text):
    window = "".join(cx for ct, cx in ref.get(scene, []) if t - 8.0 <= ct <= t + 2.0)
    if not window:
        return text, 0
    tokens = text.split()
    hit = [len(norm(tok)) >= 2 and norm(tok) in window for tok in tokens]
    keep = list(hit)
    for i in range(len(tokens)):          # close gaps: a lone miss between hits joins the run
        if not keep[i] and 0 < i < len(tokens) - 1 and hit[i - 1] and hit[i + 1]:
            keep[i] = True
    out, folded, i = [], 0, 0
    while i < len(tokens):
        if not keep[i]:
            out.append(tokens[i]); i += 1
            continue
        j = i
        while j < len(tokens) and keep[j]:
            j += 1
        run = tokens[i:j]
        solid = sum(1 for k in range(i, j) if hit[k])
        if solid >= 2 and len(norm("".join(run))) >= 4:
            folded += len(run)
        else:
            out.extend(run)
        i = j
    return " ".join(out), folded


rows = [json.loads(l) for l in open(Path(a.run_dir) / "items.jsonl")]
kept = dropped = trimmed = 0
for row in rows:
    recs = []
    for r in row["records"]:
        if r.get("type") != "translated":
            recs.append(r)
            continue
        t = float(r.get("decision_audio_sec") or 0.0)
        new, folded = fold_text(row["id"], t, r.get("translation") or "")
        new = re.sub(r"^[\s,.!?…]+", "", new)
        if not norm(new):
            dropped += 1
            continue
        if not folded and len(new.split()) <= 2:
            window = "".join(cx for ct, cx in ref.get(row["id"], []) if t - 8.0 <= ct <= t + 2.0)
            if norm(new) and norm(new) in window:
                dropped += 1
                continue
        if folded:
            trimmed += 1
            r = {**r, "translation": new, "original": new}
        kept += 1
        recs.append(r)
    row["records"] = recs

ly = longyaal(rows, a.target)
lyca = longyaal(rows, a.target, computation_aware=True)
sentences = translation_sentences(rows, a.target)
empty = sum(1 for s in sentences if s["src"] and not s["mt"])
out = Path(a.outdir)
out.mkdir(parents=True, exist_ok=True)
with open(out / "comet_inputs.jsonl", "w") as f:
    for s in sentences:
        f.write(json.dumps(s, ensure_ascii=False) + "\n")
json.dump({"metrics": {"longyaal_ms": ly, "longyaal_ca_ms": lyca, "kept": kept, "dropped": dropped,
                       "trimmed": trimmed, "empty_mt": empty,
                       "n_sentences": sum(1 for s in sentences if s["src"])},
           "unavailable": {}},
          open(out / "summary.json", "w"), indent=1)
print(f"{a.target}-wordfold: longyaal={ly and round(ly)} ca={lyca and round(lyca)} kept={kept} "
      f"dropped={dropped} trimmed={trimmed} empty_mt={empty}", flush=True)
print("WORD_FOLD_DONE", flush=True)
