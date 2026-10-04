"""FLEURS -> SFT jsonl. One line = {audio: wav of SRC language, text: 'language TGT<asr_text>' + TGT transcript}.
ST pairs (src != tgt) + ASR pairs (src == tgt). Holdout pairs excluded from train and val.
usage: build_jsonl.py --split train --out train.jsonl --per_pair 300 [--one_rec]"""
import argparse, csv, json, os, random
FLEURS = os.path.expanduser("~/datasets/fleurs/data")
LANGS = {"en": ("en_us", "English"), "ko": ("ko_kr", "Korean"), "ja": ("ja_jp", "Japanese"), "zh": ("cmn_hans_cn", "Chinese")}
ap = argparse.ArgumentParser()
ap.add_argument("--split", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--per_pair", type=int, default=300); ap.add_argument("--one_rec", action="store_true")
ap.add_argument("--holdout", default="zh-ko,ja-ko,zh-ja"); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args(); random.seed(a.seed)
hold = set(a.holdout.split(",")) if a.holdout else set()
rows = {}
for k, (d, _) in LANGS.items():
    r = {}
    for row in csv.reader(open(f"{FLEURS}/{d}/{a.split}.tsv", encoding="utf-8"), delimiter="\t"):
        if len(row) < 6: continue
        r.setdefault(row[0], []).append(row)
    rows[k] = r
n = {}
with open(a.out, "w", encoding="utf-8") as f:
    for src in LANGS:
        for tgt in LANGS:
            pair = f"{src}-{tgt}"
            if pair in hold: continue
            ids = sorted(set(rows[src]) & set(rows[tgt]), key=int); random.shuffle(ids)
            c = 0
            for uid in ids[: a.per_pair]:
                recs = rows[src][uid][:1] if a.one_rec else rows[src][uid]
                text = rows[tgt][uid][0][2].strip()
                for rec in recs:
                    wav = f"{FLEURS}/{LANGS[src][0]}/audio/{a.split}/{rec[1]}"
                    if not os.path.exists(wav): continue
                    f.write(json.dumps({"audio": wav, "text": f"language {LANGS[tgt][1]}<asr_text>{text}", "pair": pair}, ensure_ascii=False) + "\n"); c += 1
            n[pair] = c
print(a.out, sum(n.values()), n)
