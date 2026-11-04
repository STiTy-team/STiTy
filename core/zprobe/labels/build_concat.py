"""Concatenated-pair SFT examples: audio = utt_a + 0.3 s + utt_b (source language, FLEURS train),
text = 'language TGT<asr_text>' + t_a + ' <SEG> ' + t_b. Writes wavs to --wavdir. For ASR pairs the
texts come from --seg_jsonl (SEG-model pseudo labels) if given, else plain transcripts."""
import argparse, csv, json, os, random
import numpy as np, soundfile as sf, librosa
FLEURS = os.path.expanduser("~/datasets/fleurs/data")
LANGS = {"en": ("en_us", "English"), "ko": ("ko_kr", "Korean"), "ja": ("ja_jp", "Japanese"), "zh": ("cmn_hans_cn", "Chinese")}
ap = argparse.ArgumentParser()
ap.add_argument("--split", default="train"); ap.add_argument("--out", required=True); ap.add_argument("--wavdir", required=True)
ap.add_argument("--pairs", default="en-ko,en-ja,ko-en,en-en,ko-ko"); ap.add_argument("--n", type=int, default=150)
ap.add_argument("--seg_jsonl", default=""); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args(); random.seed(a.seed); os.makedirs(a.wavdir, exist_ok=True)
rows = {}
for k, (d, _) in LANGS.items():
    r = {}
    for row in csv.reader(open(f"{FLEURS}/{d}/{a.split}.tsv", encoding="utf-8"), delimiter="\t"):
        if len(row) >= 6: r.setdefault(row[0], row)
    rows[k] = r
pseudo = {}
if a.seg_jsonl:
    for l in open(a.seg_jsonl, encoding="utf-8"):
        r = json.loads(l)
        if r.get("pseudo"): pseudo[os.path.basename(r["audio"])] = r["text"].split("<asr_text>", 1)[1]
gap = np.zeros(int(16000 * 0.3), dtype=np.float32)
out = open(a.out, "w", encoding="utf-8"); cnt = {}
for pair in a.pairs.split(","):
    src, tgt = pair.split("-")
    ids = sorted(set(rows[src]) & set(rows[tgt]), key=int); random.shuffle(ids)
    c = 0
    for i in range(0, len(ids) - 1, 2):
        if c >= a.n: break
        ua, ub = ids[i], ids[i + 1]
        ws = []
        for u in (ua, ub):
            p = f"{FLEURS}/{LANGS[src][0]}/audio/{a.split}/{rows[src][u][1]}"
            if not os.path.exists(p): break
            ws.append(librosa.load(p, sr=16000)[0])
        if len(ws) < 2: continue
        wav = np.concatenate([ws[0], gap, ws[1]])
        wp = os.path.join(a.wavdir, f"{pair}_{ua}_{ub}.wav"); sf.write(wp, wav, 16000)
        def t(u):
            if src == tgt and pseudo:
                return pseudo.get(rows[src][u][1], rows[tgt][u][2].strip())
            return rows[tgt][u][2].strip()
        ta, tb = t(ua), t(ub)
        if src == tgt and pseudo and not ta.rstrip().endswith("<SEG>"): ta = ta + " <SEG>"
        text = ta + " <SEG> " + tb if not ta.rstrip().endswith("<SEG>") else ta + " " + tb
        out.write(json.dumps({"audio": wp, "text": f"language {LANGS[tgt][1]}<asr_text>{text}", "pair": pair, "concat": True}, ensure_ascii=False) + "\n"); c += 1
    cnt[pair] = c
print(a.out, sum(cnt.values()), cnt)
