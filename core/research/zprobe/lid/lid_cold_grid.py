"""Cold-start LID round 1: training distribution x pooling grid, scored on northstar utterance
onsets, overall AND content-only (interjection/laughter-led utterances scored separately - they are
renderable in either language, so they are not errors that matter).

train sets   fleurs: read-speech onsets (the baseline head's diet)
             conv:   kspon (ko) + DailyTalk (en) conversational onsets - matches the eval register
             both:   union
poolings     mean:   mean over encoder frames (baseline)
             vote:   per-frame logistic head, soft-vote mean prob; dispersion of frame probs is the
                     confidence (the mean-pool head was overconfident: 0.95 conf at 73% acc)
             tail:   mean concat mean-of-last-0.3s (recency)
windows      0.3 / 0.5 / 0.7 / 1.0 s from the onset (first loud sample)

usage: PYTHONPATH=Qwen3-ASR .venv/bin/python core/research/zprobe/lid/lid_cold_grid.py
"""
import json
import os
import random
import re
import sys
import time

import librosa
import numpy as np
import torch

sys.path.insert(0, os.getcwd())
from qwen_asr import Qwen3ASRModel  # noqa: E402

MODEL = "models/Qwen3-ASR-1.7B-koen-seg-mix-merged"
NS = "/home/mobility/stity-datasets/northstar-260928"
WINDOWS = (0.3, 0.5, 0.7, 1.0)
PER_LANG = 600
SR = 16000
rng = random.Random(0)

AMB = re.compile(r"^[\s\W]*(하+|ㅎ+|아{1,3}[\s,.…!?~]|어[\s,.…]|음+|네[\s,.…!?]|예[\s,.…]|자[\s,]|자자"
                 r"|응+|에이|아이[\s,]|에휴|흠|어유|이?모님|위하여|완전[\s!]|그래요|그럼[\s,.!]"
                 r"|um+|uh+|oh[\s,.!]|haha|h[ma]+[\s,.]|yeah|okay|ok[\s,.]|wow|mm+)", re.I)

m = Qwen3ASRModel.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda",
                                  max_inference_batch_size=1)
m.model.eval()
thinker = m.model.thinker
dev = next(thinker.parameters()).device
_enc = {}
thinker.audio_tower.ln_post.register_forward_hook(lambda mod, i, o: _enc.update(v=o.detach()))
BASE = m._build_text_prompt(context="", force_language="Korean")


@torch.no_grad()
def frames_of(wav):
    inputs = m.processor(text=[BASE], audio=[wav], return_tensors="pt", padding=True)
    inputs = inputs.to(dev).to(torch.bfloat16)
    thinker(**inputs)
    return _enc["v"].reshape(-1, _enc["v"].shape[-1]).float().cpu().numpy()


def onset_cut(wav, n):
    loud = np.flatnonzero(np.abs(wav) > 0.02)
    if loud.size == 0 or wav.shape[0] - loud[0] < n:
        return None
    return wav[loud[0]:loud[0] + n]


def fleurs_paths():
    out = {"ko": [], "en": []}
    for l in open("models/zprobe/data/prefix12/word_times.jsonl", encoding="utf-8"):
        r = json.loads(l)
        if r["lang"] in out:
            out[r["lang"]].append(r["audio"].split("#")[0])
    return {k: sorted(set(v)) for k, v in out.items()}


def conv_paths():
    out = {"ko": [], "en": []}
    for name, lang in (("kspon_manifest", "ko"), ("dailytalk_manifest", "en")):
        for l in open(f"models/zprobe/data/conv/{name}.jsonl", encoding="utf-8"):
            out[lang].append(json.loads(l)["audio"].split("#")[0])
    return {k: sorted(set(v)) for k, v in out.items()}


SOURCES = {"fleurs": fleurs_paths(), "conv": conv_paths()}
for k, v in SOURCES.items():
    for lang in v:
        rng.shuffle(v[lang])

utts = []
for line in open(f"{NS}/manifest.jsonl", encoding="utf-8"):
    it = json.loads(line)
    scene = it["audio"].split("/")[-1].removesuffix(".wav")
    head = (it.get("reference") or {}).get("transcript", "")
    utts.append((scene, it["offset"], it["duration"], it["src_lang"], bool(AMB.match(head))))
scene_wav = {s: librosa.load(f"{NS}/audio/{s}.wav", sr=SR)[0]
             for s in sorted({u[0] for u in utts})}
print(f"eval utts: {len(utts)}, 간투사 첫머리 {sum(u[4] for u in utts)}", flush=True)

t0 = time.time()
train_cache = {}


def train_frames(source, w):
    key = (source, w)
    if key in train_cache:
        return train_cache[key]
    n = int(w * SR)
    data = {}
    for li, lang in enumerate(("ko", "en")):
        fs = []
        done = 0
        for p in SOURCES[source][lang]:
            if done >= PER_LANG:
                break
            try:
                wav, _ = librosa.load(p, sr=SR, duration=w + 3.0)
            except Exception:  # noqa: BLE001
                continue
            seg = onset_cut(wav, n)
            if seg is None:
                continue
            fs.append(frames_of(seg))
            done += 1
        data[li] = fs
    train_cache[key] = data
    return data


def fit_head(X, y, l2=1e-3, iters=300):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    xt = torch.tensor((X - mu) / sd, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    w = torch.zeros(X.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=0.05)
    for _ in range(iters):
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(xt @ w + b, yt) + l2 * w.pow(2).sum()
        loss.backward()
        opt.step()
    return w.detach().numpy(), float(b.item()), mu, sd


def tail_feat(fr, frac=0.4):
    k = max(1, int(len(fr) * frac))
    return np.concatenate([fr.mean(0), fr[-k:].mean(0)])


eval_cache = {}


def eval_frames(w):
    if w in eval_cache:
        return eval_cache[w]
    n = int(w * SR)
    rows = []
    for scene, off, dur, lang, amb in utts:
        if lang not in ("ko", "en") or dur < w:
            continue
        seg = onset_cut(scene_wav[scene][int(off * SR):int((off + dur) * SR)], n)
        if seg is None:
            continue
        rows.append((frames_of(seg), lang, amb))
    eval_cache[w] = rows
    return rows


def score(preds):
    allv = [(p, l) for p, l, amb in preds]
    content = [(p, l) for p, l, amb in preds if not amb]
    return (sum(p == l for p, l in allv) / max(1, len(allv)),
            sum(p == l for p, l in content) / max(1, len(content)), len(content))


results = {}
for source in ("fleurs", "conv", "both"):
    for w in WINDOWS:
        if source == "both":
            d1, d2 = train_frames("fleurs", w), train_frames("conv", w)
            data = {li: d1[li] + d2[li] for li in (0, 1)}
        else:
            data = train_frames(source, w)
        evr = eval_frames(w)

        Xm = np.stack([fr.mean(0) for li in (0, 1) for fr in data[li]])
        ym = np.array([li for li in (0, 1) for _ in data[li]])
        wm, bm, mum, sdm = fit_head(Xm, ym)
        preds = []
        for fr, lang, amb in evr:
            p = 1 / (1 + np.exp(-(float(np.dot((fr.mean(0) - mum) / sdm, wm)) + bm)))
            preds.append(("en" if p > 0.5 else "ko", lang, amb))
        results[(source, w, "mean")] = score(preds)

        Xf = np.stack([f for li in (0, 1) for fr in data[li] for f in fr[::2]])
        yf = np.array([li for li in (0, 1) for fr in data[li] for _ in fr[::2]])
        wf, bf, muf, sdf = fit_head(Xf, yf, iters=200)
        preds = []
        for fr, lang, amb in evr:
            ps = 1 / (1 + np.exp(-(((fr - muf) / sdf) @ wf + bf)))
            p = float(ps.mean())
            preds.append(("en" if p > 0.5 else "ko", lang, amb))
        results[(source, w, "vote")] = score(preds)

        Xt = np.stack([tail_feat(fr) for li in (0, 1) for fr in data[li]])
        wt, bt, mut, sdt = fit_head(Xt, ym)
        preds = []
        for fr, lang, amb in evr:
            p = 1 / (1 + np.exp(-(float(np.dot((tail_feat(fr) - mut) / sdt, wt)) + bt)))
            preds.append(("en" if p > 0.5 else "ko", lang, amb))
        results[(source, w, "tail")] = score(preds)

        print(f"{source:6s} w={w}: " + "  ".join(
            f"{pool} 전체 {results[(source, w, pool)][0]:.3f}/내용 {results[(source, w, pool)][1]:.3f}"
            for pool in ("mean", "vote", "tail")) + f"  ({time.time() - t0:.0f}s)", flush=True)

json.dump({f"{s}|{w}|{p}": v for (s, w, p), v in results.items()},
          open("models/zprobe/data/prefix12/lid_cold_grid.json", "w"))
print("LID_GRID_DONE", flush=True)
