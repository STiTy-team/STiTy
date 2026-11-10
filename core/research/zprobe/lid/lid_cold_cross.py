"""Channel-shortcut check: train-corpus x eval-corpus matrix at w in {0.3, 0.5}.

If the probe learned language, off-diagonal (cross-corpus) accuracy stays high; if it learned the
recording channel (kspon mic vs DailyTalk studio vs FLEURS), off-diagonal collapses. Decides
whether round 2's 99.8% held-out number means anything.

usage: PYTHONPATH=Qwen3-ASR .venv/bin/python core/research/zprobe/lid/lid_cold_cross.py
"""
import json
import os
import random
import sys
import time

import librosa
import numpy as np
import torch

sys.path.insert(0, os.getcwd())
from qwen_asr import Qwen3ASRModel  # noqa: E402

MODEL = "models/Qwen3-ASR-1.7B-koen-seg-mix-merged"
WINDOWS = (0.3, 0.5)
PER_LANG = 500
EVAL_PER_LANG = 300
SR = 16000
rng = random.Random(0)

m = Qwen3ASRModel.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda",
                                  max_inference_batch_size=1)
m.model.eval()
thinker = m.model.thinker
dev = next(thinker.parameters()).device
_enc = {}
thinker.audio_tower.ln_post.register_forward_hook(lambda mod, i, o: _enc.update(v=o.detach()))
BASE = m._build_text_prompt(context="", force_language="Korean")


@torch.no_grad()
def emb(wav):
    inputs = m.processor(text=[BASE], audio=[wav], return_tensors="pt", padding=True)
    inputs = inputs.to(dev).to(torch.bfloat16)
    thinker(**inputs)
    return _enc["v"].reshape(-1, _enc["v"].shape[-1]).float().mean(0).cpu().numpy()


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


SRC = {"fleurs": fleurs_paths(), "conv": conv_paths()}
for v in SRC.values():
    for lang in v:
        rng.shuffle(v[lang])


def windows(source, lang, lo, hi, n, cap):
    out = []
    for p in SRC[source][lang][lo:hi]:
        if len(out) >= cap:
            break
        try:
            wav, _ = librosa.load(p, sr=SR, duration=n / SR + 3.0)
        except Exception:  # noqa: BLE001
            continue
        seg = onset_cut(wav, n)
        if seg is not None:
            out.append(emb(seg))
    return out


def fit(X, y):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    xt = torch.tensor((X - mu) / sd, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    w = torch.zeros(X.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=0.05)
    for _ in range(300):
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(xt @ w + b, yt) + 1e-3 * w.pow(2).sum()
        loss.backward()
        opt.step()
    return w.detach().numpy(), float(b.item()), mu, sd


t0 = time.time()
for w_sec in WINDOWS:
    n = int(w_sec * SR)
    tr = {s: {li: windows(s, lang, 0, PER_LANG * 2, n, PER_LANG)
              for li, lang in enumerate(("ko", "en"))} for s in SRC}
    ev = {s: {li: windows(s, lang, PER_LANG * 2, PER_LANG * 2 + EVAL_PER_LANG * 2, n, EVAL_PER_LANG)
              for li, lang in enumerate(("ko", "en"))} for s in SRC}
    for train_s in ("fleurs", "conv"):
        X = np.stack(tr[train_s][0] + tr[train_s][1])
        y = np.array([0] * len(tr[train_s][0]) + [1] * len(tr[train_s][1]))
        wm, bm, mu, sd = fit(X, y)
        line = []
        for eval_s in ("fleurs", "conv"):
            hit = tot = 0
            for li in (0, 1):
                for e in ev[eval_s][li]:
                    p = 1 / (1 + np.exp(-(float(np.dot((e - mu) / sd, wm)) + bm)))
                    hit += (p > 0.5) == bool(li)
                    tot += 1
            line.append(f"eval={eval_s} {hit / tot:.3f} (n={tot})")
        print(f"w={w_sec} train={train_s}: " + "  ".join(line) + f"  ({time.time() - t0:.0f}s)",
              flush=True)
print("LID_CROSS_DONE", flush=True)
