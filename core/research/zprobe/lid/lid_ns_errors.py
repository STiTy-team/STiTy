"""Dissect the northstar cold-start residual: which content utterances does the fleurs head get
wrong at w=0.5/1.0s, and why - overlap with a neighbouring utterance in the judged window,
preceding-utterance bleed, low energy, or none of those (genuinely hard).

usage: PYTHONPATH=Qwen3-ASR .venv/bin/python core/research/zprobe/lid/lid_ns_errors.py
"""
import json
import os
import random
import re
import sys

import librosa
import numpy as np
import torch

sys.path.insert(0, os.getcwd())
from qwen_asr import Qwen3ASRModel  # noqa: E402

MODEL = "models/Qwen3-ASR-1.7B-koen-seg-mix-merged"
NS = "/home/mobility/stity-datasets/northstar-260928"
SR = 16000
PER_LANG = 600
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
def emb(wav):
    inputs = m.processor(text=[BASE], audio=[np.asarray(wav, dtype=np.float32)],
                         return_tensors="pt", padding=True)
    inputs = inputs.to(dev).to(torch.bfloat16)
    thinker(**inputs)
    return _enc["v"].reshape(-1, _enc["v"].shape[-1]).float().mean(0).cpu().numpy()


def onset_idx(wav):
    loud = np.flatnonzero(np.abs(wav) > 0.02)
    return int(loud[0]) if loud.size else None


def fleurs_paths():
    out = {"ko": [], "en": []}
    for l in open("models/zprobe/data/prefix12/word_times.jsonl", encoding="utf-8"):
        r = json.loads(l)
        if r["lang"] in out:
            out[r["lang"]].append(r["audio"].split("#")[0])
    out = {k: sorted(set(v)) for k, v in out.items()}
    for v in out.values():
        rng.shuffle(v)
    return out


FL = fleurs_paths()


def collect(plist, cap, n):
    out = []
    for p in plist:
        if len(out) >= cap:
            break
        try:
            wav, _ = librosa.load(p, sr=SR, duration=n / SR + 3.0)
        except Exception:  # noqa: BLE001
            continue
        oi = onset_idx(wav)
        if oi is None or wav.shape[0] - oi < n:
            continue
        out.append(emb(wav[oi:oi + n]))
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


rows = []
for line in open(f"{NS}/manifest.jsonl", encoding="utf-8"):
    it = json.loads(line)
    scene = it["audio"].split("/")[-1].removesuffix(".wav")
    tr = (it.get("reference") or {}).get("transcript", "")
    rows.append({"scene": scene, "off": it["offset"], "dur": it["duration"],
                 "lang": it["src_lang"], "spk": it.get("speaker"), "tr": tr,
                 "amb": bool(AMB.match(tr))})
scene_wav = {s: librosa.load(f"{NS}/audio/{s}.wav", sr=SR)[0]
             for s in sorted({r["scene"] for r in rows})}

for w_sec in (0.5, 1.0):
    n = int(w_sec * SR)
    tr_fl = {li: collect(FL[lang], PER_LANG, n) for li, lang in enumerate(("ko", "en"))}
    wm, bm, mu, sd = fit(np.stack(tr_fl[0] + tr_fl[1]),
                         np.array([0] * len(tr_fl[0]) + [1] * len(tr_fl[1])))
    print(f"== w={w_sec}s 오판 (내용 발화) ==", flush=True)
    for r in rows:
        if r["lang"] not in ("ko", "en") or r["dur"] < w_sec or r["amb"]:
            continue
        chunk = scene_wav[r["scene"]][int(r["off"] * SR):int((r["off"] + r["dur"]) * SR)]
        oi = onset_idx(chunk)
        if oi is None or chunk.shape[0] - oi < n:
            continue
        seg = chunk[oi:oi + n]
        p = 1 / (1 + np.exp(-(float(np.dot((emb(seg) - mu) / sd, wm)) + bm)))
        pred = "en" if p > 0.5 else "ko"
        if pred == r["lang"]:
            continue
        t0 = r["off"] + oi / SR
        t1 = t0 + w_sec
        overlaps = [f"{o['spk']}({o['lang']})" for o in rows
                    if o is not r and o["scene"] == r["scene"]
                    and o["off"] < t1 and o["off"] + o["dur"] > t0]
        rms = float(np.sqrt(np.mean(seg ** 2)))
        print(f"{r['scene']}@{r['off']:.1f} truth={r['lang']} conf={max(p, 1 - p):.2f} "
              f"spk={r['spk']} rms={rms:.3f} 겹침={overlaps or '없음'}", flush=True)
        print(f"   전사: {r['tr'][:60]}", flush=True)
print("NS_ERR_DONE", flush=True)
