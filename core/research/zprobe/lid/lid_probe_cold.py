"""Cold-start LID: language accuracy on the FIRST w seconds of an utterance, w in {0.3,0.5,0.7,1.0}.

The trailing-window measurement (lid_probe.py) judges mid-stream; this one judges the post-reset /
turn-taking moment: windows start exactly at northstar manifest utterance offsets, truth is that
utterance's language, and a separate linear head is trained per window length on the first w seconds
of FLEURS train recordings. FusedLid is scored on the identical windows.

usage: PYTHONPATH=Qwen3-ASR .venv/bin/python core/research/zprobe/lid/lid_probe_cold.py
"""
import asyncio
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
WT = "models/zprobe/data/prefix12/word_times.jsonl"
NS = "/home/mobility/stity-datasets/northstar-260928"
WINDOWS = (0.3, 0.5, 0.7, 1.0)
PER_LANG = 600
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
def embed(wav):
    inputs = m.processor(text=[BASE], audio=[wav], return_tensors="pt", padding=True)
    inputs = inputs.to(dev).to(torch.bfloat16)
    thinker(**inputs)
    e = _enc["v"]
    return e.reshape(-1, e.shape[-1]).float().mean(0).cpu().numpy()


paths = {"ko": [], "en": []}
for l in open(WT, encoding="utf-8"):
    r = json.loads(l)
    if r["lang"] in paths:
        paths[r["lang"]].append(r["audio"].split("#")[0])
for k in paths:
    paths[k] = sorted(set(paths[k]))
    rng.shuffle(paths[k])

utts = []
for line in open(f"{NS}/manifest.jsonl", encoding="utf-8"):
    it = json.loads(line)
    scene = it["audio"].split("/")[-1].removesuffix(".wav")
    head = (it.get("reference") or {}).get("transcript", "")[:24]
    utts.append((scene, it["offset"], it["duration"], it["src_lang"], head))
scene_wav = {s: librosa.load(f"{NS}/audio/{s}.wav", sr=SR)[0]
             for s in sorted({u[0] for u in utts})}

from core.utils import lid as lidmod  # noqa: E402
fused = asyncio.run(lidmod.FusedLid.build("models/zprobe/data/lid_phone_profiles.json"))

report = {}
t0 = time.time()
for w in WINDOWS:
    n = int(w * SR)
    X, y = [], []
    for li, lang in enumerate(("ko", "en")):
        done = 0
        for p in paths[lang]:
            if done >= PER_LANG:
                break
            try:
                wav, _ = librosa.load(p, sr=SR, duration=w + 3.0)
            except Exception:  # noqa: BLE001
                continue
            loud = np.flatnonzero(np.abs(wav) > 0.02)
            if loud.size == 0 or wav.shape[0] - loud[0] < n:
                continue
            X.append(embed(wav[loud[0]:loud[0] + n]))
            y.append(li)
            done += 1
    X = np.stack(X)
    yv = np.array(y)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    xt = torch.tensor((X - mu) / sd, dtype=torch.float32)
    yt = torch.tensor(yv, dtype=torch.float32)
    wgt = torch.zeros(X.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([wgt, b], lr=0.05)
    for _ in range(300):
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(xt @ wgt + b, yt) \
            + 1e-3 * wgt.pow(2).sum()
        loss.backward()
        opt.step()
    wn, bn = wgt.detach().numpy(), float(b.item())

    probe_hit = probe_n = fused_hit = fused_n = fused_abstain = 0
    confs = []
    fails = []
    for scene, off, dur, lang, head in utts:
        if lang not in ("ko", "en") or dur < w:
            continue
        chunk = scene_wav[scene][int(off * SR):int((off + dur) * SR)]
        loud = np.flatnonzero(np.abs(chunk) > 0.02)
        if loud.size == 0 or chunk.shape[0] - loud[0] < n:
            continue
        seg = chunk[loud[0]:loud[0] + n]
        e = (embed(seg) - mu) / sd
        p = 1 / (1 + np.exp(-(float(np.dot(e, wn)) + bn)))
        pv = "en" if p > 0.5 else "ko"
        confs.append(max(p, 1 - p))
        probe_n += 1
        probe_hit += pv == lang
        if pv != lang:
            fails.append(f"{scene}@{off:.1f} truth={lang} conf={max(p,1-p):.2f} | {head}")
        fv, fa = fused.verdict(seg)
        if fv in ("ko", "en"):
            fused_n += 1
            fused_hit += fv == lang
            fused_abstain += not fa
        else:
            fused_abstain += 1
    report[w] = {"probe_acc": round(probe_hit / max(1, probe_n), 3), "n": probe_n,
                 "probe_conf_mean": round(float(np.mean(confs)), 3),
                 "fused_acc": round(fused_hit / max(1, fused_n), 3), "fused_n": fused_n,
                 "fused_abstain": fused_abstain}
    print(f"w={w}s {report[w]} ({time.time() - t0:.0f}s)", flush=True)
    if w in (0.3, 0.5):
        for f in fails:
            print("  FAIL", f, flush=True)

json.dump(report, open("models/zprobe/data/prefix12/lid_probe_cold.json", "w"))
print("LID_COLD_DONE", flush=True)
