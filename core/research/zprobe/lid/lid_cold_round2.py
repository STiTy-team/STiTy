"""Cold-start LID round 2: a held-out conversational onset eval big enough to rank configs
(round 1's northstar content-n was 56 - one utterance moved 1.8pp), plus the phone-CTC wing of
FusedLid standalone and fused with the probe.

train: first PER_LANG kspon/DailyTalk onsets (same as round 1 'conv').
eval:  the NEXT EVAL_PER_LANG utterances of each corpus (path-disjoint from training).
judges: probe mean / probe vote / phone-CTC profile / probe+phone fusion (mean of probabilities).
final:  the best config re-scored on northstar onsets (content/amb split) for the domain check.

usage: PYTHONPATH=Qwen3-ASR .venv/bin/python core/research/zprobe/lid/lid_cold_round2.py
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
WINDOWS = (0.3, 0.5, 0.7, 1.0)
PER_LANG = 600
EVAL_PER_LANG = 500
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


paths = {"ko": [], "en": []}
for name, lang in (("kspon_manifest", "ko"), ("dailytalk_manifest", "en")):
    for l in open(f"models/zprobe/data/conv/{name}.jsonl", encoding="utf-8"):
        paths[lang].append(json.loads(l)["audio"].split("#")[0])
for k in paths:
    paths[k] = sorted(set(paths[k]))
    rng.shuffle(paths[k])

from core.utils import lid as lidmod  # noqa: E402
fused = asyncio.run(lidmod.FusedLid.build("models/zprobe/data/lid_phone_profiles.json"))


import math  # noqa: E402


def phone_prob_en(seg):
    """P(en) from the phone wing alone: per-phone mean log-prob under each language profile,
    softmaxed. None when fewer than 3 phones decode (too short to judge)."""
    phones = fused._phones(seg)  # noqa: SLF001
    if len(phones) < 3:
        return None
    s = {lang: sum(math.log(prof.get(ph, 1e-4)) for ph in phones) / len(phones)
         for lang, prof in fused._profiles.items() if lang in ("ko", "en")}  # noqa: SLF001
    if "ko" not in s or "en" not in s:
        return None
    z = np.exp(np.array([s["ko"], s["en"]]) - max(s.values()))
    return float(z[1] / z.sum())


t0 = time.time()
report = {}
for w in WINDOWS:
    n = int(w * SR)

    def windows_from(plist, lo, hi):
        out = []
        for p in plist[lo:hi]:
            try:
                wav, _ = librosa.load(p, sr=SR, duration=w + 3.0)
            except Exception:  # noqa: BLE001
                continue
            seg = onset_cut(wav, n)
            if seg is not None:
                out.append(seg)
        return out

    tr = {li: windows_from(paths[lang], 0, PER_LANG * 2)[:PER_LANG]
          for li, lang in enumerate(("ko", "en"))}
    ev = {li: windows_from(paths[lang], PER_LANG * 2, PER_LANG * 2 + EVAL_PER_LANG * 2)[:EVAL_PER_LANG]
          for li, lang in enumerate(("ko", "en"))}

    trf = {li: [frames_of(s) for s in tr[li]] for li in (0, 1)}
    Xm = np.stack([fr.mean(0) for li in (0, 1) for fr in trf[li]])
    ym = np.array([li for li in (0, 1) for _ in trf[li]])
    mu, sd = Xm.mean(0), Xm.std(0) + 1e-6
    xt = torch.tensor((Xm - mu) / sd, dtype=torch.float32)
    yt = torch.tensor(ym, dtype=torch.float32)
    wgt = torch.zeros(Xm.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([wgt, b], lr=0.05)
    for _ in range(300):
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(xt @ wgt + b, yt) \
            + 1e-3 * wgt.pow(2).sum()
        loss.backward()
        opt.step()
    wm, bm = wgt.detach().numpy(), float(b.item())

    Xf = np.stack([f for li in (0, 1) for fr in trf[li] for f in fr[::2]])
    yf = np.array([li for li in (0, 1) for fr in trf[li] for _ in fr[::2]])
    muf, sdf = Xf.mean(0), Xf.std(0) + 1e-6
    xtf = torch.tensor((Xf - muf) / sdf, dtype=torch.float32)
    ytf = torch.tensor(yf, dtype=torch.float32)
    wgf = torch.zeros(Xf.shape[1], requires_grad=True)
    bf_ = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([wgf, bf_], lr=0.05)
    for _ in range(200):
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(xtf @ wgf + bf_, ytf) \
            + 1e-3 * wgf.pow(2).sum()
        loss.backward()
        opt.step()
    wf, bf = wgf.detach().numpy(), float(bf_.item())

    hits = {"mean": 0, "vote": 0, "phone": 0, "fuse": 0}
    tot = {"mean": 0, "vote": 0, "phone": 0, "fuse": 0}
    for li in (0, 1):
        lang = ("ko", "en")[li]
        for seg in ev[li]:
            fr = frames_of(seg)
            pm = 1 / (1 + np.exp(-(float(np.dot((fr.mean(0) - mu) / sd, wm)) + bm)))
            pv = float((1 / (1 + np.exp(-(((fr - muf) / sdf) @ wf + bf)))).mean())
            pp = phone_prob_en(seg)
            for name, p in (("mean", pm), ("vote", pv), ("phone", pp),
                            ("fuse", None if pp is None else (pv + pp) / 2)):
                if p is None:
                    continue
                tot[name] += 1
                hits[name] += ("en" if p > 0.5 else "ko") == lang
    report[w] = {k: (round(hits[k] / tot[k], 3) if tot[k] else None, tot[k]) for k in hits}
    print(f"w={w}: " + "  ".join(f"{k} {report[w][k][0]} (n={report[w][k][1]})" for k in hits)
          + f"  ({time.time() - t0:.0f}s)", flush=True)

json.dump({str(k): v for k, v in report.items()},
          open("models/zprobe/data/prefix12/lid_cold_round2.json", "w"))
print("LID_R2_DONE", flush=True)
