"""Cold-start LID round 3: break the label-corpus confound and the channel cues.

Round 2's held-out 99.8% was a corpus classifier (kspon=ko, DailyTalk=en makes any channel cue a
perfect discriminant; cross-corpus eval collapsed to 63%). Fix: per-label corpus mixing
(ko = FLEURS-ko + kspon, en = FLEURS-en + DailyTalk) plus channel augmentation on training windows
(gain jitter, additive noise SNR 10-30dB, random one-pole lowpass). Eval on three fronts:
northstar onsets (content/amb split - the cross-domain verdict), conv held-out, fleurs held-out.

usage: PYTHONPATH=Qwen3-ASR .venv/bin/python core/research/zprobe/lid/lid_cold_round3.py
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
WINDOWS = (0.3, 0.5, 1.0)
PER_CORPUS = 400
EVAL_PER_LANG = 300
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
def emb(wav):
    inputs = m.processor(text=[BASE], audio=[wav.astype(np.float32)],
                         return_tensors="pt", padding=True)
    inputs = inputs.to(dev).to(torch.bfloat16)
    thinker(**inputs)
    return _enc["v"].reshape(-1, _enc["v"].shape[-1]).float().mean(0).cpu().numpy()


def onset_cut(wav, n):
    loud = np.flatnonzero(np.abs(wav) > 0.02)
    if loud.size == 0 or wav.shape[0] - loud[0] < n:
        return None
    return wav[loud[0]:loud[0] + n]


def augment(wav):
    w = wav * (10 ** (rng.uniform(-6, 6) / 20))
    snr = rng.uniform(10, 30)
    noise = np.random.default_rng(rng.randrange(1 << 30)).normal(0, 1, wav.shape[0])
    p_sig = np.mean(w ** 2) + 1e-9
    noise *= np.sqrt(p_sig / (10 ** (snr / 10)) / (np.mean(noise ** 2) + 1e-9))
    w = w + noise
    if rng.random() < 0.5:
        a = rng.uniform(0.1, 0.6)
        out = np.empty_like(w)
        prev = 0.0
        for i in range(w.shape[0]):
            prev = (1 - a) * w[i] + a * prev
            out[i] = prev
        w = out
    peak = np.abs(w).max() + 1e-9
    return w / peak * min(1.0, peak)


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


FL, CV = fleurs_paths(), conv_paths()
for d in (FL, CV):
    for lang in d:
        rng.shuffle(d[lang])

utts = []
for line in open(f"{NS}/manifest.jsonl", encoding="utf-8"):
    it = json.loads(line)
    scene = it["audio"].split("/")[-1].removesuffix(".wav")
    head = (it.get("reference") or {}).get("transcript", "")
    utts.append((scene, it["offset"], it["duration"], it["src_lang"], bool(AMB.match(head))))
scene_wav = {s: librosa.load(f"{NS}/audio/{s}.wav", sr=SR)[0]
             for s in sorted({u[0] for u in utts})}


def collect(plist, lo, hi, n, cap, aug):
    out = []
    for p in plist[lo:hi]:
        if len(out) >= cap:
            break
        try:
            wav, _ = librosa.load(p, sr=SR, duration=n / SR + 3.0)
        except Exception:  # noqa: BLE001
            continue
        seg = onset_cut(wav, n)
        if seg is None:
            continue
        out.append(emb(augment(seg) if aug else seg))
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
out_report = {}
for w_sec in WINDOWS:
    n = int(w_sec * SR)
    Xs, ys = [], []
    for li, lang in enumerate(("ko", "en")):
        for plist in (FL[lang], CV[lang]):
            for e in collect(plist, 0, PER_CORPUS * 3, n, PER_CORPUS, aug=True):
                Xs.append(e)
                ys.append(li)
    wm, bm, mu, sd = fit(np.stack(Xs), np.array(ys))
    np.savez(f"models/zprobe/data/prefix12/lid_cold_head_w{w_sec}.npz",
             w=wm, b=np.array([bm]), mu=mu, sd=sd, labels=np.array(["ko", "en"]))

    def pred(e):
        return 1 / (1 + np.exp(-(float(np.dot((e - mu) / sd, wm)) + bm)))

    res = {}
    for name, plists in (("fleurs", FL), ("conv", CV)):
        hit = tot = 0
        for li, lang in enumerate(("ko", "en")):
            for e in collect(plists[lang], PER_CORPUS * 3, PER_CORPUS * 3 + EVAL_PER_LANG * 2,
                             n, EVAL_PER_LANG, aug=False):
                hit += (pred(e) > 0.5) == bool(li)
                tot += 1
        res[name] = (round(hit / max(1, tot), 3), tot)
    hit_c = tot_c = hit_a = tot_a = 0
    for scene, off, dur, lang, amb in utts:
        if lang not in ("ko", "en") or dur < w_sec:
            continue
        seg = onset_cut(scene_wav[scene][int(off * SR):int((off + dur) * SR)], n)
        if seg is None:
            continue
        ok = (pred(emb(seg)) > 0.5) == (lang == "en")
        if amb:
            tot_a += 1
            hit_a += ok
        else:
            tot_c += 1
            hit_c += ok
    res["ns_content"] = (round(hit_c / max(1, tot_c), 3), tot_c)
    res["ns_amb"] = (round(hit_a / max(1, tot_a), 3), tot_a)
    out_report[w_sec] = res
    print(f"w={w_sec}: " + "  ".join(f"{k} {v[0]} (n={v[1]})" for k, v in res.items())
          + f"  ({time.time() - t0:.0f}s)", flush=True)

json.dump({str(k): v for k, v in out_report.items()},
          open("models/zprobe/data/prefix12/lid_cold_round3.json", "w"))
print("LID_R3_DONE", flush=True)
