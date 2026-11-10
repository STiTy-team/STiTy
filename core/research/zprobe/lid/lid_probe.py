"""Internal-probe LID: does the adapter's own audio encoder already know the language?

The live slot-reset judge (FusedLid = whisper-small + phone CTC profiles) runs two extra models on
every 0.5s step. This probes the koen merged model's audio_tower.ln_post output (mean-pooled) with
a linear head instead - the same ears that will write the text - and scores both judges on identical
trailing windows over the northstar scenes, truth = max-overlap language of the manifest spans.

train windows: FLEURS train ko/en recordings (from prefix12 word_times paths), random 2.5s cuts.
eval  windows: northstar scene wavs, trailing 2.5s at 0.5s cadence (the live lid_window/step).

usage: .venv/bin/python core/research/zprobe/lid/lid_probe.py [--per-lang 800] [--skip-fused]
"""
import argparse
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

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="models/Qwen3-ASR-1.7B-koen-seg-mix-merged")
ap.add_argument("--word-times", default="models/zprobe/data/prefix12/word_times.jsonl")
ap.add_argument("--ns", default="/home/mobility/stity-datasets/northstar-260928")
ap.add_argument("--per-lang", type=int, default=800)
ap.add_argument("--window-sec", type=float, default=2.5)
ap.add_argument("--step-sec", type=float, default=0.5)
ap.add_argument("--profiles", default="models/zprobe/data/lid_phone_profiles.json")
ap.add_argument("--skip-fused", action="store_true")
ap.add_argument("--out", default="models/zprobe/data/prefix12/lid_probe_report.json")
ap.add_argument("--head-out", default="models/zprobe/data/prefix12/lid_probe_head.npz")
a = ap.parse_args()
rng = random.Random(0)
SR = 16000
WIN = int(a.window_sec * SR)

from qwen_asr import Qwen3ASRModel  # noqa: E402

m = Qwen3ASRModel.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda",
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
    e = e.reshape(-1, e.shape[-1])
    return e.float().mean(0).cpu().numpy()


print("== train windows (FLEURS ko/en) ==", flush=True)
paths = {"ko": [], "en": []}
for l in open(a.word_times, encoding="utf-8"):
    r = json.loads(l)
    if r["lang"] in paths:
        paths[r["lang"]].append(r["audio"].split("#")[0])
X, y = [], []
t0 = time.time()
for li, lang in enumerate(("ko", "en")):
    ps = sorted(set(paths[lang]))
    rng.shuffle(ps)
    done = 0
    for p in ps:
        if done >= a.per_lang:
            break
        try:
            wav, _ = librosa.load(p, sr=SR)
        except Exception:  # noqa: BLE001
            continue
        if wav.shape[0] < WIN:
            continue
        s = rng.randint(0, wav.shape[0] - WIN)
        X.append(embed(wav[s:s + WIN]))
        y.append(li)
        done += 1
    print(f"{lang}: {done} windows {time.time() - t0:.0f}s", flush=True)
X = np.stack(X)
y = np.array(y)

mu, sd = X.mean(0), X.std(0) + 1e-6
Xn = (X - mu) / sd
w = torch.zeros(X.shape[1], requires_grad=True)
b = torch.zeros(1, requires_grad=True)
xt = torch.tensor(Xn, dtype=torch.float32)
yt = torch.tensor(y, dtype=torch.float32)
opt = torch.optim.Adam([w, b], lr=0.05)
for it in range(300):
    opt.zero_grad()
    loss = torch.nn.functional.binary_cross_entropy_with_logits(xt @ w + b, yt) + 1e-3 * w.pow(2).sum()
    loss.backward()
    opt.step()
acc_tr = (((xt @ w + b) > 0).float() == yt).float().mean().item()
print(f"probe train acc {acc_tr:.3f} (loss {loss.item():.4f})", flush=True)
np.savez(a.head_out,
         w=w.detach().numpy(), b=b.detach().numpy(), mu=mu, sd=sd,
         labels=np.array(["ko", "en"]), model=np.array([a.model]))
print(f"head saved -> {a.head_out}", flush=True)


def probe_verdict(wav):
    e = (embed(wav) - mu) / sd
    z = float(np.dot(e, w.detach().numpy()) + b.item())
    p = 1 / (1 + np.exp(-z))
    return ("en" if p > 0.5 else "ko"), max(p, 1 - p)


fused = None
if not a.skip_fused:
    from core.utils import lid as lidmod
    fused = asyncio.run(lidmod.FusedLid.build(a.profiles))

print("== eval (northstar trailing windows) ==", flush=True)
spans = {}
for line in open(f"{a.ns}/manifest.jsonl", encoding="utf-8"):
    it = json.loads(line)
    scene = it["audio"].split("/")[-1].removesuffix(".wav")
    spans.setdefault(scene, []).append((it["offset"], it["offset"] + it["duration"], it["src_lang"]))


def truth_of(scene, lo, hi):
    best, lang = 0.0, None
    for s, e, l in spans[scene]:
        d = min(hi, e) - max(lo, s)
        if d > best:
            best, lang = d, l
    return lang, best


rows = []
for scene in sorted(spans):
    wav_path = f"{a.ns}/audio/{scene}.wav"
    if not os.path.exists(wav_path):
        import glob
        g = glob.glob(f"{a.ns}/**/{scene}.wav", recursive=True)
        if not g:
            print(f"missing wav for {scene}", flush=True)
            continue
        wav_path = g[0]
    wav, _ = librosa.load(wav_path, sr=SR)
    switches = [s for (s, e, l), (s2, e2, l2) in zip(spans[scene], spans[scene][1:]) if l != l2 for s in (s2,)]
    t = a.window_sec
    while t * SR <= wav.shape[0]:
        seg = wav[int((t - a.window_sec) * SR):int(t * SR)]
        tr, ov = truth_of(scene, t - a.window_sec, t)
        if tr in ("ko", "en") and ov >= 1.0:
            pv, pc = probe_verdict(seg)
            fv, fa = (None, None)
            if fused is not None:
                fv, fa = fused.verdict(seg)
            near = any(abs(t - sw) <= 2.5 or abs((t - a.window_sec) - sw) <= 2.5 for sw in switches)
            rows.append({"scene": scene, "t": round(t, 1), "truth": tr, "near_switch": near,
                         "probe": pv, "probe_conf": round(float(pc), 3),
                         "fused": fv, "fused_agree": bool(fa) if fa is not None else None})
        t += a.step_sec

n = len(rows)


def acc(sel, key, gate=None):
    xs = [r for r in rows if sel(r) and r[key] in ("ko", "en") and (gate is None or gate(r))]
    hit = sum(r[key] == r["truth"] for r in xs)
    return (hit / len(xs) if xs else 0.0), len(xs)


report = {"windows": n, "train_acc": acc_tr,
          "probe_all": acc(lambda r: True, "probe"),
          "probe_conf80": acc(lambda r: True, "probe", lambda r: r["probe_conf"] >= 0.8),
          "probe_near_switch": acc(lambda r: r["near_switch"], "probe"),
          "probe_steady": acc(lambda r: not r["near_switch"], "probe")}
if fused is not None:
    report.update({
        "fused_all": acc(lambda r: True, "fused"),
        "fused_agree": acc(lambda r: True, "fused", lambda r: r["fused_agree"]),
        "fused_near_switch": acc(lambda r: r["near_switch"], "fused"),
        "fused_steady": acc(lambda r: not r["near_switch"], "fused")})
os.makedirs(os.path.dirname(a.out), exist_ok=True)
json.dump({"report": report, "rows": rows}, open(a.out, "w"), ensure_ascii=False)
for k, v in report.items():
    print(f"{k}: {v}", flush=True)
print("LID_PROBE_DONE", flush=True)
