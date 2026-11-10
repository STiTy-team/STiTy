"""Cold-start LID round 4: a valid multi-corpus instrument plus the operational metric.

A) Two heads per window (fleurs-only vs fleurs+conv mixed w/ channel augment), scored PER CORPUS:
   fleurs held-out, conv held-out, aihub71524 ko, librispeech dev-clean en, northstar content/amb.
   Per-corpus reporting keeps eval free of the label=corpus confound (round 2's lesson).
B) Detection latency on northstar: judge every 0.5s from utterance onset (growing window, capped
   at 1.0s tail), confirm on two consecutive agreeing verdicts; report time-to-confirm and the
   confirmed verdict's accuracy, content/amb split.

usage: PYTHONPATH=Qwen3-ASR .venv/bin/python core/research/zprobe/lid/lid_cold_round4.py
"""
import glob
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
SR = 16000
PER_LANG = 600
HELD = 300
THIRD = 400
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
    return w + noise


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
AIHUB = sorted(glob.glob("/home/mobility/stity-datasets/aihub71524/data/wav/ko/*.wav"))
LIBRI = sorted(glob.glob(os.path.expanduser("~/datasets/librispeech/dev-clean/*/*/*.flac")))
for d in (FL, CV):
    for lang in d:
        rng.shuffle(d[lang])
rng.shuffle(AIHUB)
rng.shuffle(LIBRI)
print(f"aihub {len(AIHUB)}, libri {len(LIBRI)}", flush=True)

utts = []
for line in open(f"{NS}/manifest.jsonl", encoding="utf-8"):
    it = json.loads(line)
    scene = it["audio"].split("/")[-1].removesuffix(".wav")
    head = (it.get("reference") or {}).get("transcript", "")
    utts.append((scene, it["offset"], it["duration"], it["src_lang"], bool(AMB.match(head))))
scene_wav = {s: librosa.load(f"{NS}/audio/{s}.wav", sr=SR)[0]
             for s in sorted({u[0] for u in utts})}


def collect(plist, lo, cap, n, aug):
    out = []
    i = lo
    while len(out) < cap and i < len(plist):
        p = plist[i]
        i += 1
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
heads = {}
print("== A. 헤드 2종 x 코퍼스별 채점 ==", flush=True)
for w_sec in (0.5, 1.0):
    n = int(w_sec * SR)
    tr_fl = {li: collect(FL[lang], 0, PER_LANG, n, aug=False)
             for li, lang in enumerate(("ko", "en"))}
    heads[("fleurs", w_sec)] = fit(np.stack(tr_fl[0] + tr_fl[1]),
                                   np.array([0] * len(tr_fl[0]) + [1] * len(tr_fl[1])))
    tr_mix = {li: tr_fl[li] + collect(CV[lang], 0, PER_LANG, n, aug=True)
              for li, lang in enumerate(("ko", "en"))}
    heads[("mixed", w_sec)] = fit(np.stack(tr_mix[0] + tr_mix[1]),
                                  np.array([0] * len(tr_mix[0]) + [1] * len(tr_mix[1])))

    evals = {
        "fl_ko": (collect(FL["ko"], PER_LANG, HELD, n, False), "ko"),
        "fl_en": (collect(FL["en"], PER_LANG, HELD, n, False), "en"),
        "cv_ko": (collect(CV["ko"], PER_LANG, HELD, n, False), "ko"),
        "cv_en": (collect(CV["en"], PER_LANG, HELD, n, False), "en"),
        "aihub_ko": (collect(AIHUB, 0, THIRD, n, False), "ko"),
        "libri_en": (collect(LIBRI, 0, THIRD, n, False), "en"),
    }
    ns_embs = []
    for scene, off, dur, lang, amb in utts:
        if lang not in ("ko", "en") or dur < w_sec:
            continue
        seg = onset_cut(scene_wav[scene][int(off * SR):int((off + dur) * SR)], n)
        if seg is not None:
            ns_embs.append((emb(seg), lang, amb))

    for name in ("fleurs", "mixed"):
        wm, bm, mu, sd = heads[(name, w_sec)]

        def acc(embs, lang):
            if not embs:
                return None
            hit = sum((1 / (1 + np.exp(-(float(np.dot((e - mu) / sd, wm)) + bm))) > 0.5)
                      == (lang == "en") for e in embs)
            return round(hit / len(embs), 3)

        row = {k: acc(v[0], v[1]) for k, v in evals.items()}
        row["ns_content"] = round(sum((1 / (1 + np.exp(-(float(np.dot((e - mu) / sd, wm)) + bm))) > 0.5)
                                      == (lg == "en") for e, lg, amb in ns_embs if not amb)
                                  / max(1, sum(1 for _, _, amb in ns_embs if not amb)), 3)
        row["ns_amb"] = round(sum((1 / (1 + np.exp(-(float(np.dot((e - mu) / sd, wm)) + bm))) > 0.5)
                                  == (lg == "en") for e, lg, amb in ns_embs if amb)
                              / max(1, sum(1 for _, _, amb in ns_embs if amb)), 3)
        print(f"w={w_sec} {name:6s}: " + "  ".join(f"{k} {v}" for k, v in row.items())
              + f"  ({time.time() - t0:.0f}s)", flush=True)

print("== B. 확정 지연 시뮬레이션 (북극성, 2연속 일치 확정) ==", flush=True)
for head_name in ("fleurs", "mixed"):
    confirms = {"content": [], "amb": []}
    for scene, off, dur, lang, amb in utts:
        if lang not in ("ko", "en") or dur < 1.0:
            continue
        chunk = scene_wav[scene][int(off * SR):int((off + dur) * SR)]
        loud = np.flatnonzero(np.abs(chunk) > 0.02)
        if loud.size == 0:
            continue
        chunk = chunk[loud[0]:]
        prev = None
        done = False
        for k in range(1, 7):
            t = 0.5 * k
            if t > dur or int(t * SR) > chunk.shape[0]:
                break
            w_sec = 0.5 if k == 1 else 1.0
            wm, bm, mu, sd = heads[(head_name, w_sec)]
            seg = chunk[max(0, int((t - w_sec) * SR)):int(t * SR)]
            p = 1 / (1 + np.exp(-(float(np.dot((emb(seg) - mu) / sd, wm)) + bm)))
            v = "en" if p > 0.5 else "ko"
            if prev is not None and v == prev:
                confirms["amb" if amb else "content"].append((t, v == lang))
                done = True
                break
            prev = v
        if not done and prev is not None:
            confirms["amb" if amb else "content"].append((None, prev == lang))
    for kind, rows in confirms.items():
        ts = [t for t, ok in rows if t is not None]
        accs = [ok for t, ok in rows if t is not None]
        print(f"{head_name} {kind}: 확정 {len(ts)}/{len(rows)}, 중앙 {np.median(ts) if ts else '-'}s, "
              f"확정 정확도 {np.mean(accs):.3f}" if accs else f"{head_name} {kind}: 확정 없음", flush=True)
print("LID_R4_DONE", flush=True)
