#!/usr/bin/env python
"""Throwaway probe: where (if anywhere) does Qwen3-ASR hold a language-neutral
representation of what was said?

extract: run FLEURS n-way parallel utterances through the frozen model, store
         pooled hidden states per layer (encoder, LLM 0..28) + logit-lens stats.
analyze: cross-lingual retrieval, audio/text convergence, logit lens tables.
gen:     cheap behavioural checks (text translation survival, language-tag swap).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

FLEURS = Path(os.path.expanduser("~/datasets/fleurs/data"))
LANGS = {"en": ("en_us", "English"), "ko": ("ko_kr", "Korean"),
         "ja": ("ja_jp", "Japanese"), "zh": ("cmn_hans_cn", "Chinese")}
MODEL = os.environ.get("PROBE_MODEL", "Qwen/Qwen3-ASR-1.7B")


def script_of(s: str) -> str:
    for ch in s:
        o = ord(ch)
        if 0xAC00 <= o <= 0xD7A3 or 0x1100 <= o <= 0x11FF or 0x3130 <= o <= 0x318F:
            return "hangul"
        if 0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF:
            return "han"
        if 0x3040 <= o <= 0x30FF:
            return "kana"
        if ch.isascii() and ch.isalpha():
            return "latin"
    return "other"


SCRIPTS = ["latin", "hangul", "han", "kana", "other"]
ABBR = {"latin": "la", "hangul": "hg", "han": "hn", "kana": "ka", "other": "ot"}
LANG_SCRIPT = {"en": "latin", "ko": "hangul", "zh": "han", "ja": "kana"}


def read_rows(lang: str) -> dict[str, list[str]]:
    p = FLEURS / LANGS[lang][0] / "test.tsv"
    return {r[0]: r for r in csv.reader(open(p, encoding="utf-8"), delimiter="\t") if len(r) >= 6}


def common_ids(langs: list[str]) -> list[str]:
    sets = [set(read_rows(l)) for l in langs]
    return sorted(set.intersection(*sets), key=int)


# --------------------------------------------------------------------------- extract
ADAPTER = os.environ.get("PROBE_ADAPTER", "")


def load_model():
    import torch
    from qwen_asr import Qwen3ASRModel
    m = Qwen3ASRModel.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda",
                                      max_inference_batch_size=1)
    if ADAPTER:
        from peft import PeftModel
        pm = PeftModel.from_pretrained(m.model, ADAPTER)
        m.model = pm.merge_and_unload()
        print(f"[adapter] merged {ADAPTER}", flush=True)
    m.model.eval()
    return m


class Taps:
    """Forward hooks: encoder ln_post output (pre-projector) and every LLM layer."""

    def __init__(self, thinker):
        self.enc = None
        self.h = []  # [L0 (embeds), L1..L28]
        self.handles = []
        self.handles.append(thinker.audio_tower.ln_post.register_forward_hook(self._enc))
        self.handles.append(thinker.model.layers[0].register_forward_pre_hook(self._l0))
        for layer in thinker.model.layers:
            self.handles.append(layer.register_forward_hook(self._layer))

    def _enc(self, mod, inp, out):
        self.enc = out.detach()

    def _l0(self, mod, args):
        self.h = [args[0].detach()]

    def _layer(self, mod, inp, out):
        self.h.append((out[0] if isinstance(out, tuple) else out).detach())


def text_prompt(text: str) -> str:
    return ("<|im_start|>system\n<|im_end|>\n<|im_start|>user\n" + text +
            "<|im_end|>\n<|im_start|>assistant\n")


def extract(args):
    import librosa
    import torch
    m = load_model()
    thinker = m.model.thinker
    tok = m.processor.tokenizer
    dev = next(thinker.parameters()).device
    audio_tok = thinker.config.audio_token_id
    asr_tok = tok.convert_tokens_to_ids("<asr_text>")
    taps = Taps(thinker)
    norm, head = thinker.model.norm, thinker.lm_head

    langs = args.langs.split(",")
    ids = common_ids(langs)
    if args.limit:
        ids = ids[: args.limit]
    print(f"langs={langs} common ids={len(ids)}", flush=True)
    rows = {l: read_rows(l) for l in langs}

    out = {}
    t0 = time.time()
    for lang in langs:
        store = {k: [] for k in ["enc_mean", "audio_mean", "asr_pos", "trans_mean", "text_mean", "text_last"]}
        lens = {"audio": [], "trans": [], "text": []}
        ll_top1_hit = None   # [L+1]
        ll_script = None     # [L+1, len(SCRIPTS)]
        ll_n = 0
        for i, uid in enumerate(ids):
            r = rows[lang][uid]
            wav_path = FLEURS / LANGS[lang][0] / "audio" / "test" / r[1]
            wav, _ = librosa.load(str(wav_path), sr=16000)
            ref = r[2].strip()

            # ---- audio + teacher-forced transcript
            prompt = m._build_text_prompt(context="", force_language=LANGS[lang][1]) + ref
            inputs = m.processor(text=[prompt], audio=[wav], return_tensors="pt", padding=True)
            inputs = inputs.to(dev).to(torch.bfloat16)
            with torch.no_grad():
                thinker(**inputs)
            ids_ = inputs["input_ids"][0]
            amask = ids_ == audio_tok
            asr_i = (ids_ == asr_tok).nonzero()[0, 0].item()
            H = torch.stack(taps.h, 0)[:, 0]  # [L+1, T, D]
            store["enc_mean"].append(taps.enc.float().mean(0).cpu().numpy())
            store["audio_mean"].append(H[:, amask].float().mean(1).cpu().numpy())
            store["asr_pos"].append(H[:, asr_i].float().cpu().numpy())
            store["trans_mean"].append(H[:, asr_i + 1:].float().mean(1).cpu().numpy())
            lens["audio"].append(int(amask.sum()))
            lens["trans"].append(int(ids_.shape[0] - asr_i - 1))

            # ---- logit lens at positions predicting transcript tokens
            pred_pos = torch.arange(asr_i, ids_.shape[0] - 1, device=dev)
            tgt = ids_[asr_i + 1:]
            with torch.no_grad():
                hs = H[:, pred_pos]                      # [L+1, P, D]
                logits = head(norm(hs))                  # [L+1, P, V]
                top1 = logits.argmax(-1)                 # [L+1, P]
            hit = (top1 == tgt[None]).float().sum(1).cpu().numpy()
            if ll_top1_hit is None:
                ll_top1_hit = np.zeros(H.shape[0])
                ll_script = np.zeros((H.shape[0], len(SCRIPTS)))
            ll_top1_hit += hit
            ll_n += int(tgt.shape[0])
            top1_cpu = top1.cpu().numpy()
            for l in range(top1_cpu.shape[0]):
                for t in top1_cpu[l]:
                    ll_script[l, SCRIPTS.index(script_of(tok.decode([int(t)])))] += 1

            # ---- text only (the transcript as user text)
            tp = text_prompt(ref)
            enc = tok(tp, return_tensors="pt").to(dev)
            with torch.no_grad():
                thinker(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
            H = torch.stack(taps.h, 0)[:, 0]
            # text span = tokens between "user\n" and "<|im_end|>" of the user turn
            pre = len(tok(text_prompt("")[: text_prompt("").index("<|im_end|>\n<|im_start|>assistant")])["input_ids"])
            n_text = len(tok(ref, add_special_tokens=False)["input_ids"])
            store["text_mean"].append(H[:, pre:pre + n_text].float().mean(1).cpu().numpy())
            store["text_last"].append(H[:, -1].float().cpu().numpy())
            lens["text"].append(n_text)

            if (i + 1) % 20 == 0:
                print(f"[{lang}] {i + 1}/{len(ids)}  {time.time() - t0:.0f}s", flush=True)

        out[lang] = {k: np.stack(v).astype(np.float16) for k, v in store.items()}
        out[lang]["ll_top1_acc"] = ll_top1_hit / max(ll_n, 1)
        out[lang]["ll_script"] = ll_script / ll_script.sum(1, keepdims=True)
        out[lang]["lens"] = json.dumps(lens)
        print(f"[{lang}] done  audio tokens mean={np.mean(lens['audio']):.1f} "
              f"trans tokens mean={np.mean(lens['trans']):.1f}", flush=True)

    np.savez(args.out, ids=np.array(ids), langs=np.array(langs),
             **{f"{l}__{k}": v for l, d in out.items() for k, v in d.items()})
    print("saved", args.out, flush=True)


# --------------------------------------------------------------------------- analyze
def _norm(x):
    x = x.astype(np.float32)
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def retrieval(A, B):
    """A,B: [N, D] same-index pairs. mean of acc@1 in both directions, plus
    same-pair vs random-pair cosine gap."""
    a, b = _norm(A), _norm(B)
    S = a @ b.T
    n = S.shape[0]
    acc = 0.5 * ((S.argmax(1) == np.arange(n)).mean() + (S.argmax(0) == np.arange(n)).mean())
    same = np.diag(S).mean()
    rand = (S.sum() - np.trace(S)) / (n * n - n)
    return acc, same, rand


def center(X):
    return X - X.mean(0, keepdims=True)


def lang_probe(X, y):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=300, C=0.1))
    return cross_val_score(clf, X.astype(np.float32), y, cv=5).mean()


def analyze(args):
    z = np.load(args.npz)
    langs = list(z["langs"])
    N = len(z["ids"])
    L = z[f"{langs[0]}__audio_mean"].shape[1]
    print(f"N={N} utterances per language, langs={langs}, LLM taps={L} (L0=embeds)\n")

    pairs = [(a, b) for i, a in enumerate(langs) for b in langs[i + 1:]]

    def tbl(title, header, rows):
        print(f"## {title}")
        print("| " + " | ".join(header) + " |")
        print("|" + "---|" * len(header))
        for r in rows:
            print("| " + " | ".join(r) + " |")
        print()

    # 1. cross-lingual retrieval per layer, per pooling, raw / centered
    for key in ["audio_mean", "asr_pos", "text_mean", "text_last"]:
        rows = []
        # encoder (pre-projector) as row "enc" only for audio_mean
        if key == "audio_mean":
            r = ["enc"]
            for a, b in pairs:
                A, B = z[f"{a}__enc_mean"], z[f"{b}__enc_mean"]
                acc, s, rd = retrieval(A, B)
                accc, _, _ = retrieval(center(A), center(B))
                r += [f"{acc:.2f}", f"{accc:.2f}", f"{s - rd:+.3f}"]
            rows.append(r)
        for l in range(L):
            r = [f"L{l}"]
            for a, b in pairs:
                A, B = z[f"{a}__{key}"][:, l], z[f"{b}__{key}"][:, l]
                acc, s, rd = retrieval(A, B)
                accc, _, _ = retrieval(center(A), center(B))
                r += [f"{acc:.2f}", f"{accc:.2f}", f"{s - rd:+.3f}"]
            rows.append(r)
        hdr = ["layer"] + [h for a, b in pairs for h in (f"{a}-{b} acc@1", "centered", "same-rand cos")]
        tbl(f"cross-lingual retrieval — {key} (chance {1 / N:.3f})", hdr, rows)

    # 1b. common-z tests on audio_mean: language subspace removed + joint multilingual gallery
    def remove_lang_subspace(Xs):
        """Xs: list of [N, D] per language. Subtract per-language mean, then also project out
        the span of (lang mean - global mean) directions (k-1 dims). Returns list."""
        mus = np.stack([x.mean(0) for x in Xs])                    # [k, D]
        G = mus.mean(0, keepdims=True)
        Q, _ = np.linalg.qr((mus - G).T.astype(np.float64))        # [D, k]
        out = []
        for x, mu in zip(Xs, mus):
            xc = (x - mu).astype(np.float64)
            out.append((xc - (xc @ Q) @ Q.T).astype(np.float32))
        return out

    def joint_gallery(Xs, q):
        """query language q against the union of all other languages. acc@1 = same utt in any
        other language; 'wrong-lang' = top hit is a different utterance (any language)."""
        gal = np.concatenate([_norm(x) for i, x in enumerate(Xs) if i != q])
        gid = np.concatenate([np.arange(len(x)) for i, x in enumerate(Xs) if i != q])
        S = _norm(Xs[q]) @ gal.T
        top = S.argmax(1)
        return (gid[top] == np.arange(len(Xs[q]))).mean()

    rows = []
    for l in range(L):
        Xs = [z[f"{a}__audio_mean"][:, l] for a in langs]
        Xc = [center(x) for x in Xs]
        Xp = remove_lang_subspace(Xs)
        r = [f"L{l}"]
        # mean pairwise acc over all pairs: raw / centered / subspace-removed
        for V in (Xs, Xc, Xp):
            accs = [retrieval(V[i], V[j])[0] for i in range(len(langs)) for j in range(i + 1, len(langs))]
            r.append(f"{np.mean(accs):.2f}")
        # joint gallery per query language on centered and subspace-removed
        for V in (Xc, Xp):
            r.append(" ".join(f"{a}:{joint_gallery(V, i):.2f}" for i, a in enumerate(langs)))
        rows.append(r)
    tbl("common-z tests — audio_mean (pairwise mean acc@1: raw / centered / lang-subspace removed; "
        "joint gallery acc@1 = query lang vs union of all other langs)",
        ["layer", "pair raw", "pair centered", "pair subsp-rm", "joint centered", "joint subsp-rm"], rows)

    # 2. language-ID linear probe on audio_mean (raw, and after per-lang centering is trivial -> skip)
    if len(langs) >= 2:
        rows = []
        y = np.concatenate([[i] * N for i in range(len(langs))])
        for l in range(L):
            X = np.concatenate([z[f"{a}__audio_mean"][:, l] for a in langs])
            rows.append([f"L{l}", f"{lang_probe(X, y):.3f}"])
        tbl("language-ID linear probe (5-fold acc) — audio_mean", ["layer", "acc"], rows)

    # 3. audio <-> text convergence, same language and cross language
    rows = []
    for l in range(L):
        r = [f"L{l}"]
        for a in langs:
            acc, s, rd = retrieval(z[f"{a}__audio_mean"][:, l], z[f"{a}__text_mean"][:, l])
            r += [f"{acc:.2f}", f"{s - rd:+.3f}"]
        for a, b in pairs:
            acc, s, rd = retrieval(z[f"{a}__audio_mean"][:, l], z[f"{b}__text_mean"][:, l])
            accc, _, _ = retrieval(center(z[f"{a}__audio_mean"][:, l]), center(z[f"{b}__text_mean"][:, l]))
            r += [f"{acc:.2f}", f"{accc:.2f}"]
        rows.append(r)
    hdr = ["layer"] + [h for a in langs for h in (f"{a} audio→{a} text acc", "same-rand")] + \
          [h for a, b in pairs for h in (f"{a} audio→{b} text acc", "centered")]
    tbl("audio vs text convergence (audio_mean vs text_mean)", hdr, rows)

    # 4. logit lens
    rows = []
    for l in range(L):
        r = [f"L{l}"]
        for a in langs:
            r.append(f"{z[f'{a}__ll_top1_acc'][l]:.2f}")
            sc = z[f"{a}__ll_script"][l]
            r.append(" ".join(f"{ABBR[s]}{v:.2f}" for s, v in zip(SCRIPTS, sc) if v >= 0.05))
        rows.append(r)
    hdr = ["layer"] + [h for a in langs for h in (f"{a} top1 acc", f"{a} top1 script")]
    tbl("logit lens at transcript positions (teacher-forced)", hdr, rows)


# --------------------------------------------------------------------------- gen
def gen(args):
    import librosa
    import torch
    m = load_model()
    thinker = m.model.thinker
    tok = m.processor.tokenizer
    dev = next(thinker.parameters()).device
    ids = common_ids(["en", "ko"])[: args.limit or 8]
    en, ko = read_rows("en"), read_rows("ko")

    def run_text(prompt):
        enc = tok(text_prompt(prompt), return_tensors="pt").to(dev)
        with torch.no_grad():
            o = thinker.generate(**enc, max_new_tokens=96, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        return tok.decode(o[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)

    def run_audio(lang, uid, force):
        r = (en if lang == "en" else ko)[uid]
        wav, _ = librosa.load(str(FLEURS / LANGS[lang][0] / "audio" / "test" / r[1]), sr=16000)
        prompt = m._build_text_prompt(context="", force_language=force)
        inputs = m.processor(text=[prompt], audio=[wav], return_tensors="pt", padding=True)
        inputs = inputs.to(dev).to(torch.bfloat16)
        with torch.no_grad():
            o = m.model.generate(**inputs, max_new_tokens=96)
        seq = o.sequences if hasattr(o, "sequences") else o
        return tok.decode(seq[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    print("## A. text-only translation (is the LLM's text translation ability alive?)\n")
    for uid in ids:
        e, k = en[uid][2].strip(), ko[uid][2].strip()
        t_ek = run_text("Translate the following English into Korean. Output only the translation.\n" + e)
        t_ke = run_text("Translate the following Korean into English. Output only the translation.\n" + k)
        print(f"EN: {e}\n  en→ko: {t_ek!r}")
        print(f"KO: {k}\n  ko→en: {t_ke!r}\n")

    print("\n## B. language-tag swap on audio (English audio, forced 'Korean' tag, and vice versa)\n")
    for uid in ids:
        print(f"EN audio | tag=Korean  : {run_audio('en', uid, 'Korean')!r}")
        print(f"   ref: {en[uid][2].strip()}")
        print(f"KO audio | tag=English : {run_audio('ko', uid, 'English')!r}")
        print(f"   ref: {ko[uid][2].strip()}\n")

    print("\n## C. audio + translation instruction in system prompt\n")
    for uid in ids[:4]:
        r = en[uid]
        wav, _ = librosa.load(str(FLEURS / "en_us" / "audio" / "test" / r[1]), sr=16000)
        prompt = m._build_text_prompt(context="Translate the speech into Korean.", force_language="Korean")
        inputs = m.processor(text=[prompt], audio=[wav], return_tensors="pt", padding=True).to(dev).to(torch.bfloat16)
        with torch.no_grad():
            o = m.model.generate(**inputs, max_new_tokens=96)
        seq = o.sequences if hasattr(o, "sequences") else o
        dec = tok.decode(seq[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        print(f"EN audio | sys=translate→ko: {dec!r}")
        print(f"   ref: {r[2].strip()}\n")


# --------------------------------------------------------------------------- tagswap
def tagswap(args):
    """Zero-shot ST probe: source-language audio with the target-language tag (and optionally
    a translate instruction in the system prompt). FLEURS is parallel, so the target-language
    transcript is a translation reference. Writes jsonl per utterance and a summary."""
    import json
    import librosa
    import torch
    import sacrebleu
    m = load_model()
    tok = m.processor.tokenizer
    dev = next(m.model.thinker.parameters()).device
    src, tgt = args.pair.split("-")
    ids = common_ids([src, tgt])[: args.limit] if args.limit else common_ids([src, tgt])
    rs, rt = read_rows(src), read_rows(tgt)
    ctx = f"Translate the speech into {LANGS[tgt][1]}." if args.sys else ""
    out = open(args.out, "w", encoding="utf-8")
    hyps, refs_t, refs_s, scripts = [], [], [], []
    t0 = time.time()
    for i, uid in enumerate(ids):
        r = rs[uid]
        wav, _ = librosa.load(str(FLEURS / LANGS[src][0] / "audio" / "test" / r[1]), sr=16000)
        prompt = m._build_text_prompt(context=ctx, force_language=LANGS[tgt][1])
        inputs = m.processor(text=[prompt], audio=[wav], return_tensors="pt", padding=True)
        inputs = inputs.to(dev).to(torch.bfloat16)
        with torch.no_grad():
            o = m.model.generate(**inputs, max_new_tokens=128)
        seq = o.sequences if hasattr(o, "sequences") else o
        hyp = tok.decode(seq[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        sc = script_of(hyp)
        rec = {"id": uid, "hyp": hyp, "script": sc, "ref_tgt": rt[uid][2].strip(), "ref_src": r[2].strip()}
        out.write(json.dumps(rec, ensure_ascii=False) + "\n"); out.flush()
        hyps.append(hyp); refs_t.append(rec["ref_tgt"]); refs_s.append(rec["ref_src"]); scripts.append(sc)
        if (i + 1) % 20 == 0:
            print(f"[{src}->{tgt} sys={args.sys}] {i + 1}/{len(ids)}  {time.time() - t0:.0f}s", flush=True)
    ts_summary(src, tgt, args.sys, hyps, refs_t, refs_s, scripts)


def tssum(args):
    import json
    src, tgt = args.pair.split("-")
    recs = [json.loads(l) for l in open(args.jsonl, encoding="utf-8")]
    ts_summary(src, tgt, args.sys, [r["hyp"] for r in recs], [r["ref_tgt"] for r in recs],
               [r["ref_src"] for r in recs], [r["script"] for r in recs])


def ts_summary(src, tgt, sys_flag, hyps, refs_t, refs_s, scripts):
    import sacrebleu
    class args: pass
    args.sys = sys_flag
    ids = hyps
    tgt_script = LANG_SCRIPT[tgt]

    def purity(h):
        """fraction of letter characters that belong to the target language's script set."""
        ok = {"en": lambda o: o < 0x250, "ko": lambda o: 0xAC00 <= o <= 0xD7A3 or 0x1100 <= o <= 0x11FF or 0x3130 <= o <= 0x318F,
              "zh": lambda o: 0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF,
              "ja": lambda o: 0x3040 <= o <= 0x30FF or 0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF}[tgt]
        letters = [c for c in h if c.isalpha()]
        return sum(ok(ord(c)) for c in letters) / len(letters) if letters else 0.0
    pur = np.array([purity(h) for h in hyps])
    pure = [i for i in range(len(hyps)) if pur[i] >= 0.9]
    has_kana = lambda h: any(0x3040 <= ord(c) <= 0x30FF for c in h)
    has_han = lambda h: any(0x4E00 <= ord(c) <= 0x9FFF for c in h)
    if tgt == "ja":
        on = [i for i, h in enumerate(hyps) if has_kana(h)]
    elif tgt == "zh":
        on = [i for i, h in enumerate(hyps) if has_han(h) and not has_kana(h)]
    else:
        on = [i for i, s in enumerate(scripts) if s == tgt_script]
    tk = "zh" if tgt == "zh" else ("char" if tgt == "ja" else "13a")
    chrf_all = sacrebleu.corpus_chrf(hyps, [refs_t]).score
    chrf_on = sacrebleu.corpus_chrf([hyps[i] for i in on], [[refs_t[i] for i in on]]).score if on else float("nan")
    bleu_on = sacrebleu.corpus_bleu([hyps[i] for i in on], [[refs_t[i] for i in on]], tokenize=tk).score if on else float("nan")
    chrf_src = sacrebleu.corpus_chrf(hyps, [refs_s]).score
    from collections import Counter
    print(f"\n## tagswap {src}->{tgt} sys={args.sys} N={len(ids)}")
    print(f"output script: {dict(Counter(scripts))}")
    print(f"target-script rate: {len(on) / len(ids):.3f}")
    print(f"script purity: mean={pur.mean():.3f}  rate(purity>=0.9)={len(pure) / len(ids):.3f}  "
          f"rate(purity<0.1, i.e. stayed source)={(pur < 0.1).mean():.3f}")
    if pure:
        print(f"chrF vs target ref, purity>=0.9 only: {sacrebleu.corpus_chrf([hyps[i] for i in pure], [[refs_t[i] for i in pure]]).score:.1f}")
    print(f"chrF vs target ref: all={chrf_all:.1f}  on-target-only={chrf_on:.1f}  BLEU on-target-only={bleu_on:.1f}")
    print(f"chrF vs source ref (how much is still just transcription): {chrf_src:.1f}")


# --------------------------------------------------------------------------- steer
def steer(args):
    """Activation steering: at LLM tap L (L_k = output of layers[k-1]; L0 = embeds), add
    alpha * (mean_tgt - mean_src) of audio_mean (from an extract npz) to the hidden states at the
    audio-token positions during prefill, then generate with the target-language tag.
    --where audio|all : audio positions only, or every prefill position."""
    import json
    import librosa
    import torch
    m = load_model()
    thinker = m.model.thinker
    tok = m.processor.tokenizer
    dev = next(thinker.parameters()).device
    audio_tok = thinker.config.audio_token_id
    src, tgt = args.pair.split("-")
    z = np.load(args.npz)
    layers = [int(x) for x in args.layers.split(",")]
    pos = args.pos or ("all" if args.where == "all" else "audio")
    asr_tok = tok.convert_tokens_to_ids("<asr_text>")

    def direction(key, L):
        A, B = z[f"{src}__{key}"][:, L].astype(np.float64), z[f"{tgt}__{key}"][:, L].astype(np.float64)
        mu_s, mu_t = A.mean(0), B.mean(0)
        if args.vec == "mean":
            return mu_t - mu_s
        # probe: LDA direction with shrinkage, scaled so the mean gap along it is covered
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        lda = LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto")
        lda.fit(np.concatenate([A, B]), np.array([0] * len(A) + [1] * len(B)))
        d = lda.coef_[0]; d = d / np.linalg.norm(d)
        gap = float((mu_t - mu_s) @ d)
        cos = gap / np.linalg.norm(mu_t - mu_s)
        print(f"  probe dir L{L} {key}: gap along dir={gap:.2f} (|mean diff|={np.linalg.norm(mu_t - mu_s):.2f}, cos={cos:.2f})", flush=True)
        return gap * d

    deltas, deltas_asr = {}, {}
    for L in layers:
        if pos in ("audio", "all", "audio+asr"):
            d = torch.tensor(direction("audio_mean", L), dtype=torch.float32)
            print(f"L{L} audio: |delta|={d.norm():.2f}", flush=True)
            deltas[L] = (args.alpha * d).to(dev, torch.bfloat16)
        if pos in ("asr", "audio+asr"):
            d = torch.tensor(direction("asr_pos", L), dtype=torch.float32)
            print(f"L{L} asr_pos: |delta|={d.norm():.2f}", flush=True)
            deltas_asr[L] = (args.alpha * d).to(dev, torch.bfloat16)
    state = {"mask": None, "asr_i": None}

    def make_hook(L):
        def hook(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            if h.shape[1] == 1:          # decode step: KV cache already carries the shift
                if not args.decode:
                    return None
                h = h.clone()
                h[0, 0] = h[0, 0] + (deltas_asr.get(L) if pos == "asr" else deltas[L])
                return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
            h = h.clone()
            if L in deltas:
                mask = state["mask"] if pos != "all" else torch.ones_like(state["mask"])
                h[0, mask] = h[0, mask] + deltas[L]
            if L in deltas_asr:
                h[0, state["asr_i"]] = h[0, state["asr_i"]] + deltas_asr[L]
            return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
        return hook
    handles = []
    for L in layers:
        if L == 0:
            raise SystemExit("L0 (embeddings) not supported; use L>=1")
        handles.append(thinker.model.layers[L - 1].register_forward_hook(make_hook(L)))

    ids = common_ids([src, tgt])[: args.limit] if args.limit else common_ids([src, tgt])
    rs, rt = read_rows(src), read_rows(tgt)
    out = open(args.out, "w", encoding="utf-8")
    hyps, refs_t, refs_s, scripts = [], [], [], []
    t0 = time.time()
    for i, uid in enumerate(ids):
        r = rs[uid]
        wav, _ = librosa.load(str(FLEURS / LANGS[src][0] / "audio" / "test" / r[1]), sr=16000)
        prompt = m._build_text_prompt(context="", force_language=LANGS[tgt][1])
        inputs = m.processor(text=[prompt], audio=[wav], return_tensors="pt", padding=True)
        inputs = inputs.to(dev).to(torch.bfloat16)
        state["mask"] = inputs["input_ids"][0] == audio_tok
        state["asr_i"] = (inputs["input_ids"][0] == asr_tok).nonzero()[-1, 0].item()
        with torch.no_grad():
            o = m.model.generate(**inputs, max_new_tokens=128)
        seq = o.sequences if hasattr(o, "sequences") else o
        hyp = tok.decode(seq[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        sc = script_of(hyp)
        rec = {"id": uid, "hyp": hyp, "script": sc, "ref_tgt": rt[uid][2].strip(), "ref_src": r[2].strip()}
        out.write(json.dumps(rec, ensure_ascii=False) + "\n"); out.flush()
        hyps.append(hyp); refs_t.append(rec["ref_tgt"]); refs_s.append(rec["ref_src"]); scripts.append(sc)
        if (i + 1) % 20 == 0:
            print(f"[steer {src}->{tgt} L{args.layers} a={args.alpha}] {i + 1}/{len(ids)}  {time.time() - t0:.0f}s", flush=True)
    print(f"\n## steer {src}->{tgt} layers={args.layers} alpha={args.alpha} pos={pos} vec={args.vec} decode={args.decode}")
    ts_summary(src, tgt, False, hyps, refs_t, refs_s, scripts)
    for h in handles:
        h.remove()


# --------------------------------------------------------------------------- neurons
def neurons_extract(args):
    """Per-language FFN neuron statistics: for every layer and every intermediate neuron,
    P(act_fn(gate) > 0) and mean activation, over audio positions and over teacher-forced
    transcript positions separately."""
    import librosa
    import torch
    m = load_model()
    thinker = m.model.thinker
    tok = m.processor.tokenizer
    dev = next(thinker.parameters()).device
    audio_tok = thinker.config.audio_token_id
    asr_tok = tok.convert_tokens_to_ids("<asr_text>")
    acts = []
    handles = [layer.mlp.act_fn.register_forward_hook(lambda mod, i, o: acts.append(o.detach()))
               for layer in thinker.model.layers]
    langs = args.langs.split(",")
    ids = common_ids(langs)[: args.limit] if args.limit else common_ids(langs)
    rows = {l: read_rows(l) for l in langs}
    nL, I = len(thinker.model.layers), thinker.model.layers[0].mlp.gate_proj.out_features
    out = {"langs": np.array(langs), "ids": np.array(ids)}
    t0 = time.time()
    for lang in langs:
        st = {r: {"n": 0, "pos": np.zeros((nL, I)), "sum": np.zeros((nL, I))} for r in ("audio", "trans")}
        for i, uid in enumerate(ids):
            r = rows[lang][uid]
            wav, _ = librosa.load(str(FLEURS / LANGS[lang][0] / "audio" / "test" / r[1]), sr=16000)
            prompt = m._build_text_prompt(context="", force_language=LANGS[lang][1]) + r[2].strip()
            inputs = m.processor(text=[prompt], audio=[wav], return_tensors="pt", padding=True)
            inputs = inputs.to(dev).to(torch.bfloat16)
            acts.clear()
            with torch.no_grad():
                thinker(**inputs)
            ids_ = inputs["input_ids"][0]
            amask = ids_ == audio_tok
            asr_i = (ids_ == asr_tok).nonzero()[-1, 0].item()
            A = torch.stack(acts, 0)[:, 0].float()            # [nL, T, I]
            for region, sel in (("audio", A[:, amask]), ("trans", A[:, asr_i + 1:])):
                st[region]["n"] += sel.shape[1]
                st[region]["pos"] += (sel > 0).sum(1).cpu().numpy()
                st[region]["sum"] += sel.sum(1).cpu().numpy()
            if (i + 1) % 40 == 0:
                print(f"[{lang}] {i + 1}/{len(ids)}  {time.time() - t0:.0f}s", flush=True)
        for region in ("audio", "trans"):
            out[f"{lang}__{region}__p"] = (st[region]["pos"] / st[region]["n"]).astype(np.float32)
            out[f"{lang}__{region}__mean"] = (st[region]["sum"] / st[region]["n"]).astype(np.float32)
        print(f"[{lang}] done", flush=True)
    for h in handles:
        h.remove()
    np.savez(args.out, **out)
    print("saved", args.out)


def select_neurons(z, langs, lang, region, topk, min_p=0.0):
    """language-specific score = p(lang) - max p(other langs). Returns (layer, idx) arrays of top-k."""
    P = np.stack([z[f"{l}__{region}__p"] for l in langs])        # [k, nL, I]
    li = langs.index(lang)
    others = np.delete(P, li, 0).max(0)
    score = P[li] - others
    score[P[li] < min_p] = -1
    flat = np.argsort(score.ravel())[::-1][:topk]
    L, I = np.unravel_index(flat, score.shape)
    return L, I, score.ravel()[flat]


def neurons_steer(args):
    """Deactivate source-language-specific neurons and/or set target-language-specific neurons
    to their target-language mean activation, at every position (prefill and decode)."""
    import json
    import librosa
    import torch
    m = load_model()
    thinker = m.model.thinker
    tok = m.processor.tokenizer
    dev = next(thinker.parameters()).device
    src, tgt = args.pair.split("-")
    z = np.load(args.stats)
    langs = list(z["langs"])
    nL = len(thinker.model.layers)
    off = {L: [] for L in range(nL)}
    on = {L: ([], []) for L in range(nL)}
    if args.mode in ("off", "both"):
        Ls, Is, sc = select_neurons(z, langs, src, args.region, args.topk, args.min_p)
        for L, I in zip(Ls, Is):
            off[L].append(int(I))
        print(f"src-specific ({src}, {args.region}): {len(Ls)} neurons, score range {sc.min():.2f}..{sc.max():.2f}, "
              f"layers {np.bincount(Ls, minlength=nL).tolist()}", flush=True)
    if args.mode in ("on", "both"):
        Ls, Is, sc = select_neurons(z, langs, tgt, args.region, args.topk, args.min_p)
        mean_t = z[f"{tgt}__{args.region}__mean"]
        for L, I in zip(Ls, Is):
            on[L][0].append(int(I)); on[L][1].append(float(mean_t[L, I]) * args.gain)
        print(f"tgt-specific ({tgt}, {args.region}): {len(Ls)} neurons, score range {sc.min():.2f}..{sc.max():.2f}, "
              f"layers {np.bincount(Ls, minlength=nL).tolist()}", flush=True)
    handles = []
    for L, layer in enumerate(thinker.model.layers):
        oi = torch.tensor(off[L], dtype=torch.long, device=dev) if off[L] else None
        ni = torch.tensor(on[L][0], dtype=torch.long, device=dev) if on[L][0] else None
        nv = torch.tensor(on[L][1], dtype=torch.bfloat16, device=dev) if on[L][0] else None
        if oi is None and ni is None:
            continue
        def hook(mod, inp, out, oi=oi, ni=ni, nv=nv):
            out = out.clone()
            if oi is not None:
                out[..., oi] = 0
            if ni is not None:
                out[..., ni] = nv
            return out
        handles.append(layer.mlp.act_fn.register_forward_hook(hook))

    ids = common_ids([src, tgt])[: args.limit] if args.limit else common_ids([src, tgt])
    rs, rt = read_rows(src), read_rows(tgt)
    out = open(args.out, "w", encoding="utf-8")
    hyps, refs_t, refs_s, scripts = [], [], [], []
    t0 = time.time()
    for i, uid in enumerate(ids):
        r = rs[uid]
        wav, _ = librosa.load(str(FLEURS / LANGS[src][0] / "audio" / "test" / r[1]), sr=16000)
        prompt = m._build_text_prompt(context="", force_language=LANGS[tgt][1])
        inputs = m.processor(text=[prompt], audio=[wav], return_tensors="pt", padding=True)
        inputs = inputs.to(dev).to(torch.bfloat16)
        with torch.no_grad():
            o = m.model.generate(**inputs, max_new_tokens=128)
        seq = o.sequences if hasattr(o, "sequences") else o
        hyp = tok.decode(seq[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        sc = script_of(hyp)
        rec = {"id": uid, "hyp": hyp, "script": sc, "ref_tgt": rt[uid][2].strip(), "ref_src": r[2].strip()}
        out.write(json.dumps(rec, ensure_ascii=False) + "\n"); out.flush()
        hyps.append(hyp); refs_t.append(rec["ref_tgt"]); refs_s.append(rec["ref_src"]); scripts.append(sc)
        if (i + 1) % 20 == 0:
            print(f"[neurons {src}->{tgt} {args.mode} k={args.topk}] {i + 1}/{len(ids)}  {time.time() - t0:.0f}s", flush=True)
    print(f"\n## neurons {src}->{tgt} mode={args.mode} region={args.region} topk={args.topk} min_p={args.min_p} gain={args.gain}")
    ts_summary(src, tgt, False, hyps, refs_t, refs_s, scripts)
    for h in handles:
        h.remove()


# --------------------------------------------------------------------------- segswap
def segswap(args):
    """Concatenate k consecutive FLEURS test utterances of the source language (0.3 s silence
    between), decode with the TARGET tag, and count <SEG> tokens in the output. With --pair xx-xx
    (same language) this is the ASR sanity check: does the SEG model still segment?"""
    import json
    import librosa
    import torch
    m = load_model()
    tok = m.processor.tokenizer
    dev = next(m.model.thinker.parameters()).device
    seg_id = tok.convert_tokens_to_ids("<SEG>")
    src, tgt = args.pair.split("-")
    ids = common_ids([src, tgt]) if src != tgt else sorted(read_rows(src), key=int)
    rs, rt = read_rows(src), read_rows(tgt)
    out = open(args.out, "w", encoding="utf-8")
    gap = np.zeros(int(16000 * 0.3), dtype=np.float32)
    n_groups = 0; seg_counts = []; purities = []; placed = 0
    t0 = time.time()
    for g in range(0, min(len(ids), args.limit * args.k if args.limit else len(ids)), args.k):
        grp = ids[g:g + args.k]
        if len(grp) < args.k: break
        wavs = []
        for uid in grp:
            w, _ = librosa.load(str(FLEURS / LANGS[src][0] / "audio" / "test" / rs[uid][1]), sr=16000)
            wavs.append(w)
        wav = np.concatenate(sum(([w, gap] for w in wavs), [])[:-1])
        prompt = m._build_text_prompt(context="", force_language=LANGS[tgt][1])
        inputs = m.processor(text=[prompt], audio=[wav], return_tensors="pt", padding=True).to(dev).to(torch.bfloat16)
        with torch.no_grad():
            o = m.model.generate(**inputs, max_new_tokens=96 * args.k)
        seq = o.sequences if hasattr(o, "sequences") else o
        gen = seq[0, inputs["input_ids"].shape[1]:].tolist()
        n_seg = sum(1 for t in gen if t == seg_id)
        hyp = tok.decode(gen, skip_special_tokens=False).replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()
        letters = [c for c in hyp if c.isalpha()]
        ok = {"en": lambda o: o < 0x250, "ko": lambda o: 0xAC00 <= o <= 0xD7A3,
              "zh": lambda o: 0x4E00 <= o <= 0x9FFF, "ja": lambda o: 0x3040 <= o <= 0x30FF or 0x4E00 <= o <= 0x9FFF}[tgt]
        pur = sum(ok(ord(c)) for c in letters) / len(letters) if letters else 0.0
        rec = {"ids": grp, "hyp": hyp, "n_seg": n_seg, "purity": pur,
               "ref_src": " || ".join(rs[u][2].strip() for u in grp), "ref_tgt": " || ".join(rt[u][2].strip() for u in grp)}
        out.write(json.dumps(rec, ensure_ascii=False) + "\n"); out.flush()
        n_groups += 1; seg_counts.append(n_seg); purities.append(pur)
        if n_seg >= args.k - 1: placed += 1
    from collections import Counter
    print(f"\n## segswap {src}->{tgt} k={args.k} groups={n_groups}")
    print(f"<SEG> per output: {dict(sorted(Counter(seg_counts).items()))}  mean={np.mean(seg_counts):.2f} (expected >= {args.k - 1})")
    print(f"groups with >= k-1 SEG: {placed / max(1, n_groups):.3f}   mean target-script purity: {np.mean(purities):.3f}")


# --------------------------------------------------------------------------- segprobe
def segprobe(args):
    """Where does the <SEG> decision live? Concatenate 2 FLEURS utterances (gap seconds of silence),
    teacher-force 'language X<asr_text>s1 <SEG> s2', and at every transcript position collect the
    hidden state of every layer. Boundary position = the one whose next token is <SEG>.
    Per layer: logit-lens P(<SEG>) and top-1 rate at boundary vs non-boundary positions, and a
    5-fold linear probe (boundary vs not, balanced) AUC. Also the audio-side probe: audio token
    hidden states labelled by whether they fall in the first or second utterance (does z know
    which sentence it is in?)."""
    import librosa
    import torch
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    m = load_model()
    thinker = m.model.thinker
    tok = m.processor.tokenizer
    dev = next(thinker.parameters()).device
    audio_tok = thinker.config.audio_token_id
    asr_tok = tok.convert_tokens_to_ids("<asr_text>")
    seg_id = tok.convert_tokens_to_ids("<SEG>")
    taps = Taps(thinker)
    norm, head = thinker.model.norm, thinker.lm_head
    lang = args.lang
    rows = read_rows(lang); ids = sorted(rows, key=int)
    gap = np.zeros(int(16000 * args.gap), dtype=np.float32)
    Xb, Xn, Xa, ya = [], [], [], []       # boundary feats, non-boundary feats, audio feats, audio labels
    Xp, pp = [], []                        # non-boundary positions whose CURRENT token is punctuation (hard negatives)
    Xcb, pcb, Xcn, pcn = [], [], [], []     # comma positions: with SEG next / without (the real decision)
    pb, pn, top_b, top_n = [], [], [], []  # logit-lens P(SEG) at boundary / non-boundary, top1 hits
    L = None; n = 0
    for g in range(0, len(ids), 2):
        if n >= args.limit: break
        a, b = ids[g], ids[g + 1] if g + 1 < len(ids) else None
        if b is None: break
        w1, _ = librosa.load(str(FLEURS / LANGS[lang][0] / "audio" / "test" / rows[a][1]), sr=16000)
        w2, _ = librosa.load(str(FLEURS / LANGS[lang][0] / "audio" / "test" / rows[b][1]), sr=16000)
        wav = np.concatenate([w1, gap, w2])
        text = rows[a][2].strip() + " <SEG> " + rows[b][2].strip()
        if args.self:
            # use the SEG model's own segmented transcript as the teacher-forced text: boundaries are
            # then wherever IT puts <SEG>, and other punctuation positions are real non-boundaries
            pr0 = m._build_text_prompt(context="", force_language=LANGS[lang][1])
            inp0 = m.processor(text=[pr0], audio=[wav], return_tensors="pt", padding=True).to(dev).to(torch.bfloat16)
            with torch.no_grad():
                o = m.model.generate(**inp0, max_new_tokens=200)
            seq = o.sequences if hasattr(o, "sequences") else o
            text = tok.decode(seq[0, inp0["input_ids"].shape[1]:].tolist(), skip_special_tokens=False)
            text = text.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()
            if "<SEG>" not in text: continue
        prompt = m._build_text_prompt(context="", force_language=LANGS[lang][1]) + text
        inputs = m.processor(text=[prompt], audio=[wav], return_tensors="pt", padding=True).to(dev).to(torch.bfloat16)
        with torch.no_grad():
            thinker(**inputs)
        ids_ = inputs["input_ids"][0]
        asr_i = (ids_ == asr_tok).nonzero()[-1, 0].item()
        H = torch.stack(taps.h, 0)[:, 0]              # [L+1, T, D]
        L = H.shape[0]
        # transcript positions: predict token t+1 from position t, for t in [asr_i, T-2]
        pos = list(range(asr_i, ids_.shape[0] - 1))
        nxt = ids_[asr_i + 1:].tolist()
        with torch.no_grad():
            probs = torch.softmax(head(norm(H[:, pos])).float(), -1)   # [L+1, P, V]
            pseg = probs[..., seg_id].cpu().numpy()                     # [L+1, P]
            top1 = probs.argmax(-1).cpu().numpy()
        feats = H[:, pos].float().cpu().numpy()                        # [L+1, P, D]
        cur = ids_[asr_i:ids_.shape[0] - 1].tolist()                     # current token at each position
        seg_next = set()   # positions j whose next token is <SEG>; if cur[j] is a bare space token, the decision
                           # was taken one step earlier at the word/punctuation token -> use jj = j-1 for features
        for j, t in enumerate(nxt):
            if t == seg_id:
                jj = j - 1 if (j > 0 and tok.decode([cur[j]]).strip() == "") else j
                seg_next.add(jj)
        for j, t in enumerate(nxt):
            curs = tok.decode([cur[j]]).strip()[-1:]
            if j in seg_next:
                Xb.append(feats[:, j]); pb.append(pseg[:, j]); top_b.append(top1[:, j] == seg_id)
                if curs == ",": Xcb.append(feats[:, j]); pcb.append(pseg[:, j])
            elif t != seg_id and tok.decode([cur[j]]).strip() != "":
                Xn.append(feats[:, j]); pn.append(pseg[:, j]); top_n.append(top1[:, j] == seg_id)
                if curs in ",.?!;:":
                    Xp.append(feats[:, j]); pp.append(pseg[:, j])
                if curs == ",": Xcn.append(feats[:, j]); pcn.append(pseg[:, j])
        # audio side: which utterance does each audio token belong to (by time share)
        amask = (ids_ == audio_tok).nonzero()[:, 0].tolist()
        n1 = int(round(len(amask) * (len(w1) + len(gap) / 2) / len(wav)))
        A = H[:, amask].float().cpu().numpy()
        Xa.append(A); ya.append(np.array([0] * n1 + [1] * (len(amask) - n1)))
        n += 1
    Xb, Xn = np.stack(Xb), np.stack(Xn)                 # [Nb, L+1, D], [Nn, L+1, D]
    pb, pn, top_b, top_n = map(np.stack, (pb, pn, top_b, top_n))
    rng = np.random.default_rng(0)
    sel = rng.choice(len(Xn), size=min(len(Xn), 4 * len(Xb)), replace=False)
    Xn_s = Xn[sel]
    Xp = np.stack(Xp) if Xp else np.zeros((0,) + Xb.shape[1:]); pp = np.stack(pp) if len(pp) else np.zeros((0, L))
    print(f"## segprobe lang={lang} gap={args.gap}s self={args.self} pairs={n} boundary positions={len(Xb)} other positions={len(Xn)} (probe uses {len(Xn_s)}), punctuation non-boundary positions={len(Xp)}")
    Xcb = np.stack(Xcb) if Xcb else None; Xcn = np.stack(Xcn) if Xcn else None
    pcb = np.stack(pcb) if len(pcb) else None; pcn = np.stack(pcn) if len(pcn) else None
    print(f"comma positions: with SEG {0 if Xcb is None else len(Xcb)}, without SEG {0 if Xcn is None else len(Xcn)}")
    print("| layer | P(SEG) at boundary | P(SEG) at other punct | P(SEG) elsewhere | top1=SEG at boundary | probe AUC vs all | probe AUC vs punct-only | COMMA: P(SEG) seg/noseg | COMMA probe AUC |")
    print("|---|---|---|---|---|---|---|---|---|")
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    def auc_of(Xpos, Xneg, l):
        X = np.concatenate([Xpos[:, l], Xneg[:, l]]); y = np.array([1] * len(Xpos) + [0] * len(Xneg))
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=300, C=0.1, class_weight="balanced"))
        pr = cross_val_predict(clf, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
        return roc_auc_score(y, pr)
    for l in range(L):
        a1 = auc_of(Xb, Xn_s, l); a2 = auc_of(Xb, Xp, l) if len(Xp) >= 10 else float("nan")
        ok = Xcb is not None and Xcn is not None and len(Xcb) >= 10 and len(Xcn) >= 10
        a3 = auc_of(Xcb, Xcn, l) if ok else float("nan")
        cp = f"{pcb[:, l].mean():.3f}/{pcn[:, l].mean():.3f}" if ok else "nan"
        print(f"| L{l} | {pb[:, l].mean():.3f} | {pp[:, l].mean() if len(pp) else float('nan'):.4f} | {pn[:, l].mean():.4f} | {top_b[:, l].mean():.2f} | {a1:.3f} | {a2:.3f} | {cp} | {a3:.3f} |")
    # (audio-side first/second-utterance probe removed: position alone separates the halves after L1)


# --------------------------------------------------------------------------- seglabel
def seglabel(args):
    """Pseudo-label: for rows of --pair (e.g. en-en) in an SFT jsonl, replace the target text with
    the SEG model's own transcript (keeps <SEG>). Other rows copied unchanged."""
    import json
    import librosa
    import torch
    m = load_model()
    tok = m.processor.tokenizer
    dev = next(m.model.thinker.parameters()).device
    seg_id = tok.convert_tokens_to_ids("<SEG>")
    lang = args.pair.split("-")[1]
    out = open(args.out, "w", encoding="utf-8")
    n = 0; nseg = 0; t0 = time.time()
    for line in open(args.inp, encoding="utf-8"):
        r = json.loads(line)
        if r.get("pair") != args.pair:
            out.write(line); continue
        wav, _ = librosa.load(r["audio"], sr=16000)
        prompt = m._build_text_prompt(context="", force_language=LANGS[lang][1])
        inputs = m.processor(text=[prompt], audio=[wav], return_tensors="pt", padding=True).to(dev).to(torch.bfloat16)
        with torch.no_grad():
            o = m.model.generate(**inputs, max_new_tokens=160)
        seq = o.sequences if hasattr(o, "sequences") else o
        gen = seq[0, inputs["input_ids"].shape[1]:].tolist()
        hyp = tok.decode(gen, skip_special_tokens=False).replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()
        nseg += sum(1 for t in gen if t == seg_id); n += 1
        r["text"] = f"language {LANGS[lang][1]}<asr_text>{hyp}"; r["pseudo"] = True
        out.write(json.dumps(r, ensure_ascii=False) + "\n")
        if n % 50 == 0: print(f"{n} labelled, {nseg} SEG, {time.time() - t0:.0f}s", flush=True)
    print(f"done: {n} rows relabelled with {nseg} <SEG> ({nseg / max(1, n):.2f} per utt) -> {args.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--langs", default="en,ko")
    e.add_argument("--limit", type=int, default=0)
    e.add_argument("--out", required=True)
    a = sub.add_parser("analyze")
    a.add_argument("--npz", required=True)
    g = sub.add_parser("gen")
    g.add_argument("--limit", type=int, default=8)
    t = sub.add_parser("tagswap")
    t.add_argument("--pair", default="en-ko")
    t.add_argument("--limit", type=int, default=0)
    t.add_argument("--sys", action="store_true")
    t.add_argument("--out", required=True)
    u = sub.add_parser("tssum")
    u.add_argument("--pair", required=True)
    u.add_argument("--jsonl", required=True)
    u.add_argument("--sys", action="store_true")
    st = sub.add_parser("steer")
    st.add_argument("--pair", required=True)
    st.add_argument("--npz", required=True)
    st.add_argument("--layers", default="9")
    st.add_argument("--alpha", type=float, default=1.0)
    st.add_argument("--where", default="audio", choices=["audio", "all"])
    st.add_argument("--decode", action="store_true", help="also add delta at every decode step")
    st.add_argument("--pos", default=None, choices=["audio", "asr", "all", "audio+asr"],
                    help="positions to shift (overrides --where). asr = the <asr_text> position, using asr_pos vectors")
    st.add_argument("--vec", default="mean", choices=["mean", "probe"],
                    help="mean = mean difference; probe = LDA direction scaled to the mean gap")
    st.add_argument("--limit", type=int, default=0)
    st.add_argument("--out", required=True)
    ne = sub.add_parser("neurons-extract")
    ne.add_argument("--langs", default="en,ko,ja,zh")
    ne.add_argument("--limit", type=int, default=0)
    ne.add_argument("--out", required=True)
    ns = sub.add_parser("neurons-steer")
    ns.add_argument("--pair", required=True)
    ns.add_argument("--stats", required=True)
    ns.add_argument("--mode", default="both", choices=["off", "on", "both"])
    ns.add_argument("--region", default="trans", choices=["audio", "trans"])
    ns.add_argument("--topk", type=int, default=400)
    ns.add_argument("--min_p", type=float, default=0.0)
    ns.add_argument("--gain", type=float, default=1.0)
    ns.add_argument("--limit", type=int, default=0)
    ns.add_argument("--out", required=True)
    sg = sub.add_parser("segswap")
    sg.add_argument("--pair", required=True); sg.add_argument("--k", type=int, default=2)
    sg.add_argument("--limit", type=int, default=40); sg.add_argument("--out", required=True)
    sp = sub.add_parser("segprobe")
    sp.add_argument("--lang", default="en"); sp.add_argument("--gap", type=float, default=0.3)
    sp.add_argument("--limit", type=int, default=100)
    sp.add_argument("--self", action="store_true", help="teacher-force the SEG model's own segmented transcript")
    sl = sub.add_parser("seglabel")
    sl.add_argument("--inp", required=True); sl.add_argument("--out", required=True); sl.add_argument("--pair", default="en-en")
    args = ap.parse_args()
    {"extract": extract, "analyze": analyze, "gen": gen, "tagswap": tagswap, "tssum": tssum,
     "steer": steer, "neurons-extract": neurons_extract, "neurons-steer": neurons_steer,
     "segswap": segswap, "segprobe": segprobe, "seglabel": seglabel}[args.cmd](args)
