"""CPU streaming simulation: does a langid-triggered slot reset stop cross-language transcription?

A cut-down copy of the qwen3_seg streaming loop on the transformers backend (vLLM streaming is not
available on CPU): 2 s chunks, the whole slot audio re-fed each chunk, prompt prefix = decoded text
up to the last <SEG> (none for the first `UNFIXED` chunks, at most 96 tokens), every new <SEG>
commits, and a slot that decodes to a trailing <SEG> resets with no audio carried. No VAD, no
dedup rules.

The langid is simulated from the reference words: a language switch is known LAG seconds after
the first word of the new language (the timeline mode's median delay). Modes:
  none  - no langid reset (today's vela)
  a     - on a switch, commit the uncommitted tail, reset keeping KEEP s of audio
  b     - on a switch, drop the uncommitted tail, reset keeping KEEP s of audio

usage: uv run --project bench python core/research/lid_module/reset_sim.py \
         --clip scene5:0:45 --modes none a b --out logs/audio-lid/eval/reset_sim.jsonl
"""

import argparse
import json
import os
import re
import sys
import time

sys.path.insert(0, os.getcwd())

from core.utils import env  # noqa: E402

env.load()

import librosa  # noqa: E402
import torch  # noqa: E402

AUDIO = "/home/mobility/STiTy-team/datasets/northstar-260928/audio/{}.wav"
ITEMS = "bench/runs/northstar_ko+en-ko@v4/common_audiolid-timeline/items.jsonl"
SR = 16000
CHUNK = 2.0
UNFIXED = 2
MAX_PREFIX_TOKENS = 96
LAG = 0.7
KEEP = 1.5
SEG = "<SEG>"


def word_lang(w: str) -> str:
    return "ko" if re.search("[가-힣]", w) else "en" if re.search("[A-Za-z]", w) else ""


def switches(words, t0, t1):
    """(known_at, lang) for each change of spoken language inside the clip."""
    out, cur = [], None
    for w in words:
        if not (t0 <= w["start"] < t1):
            continue
        lang = word_lang(w["word"])
        if not lang or lang == cur:
            continue
        if cur is not None:
            out.append((w["start"] + LAG, lang))
        cur = lang
    return out


class Sim:
    def __init__(self, model, base_prompt):
        self.m = model
        self.base = base_prompt
        self.tok = model.processor.tokenizer

    def decode(self, wav, prefix):
        self.m._build_text_prompt = lambda context, force_language: self.base + prefix
        return prefix + self.m._infer_asr_transformers([""], [wav], [None])[0]

    def prefix_of(self, raw, chunk_id):
        if chunk_id < UNFIXED:
            return ""
        last = raw.rfind(SEG)
        if last == -1:
            return ""
        ids = self.tok.encode(raw[: last + len(SEG)])
        start = max(0, len(ids) - MAX_PREFIX_TOKENS)
        return self.tok.decode(ids[start:])

    def run(self, wav, t0, sw, mode):
        commits, raw, chunk_id, done, slot_start = [], "", 0, 0, 0
        pending = list(sw)
        n = int(CHUNK * SR)
        segs = [""]
        for k in range(0, len(wav) - n + 1, n):
            t = t0 + (k + n) / SR
            raw = self.decode(wav[slot_start : k + n], self.prefix_of(raw, chunk_id))
            chunk_id += 1
            body = raw.split("<asr_text>", 1)[-1]
            segs = body.split(SEG)
            for s in segs[done:-1]:
                if s.strip():
                    commits.append({"t": round(t, 1), "text": s.strip(), "why": "seg"})
            done = max(done, len(segs) - 1)
            tail = segs[-1].strip()
            reset = None
            if mode != "none" and pending and pending[0][0] <= t:
                lang = pending.pop(0)[1]
                while pending and pending[0][0] <= t:
                    lang = pending.pop(0)[1]
                reset = ("lid", lang)
            if reset:
                if mode == "a" and tail:
                    commits.append({"t": round(t, 1), "text": tail, "why": "lid"})
                elif mode == "b" and tail:
                    commits.append({"t": round(t, 1), "text": "", "why": f"lid-drop:{tail}"})
                raw, chunk_id, done, segs = "", 0, 0, [""]
                slot_start = max(0, k + n - int(KEEP * SR))
                continue
            if body.rstrip().endswith(SEG) or k + n - slot_start > 90 * SR:
                raw, chunk_id, done, segs = "", 0, 0, [""]
                slot_start = k + n
        if raw and segs[-1].strip():
            commits.append({"t": round(t, 1), "text": segs[-1].strip(), "why": "end"})
        return commits


def norm(t):
    return re.sub(r"[\s\W_]+", "", re.sub(r"<[^>]+>", "", t).lower())


def cer(ref, hyp):
    r, h = norm(ref), norm(hyp)
    d = list(range(len(h) + 1))
    for i, a in enumerate(r, 1):
        prev, d[0] = d[0], i
        for j, b in enumerate(h, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (a != b))
    return d[len(h)] / max(1, len(r))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", nargs="+", required=True, help="scene:start:end")
    ap.add_argument("--modes", nargs="+", default=["none", "a", "b"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from qwen_asr.inference.qwen3_asr import Qwen3ASRModel

    torch.set_num_threads(os.cpu_count() or 8)
    model = Qwen3ASRModel.from_pretrained(
        "Doo12/Qwen3-ASR-1.7B-koen-seg-mix-merged",
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="cuda:0" if torch.cuda.is_available() else "cpu",
        max_new_tokens=160,
    )
    sim = Sim(model, model._build_text_prompt(context="", force_language=None))
    rows = {json.loads(line)["id"]: json.loads(line) for line in open(ITEMS, encoding="utf-8")}
    out = open(a.out, "a", encoding="utf-8")
    for spec in a.clip:
        scene, s0, s1 = spec.split(":")
        t0, t1 = float(s0), float(s1)
        wav, _ = librosa.load(AUDIO.format(scene), sr=SR)
        wav = wav[int(t0 * SR) : int(t1 * SR)]
        words = rows[scene]["reference_alignment"]
        ref = " ".join(w["word"] for w in words if t0 <= w["start"] < t1 - 0.3)
        sw = switches(words, t0, t1)
        for mode in a.modes:
            st = time.time()
            commits = sim.run(wav, t0, sw, mode)
            hyp = " ".join(c["text"] for c in commits)
            rec = {"clip": spec, "mode": mode, "switches": sw, "cer": round(cer(ref, hyp), 4),
                   "sec": round(time.time() - st), "commits": commits, "ref": ref}
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            print(f"{spec} {mode}: cer {rec['cer']} ({rec['sec']}s)", flush=True)
            for c in commits:
                print(f"    {c['t']:6.1f} {c['why']:<5} {c['text']}", flush=True)
    print("RESET_SIM_DONE", flush=True)


if __name__ == "__main__":
    main()
