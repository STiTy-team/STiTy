"""Word-prefix translations by Hy-MT2 (local, no API cost): the "what is known after source word k" reference
for the schedule labeler's invention gate.

For every sentence in sentences.jsonl, source tokens 1..k (whitespace split, k = 1..K) are translated into the target
language. Output: one line per sentence {"sid", "prefixes": [t1..tK]}. Resumable (sids already in --out are skipped,
one flush per sentence). known_sets() turns the prefixes into cumulative content-word stem sets.

usage: PYTHONPATH=. python hy_word_prefixes.py --sentences S.jsonl --out O.jsonl [--batch 64] [--limit 0]"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from schedule_gate import words  # noqa: E402

CHUNK = 64  # sentences per resumable chunk


def prefix_strings(src, lang):
    """Prefix k = first k whitespace tokens joined by one space; one string per token."""
    t = words(src, lang)
    return [" ".join(t[: k + 1]) for k in range(len(t))]


def known_sets(prefixes, tgt):
    """Cumulative union over k of content(prefix_k, tgt); never shrinks as k grows."""
    from build_prefix import content
    out, acc = [], set()
    for p in prefixes:
        acc = acc | content(p, tgt)
        out.append(set(acc))
    return out


def load_model():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from core.translator.local_translator import HyMTTranslator
    name = "tencent/Hy-MT2-1.8B"
    tr = HyMTTranslator(model_name=name, quant="none", context_window=0)
    tok = AutoTokenizer.from_pretrained(name, padding_side="left")
    tok.model_input_names = ["input_ids", "attention_mask"]
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tr._tokenizer = tok
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).to("cuda").eval()
    return torch, tr, tok, model


def translate(jobs, batch, torch, tr, tok, model):
    """jobs: [(text, src_lang, tgt_lang)] -> translations in the same order (length-sorted batches)."""
    res = [None] * len(jobs)
    order = sorted(range(len(jobs)), key=lambda i: len(jobs[i][0]))
    for b in range(0, len(order), batch):
        idx = order[b: b + batch]
        # _build_prompt(text, target_code, source_code, context)
        enc = tok([tr._build_prompt(jobs[i][0], jobs[i][2], jobs[i][1], None) for i in idx], return_tensors="pt",
                  padding=True, add_special_tokens=False).to("cuda")
        with torch.inference_mode():
            o = model.generate(**enc, max_new_tokens=200, do_sample=False, pad_token_id=tok.pad_token_id)
        for i, seq in zip(idx, o[:, enc["input_ids"].shape[1]:]):
            res[i] = tok.decode(seq, skip_special_tokens=True).strip().strip('"').strip()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sentences", default="models/zprobe/data/prefix14/sentences.jsonl")
    ap.add_argument("--out", default="models/zprobe/data/prefix14/hy_prefixes.jsonl")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)["sid"] for l in open(a.out, encoding="utf-8") if l.strip()}
    rows = [json.loads(l) for l in open(a.sentences, encoding="utf-8") if l.strip()]
    if a.limit:
        rows = rows[: a.limit]
    todo = [r for r in rows if r["sid"] not in done]
    print(f"sentences={len(rows)} done={len(done)} todo={len(todo)}", flush=True)
    n_sent = n_pref = 0
    if todo:
        torch, tr, tok, model = load_model()
        t0 = time.time()
        with open(a.out, "a", encoding="utf-8") as out:
            for c in range(0, len(todo), CHUNK):
                chunk = todo[c: c + CHUNK]
                jobs, spans = [], []
                for r in chunk:
                    ps = prefix_strings(r["src"], r["lang"])
                    spans.append((len(jobs), len(ps)))
                    jobs += [(p, r["lang"], r["tgt"]) for p in ps]
                res = translate(jobs, a.batch, torch, tr, tok, model)
                for r, (s, n) in zip(chunk, spans):
                    out.write(json.dumps({"sid": r["sid"], "prefixes": res[s: s + n]}, ensure_ascii=False) + "\n")
                    out.flush()
                    n_pref += n
                    n_sent += 1
                    if n_sent % 500 == 0:
                        print(f"{n_sent}/{len(todo)} sentences {n_pref} prefixes {time.time() - t0:.0f}s", flush=True)
    print(f"HY_WORD_PREFIXES_DONE sentences={n_sent} prefixes={n_pref}", flush=True)


if __name__ == "__main__":
    main()
