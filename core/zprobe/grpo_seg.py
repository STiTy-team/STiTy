"""Stage B: GRPO on the SEG model's source-side <SEG> policy, reward = per-target future-blind
translation quality (COMET via comet_server) minus lambda * latency.
Policy model: SEG model + fresh LoRA (trainable) in ASR mode with a target tag in the system prompt
("segment for Korean"). Translator: a frozen copy (SEG model + translation adapter merged).
Rollout per utterance: G sampled English transcripts with <SEG>; boundaries timed by the forced
aligner; each segmentation translated incrementally (audio prefix + committed target prefix);
reward R = COMET(concat, ref) - lam * mean_seg_dur - pen * [no boundary and dur > max_seg].
GRPO: advantage = (R - mean_G) / (std_G + eps); loss = -sum_t adv * logprob(token_t) / len over the
sampled transcript tokens (all positions: every token is a SEG-or-not decision), + beta * KL to the
frozen reference (approximated with the policy's own logprobs under the base weights via adapter
disable). Smoke: --utts 4 --G 4 --steps 2."""
import argparse, json, os, random, re, sys, time, requests
import numpy as np, librosa, torch
sys.path.insert(0, os.environ.get("PROBE_Z_DIR", os.path.dirname(os.path.abspath(__file__)))); import probe_z as P
from qwen_asr import Qwen3ASRModel, Qwen3ForcedAligner
from peft import LoraConfig, get_peft_model, PeftModel
ap = argparse.ArgumentParser()
ap.add_argument("--seg_model", required=True); ap.add_argument("--trans_adapter", required=True)
ap.add_argument("--tgt", default="ko"); ap.add_argument("--split", default="train"); ap.add_argument("--utts", type=int, default=200)
ap.add_argument("--G", type=int, default=8); ap.add_argument("--steps", type=int, default=100); ap.add_argument("--batch_utts", type=int, default=2)
ap.add_argument("--lr", type=float, default=2e-5); ap.add_argument("--lam", type=float, default=0.02); ap.add_argument("--max_seg", type=float, default=6.0); ap.add_argument("--pen", type=float, default=0.1)
ap.add_argument("--beta", type=float, default=0.02); ap.add_argument("--temp", type=float, default=1.0); ap.add_argument("--comet", default="http://127.0.0.1:8777/score")
ap.add_argument("--out", required=True); ap.add_argument("--lora_r", type=int, default=32)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); random.seed(0); torch.manual_seed(0)
dev = "cuda"
# --- translator (frozen): SEG model + translation adapter merged
os.environ["PROBE_MODEL"] = a.seg_model; os.environ["PROBE_ADAPTER"] = a.trans_adapter
P.MODEL = a.seg_model; P.ADAPTER = a.trans_adapter
trans = P.load_model(); ttok = trans.processor.tokenizer
# --- policy: SEG model + new LoRA
pol = Qwen3ASRModel.from_pretrained(a.seg_model, dtype=torch.bfloat16, device_map="cuda", max_inference_batch_size=1)
ptok = pol.processor.tokenizer; seg_id = ptok.convert_tokens_to_ids("<SEG>")
lcfg = LoraConfig(r=a.lora_r, lora_alpha=2 * a.lora_r, lora_dropout=0.0, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
pol.model = get_peft_model(pol.model, lcfg); pol.model.print_trainable_parameters()
opt = torch.optim.AdamW([p for p in pol.model.parameters() if p.requires_grad], lr=a.lr)
aligner = Qwen3ForcedAligner.from_pretrained("Qwen/Qwen3-ForcedAligner-0.6B", dtype=torch.bfloat16, device_map="cuda")
import csv
def read_rows(lang):
    pth = P.FLEURS / P.LANGS[lang][0] / f"{a.split}.tsv"; d = {}
    for r in csv.reader(open(pth, encoding="utf-8"), delimiter="\t"):
        if len(r) >= 6: d.setdefault(r[0], r)
    return d
rows_en, rows_t = read_rows("en"), read_rows(a.tgt)
ids = [u for u in sorted(set(rows_en) & set(rows_t), key=int) if (P.FLEURS / "en_us" / "audio" / a.split / rows_en[u][1]).exists()][: a.utts]
print(f"{len(ids)} utterances with audio", flush=True)
ctx = f"Segment for {P.LANGS[a.tgt][1]} interpretation."     # target conditioning in the system prompt
def pol_prompt(): return pol._build_text_prompt(context=ctx, force_language="English")
def sample_transcripts(wav, G):
    inp = pol.processor(text=[pol_prompt()] * G, audio=[wav] * G, return_tensors="pt", padding=True).to(dev).to(torch.bfloat16)
    with torch.no_grad():
        o = pol.model.generate(**inp, max_new_tokens=160, do_sample=True, temperature=a.temp, top_p=1.0, pad_token_id=ptok.eos_token_id)
    seq = o.sequences if hasattr(o, "sequences") else o
    L = inp["input_ids"].shape[1]
    outs = []
    for g in range(G):
        toks = seq[g, L:].tolist()
        if ptok.eos_token_id in toks: toks = toks[: toks.index(ptok.eos_token_id) + 1]
        outs.append(toks)
    return inp, outs
def trans_gen(wav, prefix):
    prompt = trans._build_text_prompt(context="", force_language=P.LANGS[a.tgt][1]) + prefix
    inp = trans.processor(text=[prompt], audio=[wav], return_tensors="pt", padding=True).to(dev).to(torch.bfloat16)
    with torch.no_grad(): o = trans.model.generate(**inp, max_new_tokens=160)
    seq = o.sequences if hasattr(o, "sequences") else o
    txt = ttok.decode(seq[0, inp["input_ids"].shape[1]:].tolist(), skip_special_tokens=False)
    return re.sub(r"\s+", " ", txt.replace("<|im_end|>", "").replace("<|endoftext|>", "").replace("<SEG>", " ")).strip()
def bounds_of(wav, toks):
    text = ptok.decode(toks, skip_special_tokens=False).replace("<|im_end|>", "").replace("<|endoftext|>", "")
    parts = [p.strip() for p in text.split("<SEG>") if p.strip()]
    if not parts: return [], text
    plain = " ".join(parts); dur = len(wav) / 16000
    try: items = aligner.align((wav, 16000), plain, "English")[0]
    except Exception: return [], text
    items = [(float(it.end_time)) for it in items]; unit = 1000.0 if items and items[-1] > 60 else 1.0
    bs, cum = [], 0
    for wpp in [len(p.split()) for p in parts[:-1]]:
        cum += wpp
        if cum - 1 < len(items): bs.append(items[cum - 1] / unit)
    return [b for b in bs if 0.3 < b < dur - 0.3], text
def reward(wav, bs, ref_src, ref_tgt):
    dur = len(wav) / 16000; cuts = sorted(bs) + [dur]; committed = ""; prev = 0.0; durs = []
    for t in cuts:
        committed = (committed + " " + trans_gen(wav[: int(t * 16000)], committed)).strip(); durs.append(t - prev); prev = t
    return committed, float(np.mean(durs)), max(durs)
log = open(os.path.join(a.out, "train.log"), "a"); step = 0; t0 = time.time()
while step < a.steps:
    batch = random.sample(ids, a.batch_utts); opt.zero_grad(); stats = []
    for uid in batch:
        r = rows_en[uid]; wav, _ = librosa.load(str(P.FLEURS / "en_us" / "audio" / a.split / r[1]), sr=16000)
        inp, outs = sample_transcripts(wav, a.G)
        items, meta = [], []
        for toks in outs:
            bs, text = bounds_of(wav, toks); concat, msd, mxd = reward(wav, bs, r[2].strip(), rows_t[uid][2].strip())
            items.append({"src": r[2].strip(), "mt": concat, "ref": rows_t[uid][2].strip()}); meta.append((bs, msd, mxd, text))
        comet = requests.post(a.comet, json={"items": items, "model": "da"}, timeout=600).json()["scores"]
        R = np.array([c - a.lam * msd - (a.pen if mxd > a.max_seg else 0.0) for c, (bs, msd, mxd, _) in zip(comet, meta)])
        adv = (R - R.mean()) / (R.std() + 1e-6)
        # policy gradient on sampled tokens
        for g, toks in enumerate(outs):
            if abs(adv[g]) < 1e-6 or len(toks) == 0: continue
            full = torch.cat([inp["input_ids"][g : g + 1], torch.tensor([toks], device=dev)], 1)
            am = torch.ones_like(full)
            kw = {k: v[g : g + 1] for k, v in inp.items() if k not in ("input_ids", "attention_mask")}
            logits = pol.model(input_ids=full, attention_mask=am, **kw).logits[0, inp["input_ids"].shape[1] - 1 : -1].float()
            lp = torch.log_softmax(logits, -1).gather(1, torch.tensor(toks, device=dev)[:, None])[:, 0]
            with torch.no_grad(), pol.model.disable_adapter():
                ref_logits = pol.model(input_ids=full, attention_mask=am, **kw).logits[0, inp["input_ids"].shape[1] - 1 : -1].float()
                ref_lp = torch.log_softmax(ref_logits, -1).gather(1, torch.tensor(toks, device=dev)[:, None])[:, 0]
            kl = (torch.exp(ref_lp - lp) - (ref_lp - lp) - 1).mean()
            loss = (-(float(adv[g]) * lp.mean()) + a.beta * kl) / (a.G * a.batch_utts)
            loss.backward()
        stats.append({"uid": uid, "R": R.round(3).tolist(), "comet": [round(c, 3) for c in comet], "nseg": [len(b) + 1 for b, _, _, _ in meta], "msd": [round(m, 2) for _, m, _, _ in meta]})
    torch.nn.utils.clip_grad_norm_([p for p in pol.model.parameters() if p.requires_grad], 1.0); opt.step(); step += 1
    rec = {"step": step, "t": round(time.time() - t0), "meanR": float(np.mean([np.mean(s["R"]) for s in stats])), "meanC": float(np.mean([np.mean(s["comet"]) for s in stats])), "mean_nseg": float(np.mean([np.mean(s["nseg"]) for s in stats])), "stats": stats}
    log.write(json.dumps(rec, ensure_ascii=False) + "\n"); log.flush()
    print(f"step {step} R={rec['meanR']:.3f} C={rec['meanC']:.3f} nseg={rec['mean_nseg']:.2f} {rec['t']}s", flush=True)
    if step % 20 == 0 or step == a.steps:
        pol.model.save_pretrained(os.path.join(a.out, f"adapter_step{step}"))
print("done")
