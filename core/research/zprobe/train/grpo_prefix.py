"""Step 3 of "make the translation itself stream": GRPO on the append-only streaming policy (an adapter trained on
build_prefix.py rows, e.g. prefix1), so that what it writes early is right and it writes early when it can.

Policy: the prefix adapter merged into the base (--policy_model) + a fresh LoRA (trainable). Reference for the KL term:
the same model with the LoRA disabled, i.e. the prefix adapter as trained.
Rollout per utterance (FLEURS train, en->ko and ko->en with gold references): G sampled streams. Each stream decodes
audio[0:t] for t = --first, --first + --step, ..., end; every decode samples a continuation of the text written so far
and appends it (never revised), as stream_bench.py --mode prefix does at test time.
Reward: R = COMET-DA(final text, reference) - lam * latency - mu * shortfall, latency = mean over output characters of
the audio time at which the character was written, divided by the duration (computation-unaware, 0..1), and
shortfall = max(0, 1 - len(text)/len(reference)) so that stopping mid-sentence is punished beyond what COMET sees.
Each group also gets one greedy full-audio decode as an anchor (reward only, no gradient): truncated streams then sit
below the group mean even when every sampled stream truncated. Utterances whose group scored COMET < 0.7 go into a hard
pool and --hard_frac of every later batch is drawn from it, because collapse cases are otherwise too rare to learn from.
Loss: GRPO advantage (R - mean_G) / (std_G + eps) times the mean log-probability of every sampled continuation token
(EOS included: stopping is the decision to wait), + beta * KL to the reference on the same tokens.
COMET comes from comet_server.py (start it with the COMET env). Smoke: --utts 4 --G 2 --steps 2.

usage: PYTHONPATH=Qwen3-ASR python core/research/zprobe/train/grpo_prefix.py --policy_model models/zprobe/merged/prefix1 --out DIR"""

import argparse, csv, json, os, random, re, sys, time

import numpy as np, requests, soundfile as sf, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import probe_z as P  # noqa: E402
from seg_units import continuation  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--policy_model", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--pairs", nargs="+", default=["en-ko", "ko-en"])
ap.add_argument("--utts", type=int, default=600)
ap.add_argument("--G", type=int, default=4)
ap.add_argument("--steps", type=int, default=150)
ap.add_argument("--batch_utts", type=int, default=2)
ap.add_argument("--lr", type=float, default=1e-5)
ap.add_argument("--lam", type=float, default=0.05)
ap.add_argument("--beta", type=float, default=0.04)
ap.add_argument("--temp", type=float, default=0.8)
ap.add_argument("--top_p", type=float, default=0.95)
ap.add_argument("--step", type=float, default=0.5)
ap.add_argument("--first", type=float, default=1.0)
ap.add_argument("--max_dur", type=float, default=16.0)
ap.add_argument("--lora_r", type=int, default=32)
ap.add_argument("--comet", default="http://127.0.0.1:8777/score")
ap.add_argument("--save_every", type=int, default=25)
ap.add_argument("--mu", type=float, default=0.25)
ap.add_argument("--anchor", type=int, default=1)
ap.add_argument("--hard_frac", type=float, default=0.5)
ap.add_argument(
    "--nu",
    type=float,
    default=0.3,
    help="wrong-script penalty (the output ran in the source language)",
)
ap.add_argument(
    "--nli",
    default="",
    help="NLI model for a contradiction penalty (e.g. vicgalle/xlm-roberta-large-xnli-anli)",
)
ap.add_argument("--rho", type=float, default=0.2, help="weight of P(contradiction | ref, output)")
ap.add_argument(
    "--sil_frac",
    type=float,
    default=0.0,
    help="fraction of utterances rolled out with a silence tail appended - the state where the news hallucinations fire, which plain rollouts never visit",
)
ap.add_argument("--sil_sec", type=float, default=2.0)
ap.add_argument(
    "--kappa",
    type=float,
    default=0.5,
    help="penalty weight on the share of output characters emitted after the real audio ended",
)
ap.add_argument(
    "--noise_frac",
    type=float,
    default=0.0,
    help="fraction of utterances rolled out with a speech-shaped noise tail (a crop from --noise_dir) - the VAD-open state where the torture-set gap emissions fire; a gaussian silence tail never visits it. Drawn before --sil_frac from the same roll",
)
ap.add_argument("--noise_dir", default="models/zprobe/data/noise9x")
ap.add_argument(
    "--kspon_utts",
    type=int,
    default=0,
    help="conversational ko-ko rollouts from the KsponSpeech training jsonl - the register where the passthrough substitutions happen, which FLEURS read speech never visits",
)
ap.add_argument(
    "--koko_cap",
    type=int,
    default=-1,
    help="cap on the FLEURS ko-ko utterances AFTER the even per-pair split (-1 = no cap). The transcription-reward share of the pool is the knob the CER/COMET trade rides on; --pairs splits evenly so this is the only way below one share per pair",
)
ap.add_argument(
    "--kspon_jsonl", default="Qwen3-ASR/finetuning/data/KSponSpeech/train_and_val.jsonl"
)
ap.add_argument("--kspon_wav", default="/home/mobility/stity-datasets/ksponspeech/data/wav16k")
ap.add_argument(
    "--enen_cap",
    type=int,
    default=-1,
    help="cap on the FLEURS en-en utterances after the per-pair split, same as --koko_cap for ko-ko. The en-en verbatim direction had zero training rows in grpocer25 (its transcription pool was Korean only)",
)
ap.add_argument(
    "--dailytalk_utts",
    type=int,
    default=0,
    help="conversational en-en rollouts from the DailyTalk jsonl - the English counterpart of --kspon_utts, same register argument",
)
ap.add_argument("--dailytalk_jsonl", default="Qwen3-ASR/finetuning/data/DailyTalk/train.jsonl")
ap.add_argument("--dailytalk_wav", default="Qwen3-ASR/finetuning/data/DailyTalk/audio")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
random.seed(0)
torch.manual_seed(0)
dev = "cuda"


def wrong_script(text, tgt):
    letters = [c for c in text if c.isalpha()]
    if len(letters) < 4:
        return 0.0
    hangul = sum(1 for c in letters if "가" <= c <= "힣") / len(letters)
    return float(hangul < 0.5) if tgt == "ko" else float(hangul > 0.5) if tgt == "en" else 0.0


from qwen_asr import Qwen3ASRModel  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402

if a.nli:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    nli_tok = AutoTokenizer.from_pretrained(a.nli)
    nli_model = (
        AutoModelForSequenceClassification.from_pretrained(a.nli, dtype=torch.float16)
        .to(dev)
        .eval()
    )
    nli_contra = next(i for i, n in nli_model.config.id2label.items() if "contra" in n.lower())
pol = Qwen3ASRModel.from_pretrained(
    a.policy_model, dtype=torch.bfloat16, device_map="cuda", max_inference_batch_size=1
)
type(pol.model).forward = lambda self, **kw: self.thinker(
    **kw
)  # the outer model only has generate(); logprobs need the thinker
tok = pol.processor.tokenizer
EOS = {151645, 151643, tok.eos_token_id}
pol.model = get_peft_model(
    pol.model,
    LoraConfig(
        r=a.lora_r,
        lora_alpha=2 * a.lora_r,
        lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    ),
)
pol.model.print_trainable_parameters()
opt = torch.optim.AdamW([p for p in pol.model.parameters() if p.requires_grad], lr=a.lr)


def tsv(lang):
    p = P.FLEURS / P.LANGS[lang][0] / "train.tsv"
    return {
        r[0]: r
        for r in csv.reader(open(p, encoding="utf-8"), delimiter="\t", quoting=csv.QUOTE_NONE)
        if len(r) >= 3
    }


pool = []
for pair in a.pairs:
    s, t = pair.split("-")
    rs, rt = tsv(s), tsv(t)
    for sid in sorted(set(rs) & set(rt), key=int):
        wav = P.FLEURS / P.LANGS[s][0] / "audio" / "train" / rs[sid][1]
        if wav.exists() and sf.info(str(wav)).duration <= a.max_dur:
            pool.append(
                {
                    "pair": pair,
                    "tgt": t,
                    "wav": str(wav),
                    "src": rs[sid][2].strip(),
                    "ref": rt[sid][2].strip(),
                }
            )
random.shuffle(pool)
per = a.utts // len(a.pairs)
pool = [u for pair in a.pairs for u in [x for x in pool if x["pair"] == pair][:per]]
if a.koko_cap >= 0:
    pool = [u for u in pool if u["pair"] != "ko-ko"] + [u for u in pool if u["pair"] == "ko-ko"][
        : a.koko_cap
    ]
if a.enen_cap >= 0:
    pool = [u for u in pool if u["pair"] != "en-en"] + [u for u in pool if u["pair"] == "en-en"][
        : a.enen_cap
    ]


def conversational(jsonl, wav_dir, pair, tgt, n):
    rows = []
    for line in open(jsonl, encoding="utf-8"):
        r = json.loads(line)
        wav = os.path.join(wav_dir, os.path.basename(r["audio"]))
        text = re.sub(r"\s+", " ", r["text"].split("<asr_text>")[-1].replace("<SEG>", " ")).strip()
        if os.path.exists(wav) and len(text.split()) >= 4 and sf.info(wav).duration <= a.max_dur:
            rows.append({"pair": pair, "tgt": tgt, "wav": wav, "src": text, "ref": text})
    random.shuffle(rows)
    return rows[:n]


if a.kspon_utts:
    pool += conversational(a.kspon_jsonl, a.kspon_wav, "ko-ko", "ko", a.kspon_utts)
if a.dailytalk_utts:
    pool += conversational(a.dailytalk_jsonl, a.dailytalk_wav, "en-en", "en", a.dailytalk_utts)
print(
    f"{len(pool)} utterances {dict((p, sum(u['pair'] == p for u in pool)) for p in set(u['pair'] for u in pool))}",
    flush=True,
)


def lev(x, y):
    prev = list(range(len(y) + 1))
    for i, cx in enumerate(x, 1):
        cur = [i]
        for j, cy in enumerate(y, 1):
            cur.append(min(prev[j] + 1, cur[-1] + 1, prev[j - 1] + (cx != cy)))
        prev = cur
    return prev[-1]


def verbatim_score(ref, texts):
    """Same-language rollouts are transcription: 1 - CER takes COMET's slot in the reward (COMET is not a
    transcription metric, and the substitutions this targets are exactly what CER counts)."""
    r = "".join(ref.split())
    return [1.0 - min(1.0, lev(r, "".join(t.split())) / max(len(r), 1)) for t in texts]


def inputs_for(wav, base, text):
    return (
        pol.processor(text=[base + text], audio=[wav], return_tensors="pt", padding=True)
        .to(dev)
        .to(torch.bfloat16)
    )


def rollout(u, wav, dur_real=None):
    """One sampled stream: list of decode records and the final text with per-character emission times.
    dur_real marks where the real audio ends when a silence tail was appended; characters emitted past it
    count toward the hallucination share."""
    sr = 16000
    dur = len(wav) / sr
    dur_real = dur_real or dur
    base = pol._build_text_prompt(context="", force_language=P.LANGS[u["tgt"]][1])
    raw, recs, times = "", [], []
    t = min(a.first, dur)
    while True:
        prefix = raw
        inp = inputs_for(wav[: int(t * sr)], base, prefix)
        with torch.no_grad():
            o = pol.model.generate(
                **inp,
                max_new_tokens=120 if t >= dur else 40,
                do_sample=True,
                temperature=a.temp,
                top_p=a.top_p,
                pad_token_id=tok.eos_token_id,
            )
        new = o.sequences[0, inp["input_ids"].shape[1] :].tolist()
        cut = [i for i, x in enumerate(new) if x in EOS]
        if cut:
            new = new[: cut[0] + 1]
        ids = inp["input_ids"][0]
        keep = [x for x in new if x not in EOS]
        add = continuation(tok, ids, torch.tensor(ids.tolist() + keep, device=dev))
        while (
            "\ufffd" in add and keep
        ):  # an unfinished multi-byte character is dropped and decoded again next time
            keep = keep[:-1]
            new = keep
            add = continuation(tok, ids, torch.tensor(ids.tolist() + keep, device=dev))
        add = add.replace("<SEG>", "")
        recs.append({"t": t, "prefix": prefix, "toks": new})
        if add.strip():
            raw += add
            times += [t] * len(add.replace(" ", ""))
        if t >= dur:
            break
        t = min(dur, t + a.step)
    lat = float(np.mean(times) / dur) if times else 1.0
    hall = float(np.mean([t > dur_real + 1e-6 for t in times])) if times else 0.0
    return recs, re.sub(r"\s+", " ", raw).strip(), lat, hall


def offline(u, wav):
    """Greedy full-audio decode: the anchor each group is scored against (reward only, no gradient)."""
    base = pol._build_text_prompt(context="", force_language=P.LANGS[u["tgt"]][1])
    inp = inputs_for(wav, base, "")
    with torch.no_grad():
        o = pol.model.generate(
            **inp, max_new_tokens=160, do_sample=False, pad_token_id=tok.eos_token_id
        )
    ids = [x for x in o.sequences[0, inp["input_ids"].shape[1] :].tolist() if x not in EOS]
    return re.sub(r"\s+", " ", tok.decode(ids).replace("<SEG>", "")).strip()


noise_wavs = []
if a.noise_frac > 0:
    import glob

    noise_wavs = sorted(glob.glob(os.path.join(a.noise_dir, "*.wav")))
    assert noise_wavs, f"--noise_frac > 0 but no wavs in {a.noise_dir}"


def noise_tail(n):
    y, _ = sf.read(random.choice(noise_wavs), dtype="float32")
    if y.ndim > 1:
        y = y.mean(1)
    if len(y) < n:
        y = np.tile(y, n // len(y) + 1)
    off = random.randrange(len(y) - n + 1)
    return y[off : off + n]


log = open(os.path.join(a.out, "train.log"), "a")
t0 = time.time()
hard = []
for step in range(1, a.steps + 1):
    opt.zero_grad()
    stats = []
    nh = min(len(hard), int(a.batch_utts * a.hard_frac))
    for u in random.sample(hard, nh) + random.sample(pool, a.batch_utts - nh):
        wav, _ = sf.read(u["wav"], dtype="float32")
        dur_real = len(wav) / 16000
        tail_roll = random.random()
        if tail_roll < a.noise_frac:
            wav = np.concatenate([wav, noise_tail(int(a.sil_sec * 16000))])
        elif tail_roll < a.noise_frac + a.sil_frac:
            wav = np.concatenate(
                [
                    wav,
                    np.random.default_rng(step)
                    .normal(0.0, 0.0015, int(a.sil_sec * 16000))
                    .astype(np.float32),
                ]
            )
        pol.model.eval()
        outs = [rollout(u, wav, dur_real) for _ in range(a.G)]
        texts = [txt for _, txt, _, _ in outs] + ([offline(u, wav)] if a.anchor else [])
        lats = [lat for _, _, lat, _ in outs] + ([1.0] if a.anchor else [])
        halls = np.array([h for _, _, _, h in outs] + ([0.0] if a.anchor else []))
        if u["pair"].split("-")[0] == u["tgt"]:
            comet = verbatim_score(u["ref"], texts)
        else:
            comet = requests.post(
                a.comet,
                json={
                    "items": [{"src": u["src"], "mt": t, "ref": u["ref"]} for t in texts],
                    "model": "da",
                },
                timeout=600,
            ).json()["scores"]
        short = np.array([max(0.0, 1.0 - len(t) / max(len(u["ref"]), 1)) for t in texts])
        wrong = np.array([wrong_script(t, u["pair"].split("-")[1]) for t in texts])
        contra = np.zeros(len(texts))
        if a.nli:
            with torch.no_grad():
                for k in range(0, len(texts), 8):
                    toks = nli_tok(
                        [u["ref"]] * len(texts[k : k + 8]),
                        texts[k : k + 8],
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=256,
                    ).to(dev)
                    probs = torch.softmax(nli_model(**toks).logits.float(), -1)
                    contra[k : k + 8] = probs[:, nli_contra].cpu().numpy()
        R = (
            np.array(comet)
            - a.lam * np.array(lats)
            - a.mu * short
            - a.nu * wrong
            - a.rho * contra
            - a.kappa * halls
        )
        # the anchor stays out of the advantage: it is off-policy and nearly always on top, so including it makes
        # every sampled rollout negative on average and the gradient only ever suppresses (measured: policy collapse).
        adv = (R[: a.G] - R[: a.G].mean()) / (R[: a.G].std() + 1e-6)
        # it picks the hard pool instead: rollouts failed but the same model solves it offline = policy-fixable,
        # not a mishear. An utterance graduates out of the pool once its rollouts pass.
        mc = float(np.mean(comet[: a.G]))
        if a.anchor and mc < 0.7 and comet[a.G] >= 0.75 and u not in hard:
            hard.append(u)
            if len(hard) > 200:
                hard.pop(0)
        if mc >= 0.7 and u in hard:
            hard.remove(u)
        pol.model.train()
        base = pol._build_text_prompt(context="", force_language=P.LANGS[u["tgt"]][1])
        ntok = sum(len(r["toks"]) for recs, _, _, _ in outs for r in recs) or 1
        for g, (recs, _, _, _) in enumerate(outs):
            if abs(adv[g]) < 1e-6:
                continue
            for r in recs:
                if not r["toks"]:
                    continue
                inp = inputs_for(wav[: int(r["t"] * 16000)], base, r["prefix"])
                L = inp["input_ids"].shape[1]
                toks = torch.tensor([r["toks"]], device=dev)
                full = torch.cat([inp["input_ids"], toks], 1)
                am = torch.ones_like(full)
                kw = {k: v for k, v in inp.items() if k not in ("input_ids", "attention_mask")}
                lp = torch.log_softmax(
                    pol.model(input_ids=full, attention_mask=am, **kw)
                    .logits[0, L - 1 : -1]
                    .float(),
                    -1,
                )
                lp = lp.gather(1, toks[0][:, None])[:, 0]
                with torch.no_grad(), pol.model.disable_adapter():
                    rl = torch.log_softmax(
                        pol.model(input_ids=full, attention_mask=am, **kw)
                        .logits[0, L - 1 : -1]
                        .float(),
                        -1,
                    )
                    rl = rl.gather(1, toks[0][:, None])[:, 0]
                kl = torch.exp(rl - lp) - (rl - lp) - 1
                loss = (-(float(adv[g]) * lp) + a.beta * kl).sum() / ntok / a.batch_utts
                loss.backward()
        stats.append(
            {
                "pair": u["pair"],
                "R": R.round(3).tolist(),
                "comet": [round(c, 3) for c in comet],
                "lat": [round(l, 3) for l in lats],
                "text": outs[int(np.argmax(R[: a.G]))][1][:120],
            }
        )
    torch.nn.utils.clip_grad_norm_([p for p in pol.model.parameters() if p.requires_grad], 1.0)
    opt.step()
    rec = {
        "step": step,
        "t": round(time.time() - t0),
        "meanR": float(np.mean([np.mean(s["R"][: a.G]) for s in stats])),
        "meanC": float(np.mean([np.mean(s["comet"][: a.G]) for s in stats])),
        "meanLat": float(np.mean([np.mean(s["lat"][: a.G]) for s in stats])),
        "hard": len(hard),
        "stats": stats,
    }
    log.write(json.dumps(rec, ensure_ascii=False) + "\n")
    log.flush()
    print(
        f"step {step} R={rec['meanR']:.3f} COMET={rec['meanC']:.3f} lat={rec['meanLat']:.3f} {rec['t']}s",
        flush=True,
    )
    if step % a.save_every == 0 or step == a.steps:
        pol.model.save_pretrained(os.path.join(a.out, f"adapter_step{step}"))
print("GRPO_DONE", flush=True)
