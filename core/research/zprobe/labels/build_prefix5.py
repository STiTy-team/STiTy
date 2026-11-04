"""Labels v5 = v4 plus the two gaps the northstar dual run exposed: same-language passthrough and ko->en volume.

(1) Verbatim rows — a header matching the speech language must mean "transcribe", not "paraphrase" (measured
ko WER 0.79-0.86 on passthrough). aligned.jsonl carries the source transcript and per-phrase end times, and
phrases.jsonl tiles the transcript exactly (checked: 10904/10904), so boundary rows cut the transcript at a
phrase end by non-space character count. en is capped so transcription does not drown the translation rows.
(2) Verbatim pairs joined with <SEG> — the passthrough stream of a dual decode gets the same "sentence done,
keep listening" signal as the translation stream.
(3) Extra rotations of x->en concat pairs — ko-en has only 646 aligned rows; new pairings multiply the
concat data without any new alignment or LLM cost.

usage: python core/research/zprobe/labels/build_prefix5.py --out models/zprobe/data/train_prefix5.jsonl"""
import argparse, json, random, re

NAME = {"en": "English", "ko": "Korean", "ja": "Japanese", "zh": "Chinese"}
SPACED = {"en", "ko"}
VERBATIM = {"en": (3000, 1, 1 / 3), "ko": (10**9, 2, 1.0), "ja": (10**9, 0, 0.0), "zh": (10**9, 0, 0.0)}
PAIR_ROT = {("ko", "ko"): 2, ("en", "en"): 1, ("ko", "en"): 2, ("ja", "en"): 1, ("zh", "en"): 1}
PAIR_CAP = {("en", "en"): 1500}
# non-English recordings are long (ko median 11.3 s), so 18 s rejects most pairs; the cap was a /dev/shm
# limit that no longer applies (cache on disk), and 24 s stays inside what the SFT batch held at 18 s * 2
PAIR_DUR = {("en", "en"): 18.0}

ap = argparse.ArgumentParser()
ap.add_argument("--aligned", default="models/zprobe/data/prefix3/aligned.jsonl")
ap.add_argument("--phrases", default="models/zprobe/data/prefix3/phrases.jsonl")
ap.add_argument("--base", default="models/zprobe/data/train_prefix4.jsonl")
ap.add_argument("--out", default="models/zprobe/data/train_prefix5.jsonl")
ap.add_argument("--max_dur", type=float, default=18.0)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
rng = random.Random(a.seed)
nsp = lambda s: re.sub(r"\s+", "", s)


def text_of(adds):
    return " ".join(x.strip() for x in adds if x.strip())


def clip(r, t):
    return f"{r['audio']}#end={t:.3f}" if t < r["dur"] - 0.05 else r["audio"]


def cut_src(src, phrases, i):
    want = len(nsp("".join(phrases[: i + 1])))
    seen = 0
    for pos, ch in enumerate(src):
        if not ch.isspace():
            seen += 1
        if seen == want:
            return src[: pos + 1].strip()
    return src.strip()


phr = {}
for l in open(a.phrases, encoding="utf-8"):
    r = json.loads(l)
    phr[r["sid"]] = r.get("phrases") or []

rows = [json.loads(l) for l in open(a.aligned, encoding="utf-8")
        if (x := json.loads(l)) and x["adds"] and x["ends"] and text_of(x["adds"])]
by_lang, by_dir = {}, {}
for r in rows:
    p = phr.get(r["sid"])
    if p and len(p) == len(r["ends"]) and nsp("".join(p)) == nsp(r["src"]):
        by_lang.setdefault(r["lang"], []).append(r)
    by_dir.setdefault((r["lang"], r["tgt"]), []).append(r)

out, counts = [], {}


def add(kind, row):
    out.append(row)
    counts[kind] = counts.get(kind, 0) + 1


for lang, group in sorted(by_lang.items()):
    cap, n_boundary, frac = VERBATIM[lang]
    rng.shuffle(group)
    for r in group[:cap]:
        head = f"language {NAME[lang]}<asr_text>"
        add(f"verbatim_{lang}", {"audio": r["audio"], "text": head + r["src"].strip(), "pair": f"{lang}-{lang}"})
        if n_boundary and rng.random() < frac and len(r["ends"]) >= 3:
            p = phr[r["sid"]]
            for i in rng.sample(range(len(p) - 1), min(n_boundary, len(p) - 1)):
                add(f"verbatim_{lang}_cut", {"audio": clip(r, r["ends"][i]),
                                             "text": head + cut_src(r["src"], p, i), "pair": f"{lang}-{lang}"})

seen_pairs = set()
for (lang, tgt), rot in sorted(PAIR_ROT.items()):
    same = lang == tgt
    group = by_lang.get(lang, []) if same else by_dir.get((lang, tgt), [])
    if same and lang == "en":
        group = group[: VERBATIM["en"][0]]
    made = 0
    max_dur = PAIR_DUR.get((lang, tgt), 24.0)
    for k in range(rot):
        g = group[:]
        if k == 0:
            g.sort(key=lambda r: r["dur"])  # short with short, so long recordings still find a partner
        else:
            random.Random(a.seed + 1 + k).shuffle(g)
        for A, B in zip(g[::2], g[1::2]):
            key = frozenset((A["id"], B["id"]))
            if A["dur"] + B["dur"] > max_dur or key in seen_pairs:
                continue
            if (lang, tgt) in PAIR_CAP and made >= PAIR_CAP[(lang, tgt)]:
                break
            seen_pairs.add(key)
            made += 1
            head = f"language {NAME[tgt]}<asr_text>"
            a_txt = A["src"].strip() if same else text_of(A["adds"])
            b_txt = B["src"].strip() if same else text_of(B["adds"])
            t_b = max(0.3, B["ends"][0] * 0.5)
            kind = ("verbatim_pair_" if same else "xen_pair_") + f"{lang}-{tgt}"
            add(kind, {"audio": f"{A['audio']}|{clip(B, t_b)}", "text": f"{head}{a_txt} <SEG>", "pair": f"{lang}-{tgt}"})
            add(kind, {"audio": f"{A['audio']}|{B['audio']}", "text": f"{head}{a_txt} <SEG> {b_txt}", "pair": f"{lang}-{tgt}"})

base = [l.rstrip("\n") for l in open(a.base, encoding="utf-8")]
with open(a.out, "w", encoding="utf-8") as f:
    for l in base:
        f.write(l + "\n")
    for r in out:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"{a.out}: {len(base)} base + {len(out)} new")
for k in sorted(counts):
    print(f"  {k}: {counts[k]}")
