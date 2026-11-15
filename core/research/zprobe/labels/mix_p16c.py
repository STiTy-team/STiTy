"""prefix16c training rows: the prefix16 rows (all-해요체 labels, no rescued waits, catch-up heavy, empty <= 20%) mixed
with the carried prefix13tr corpus (the v8b/v10 lineage prefix10f came from: phrase-by-phrase labels written with the
whole sentence in view, 해요체 throughout) and the prefix15-mtpl conversational pseudo-label rows (Hy-MT2 translations
of DailyTalk and KsponSpeech).

  carried:  en-ko and ko-en rows of every source, en-ko covost capped (--cap en-ko/covost=N), x->en rows of other
            sources (zh-en, ja-en) capped together (--cap x-en=N) as extra English-target supervision; pairs whose
            target is neither ko nor en are left out (brig decodes ko and en only)
  mtpl:     rows with a 합니다체 ending in cond or text are left out (the rest is 해요체 or English), the 4090 silence
            path is remapped, and empty rows are capped at --mtpl-empty of the kept mtpl rows (they are 55% raw - the
            prefix15 wait-heavy shape that made the model defer)

usage: .venv/bin/python core/research/zprobe/labels/mix_p16c.py --p16 TRAIN --p16-val VAL --carried TRAIN13TR \
         --mtpl ROWS --mtpl-val ROWS_VAL --out models/zprobe/data/train_prefix16c.jsonl --val-out ..."""
import argparse
import json
import random
import re

FORMAL = re.compile(r"(니다|니까|십시오)[.?!…\"'”’]*(\s|$)")
SIL_4090 = "/home/skkai/STiTy-wt/mtpl/models/zprobe/data/silence_tail_4s.wav"
KO_EN = {"en-ko", "ko-en"}


def is_formal(r: dict) -> bool:
    return r["pair"].endswith("-ko") and bool(FORMAL.search((r.get("cond") or "") + " " + r["text"]))


def is_end_dump(r: dict) -> bool:
    """A row that emits text on the full recording with no silence tail: 'say it all once the utterance is over'.
    prefix16c showed these (14.5% of its emit rows, 2% in the prefix16 part) pull the model into waiting."""
    a = r["audio"]
    return bool(r["text"].strip()) and "|" not in a and "#end=" not in a


def cap_empty(rows: list[dict], share: float, rng) -> list[dict]:
    full = [r for r in rows if r["text"].strip()]
    empty = [r for r in rows if not r["text"].strip()]
    n = int(len(full) * share / (1 - share)) if share < 1 else len(empty)
    rng.shuffle(empty)
    return full + empty[:n]


def remap(r: dict, sil: str) -> dict:
    return {**r, "audio": "|".join(p.replace(SIL_4090, sil) for p in r["audio"].split("|"))}


def source_of(audio: str) -> str:
    a = audio.lower()
    for k in ("covost", "fleurs", "dailytalk", "kspon", "aihub", "kosp2e"):
        if k in a:
            return k
    return "other"


def pick_carried(rows: list[dict], caps: dict, rng) -> list[dict]:
    groups = {}
    for r in rows:
        tgt = r["pair"].split("-")[1]
        if r["pair"] in KO_EN:
            key = f"{r['pair']}/{source_of(r['audio'].split('|')[0])}"
        elif tgt == "en":
            key = "x-en"
        else:
            continue
        groups.setdefault(key, []).append(r)
    out = []
    for key, g in groups.items():
        if key in caps and len(g) > caps[key]:
            g = rng.sample(g, caps[key])
        out += g
    return out


def read(path):
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p16", required=True)
    ap.add_argument("--p16-val", required=True)
    ap.add_argument("--carried", required=True)
    ap.add_argument("--mtpl", required=True)
    ap.add_argument("--mtpl-val", required=True)
    ap.add_argument("--sil-wav", default="models/zprobe/data/silence_tail_4s.wav")
    ap.add_argument("--cap", action="append", default=[], metavar="KEY=N", help="en-ko/covost=15000, x-en=10000")
    ap.add_argument("--mtpl-empty", type=float, default=0.25)
    ap.add_argument("--mtpl-val-n", type=int, default=300)
    ap.add_argument("--out", required=True)
    ap.add_argument("--val-out", required=True)
    ap.add_argument("--seed", type=int, default=16)
    ap.add_argument("--drop-end-dumps", action="store_true", help="leave out end-of-utterance dump rows of the carried and mtpl corpora")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    import os
    sil = os.path.abspath(a.sil_wav)
    caps = {k: int(v) for k, v in (c.split("=") for c in a.cap)}

    p16 = read(a.p16)
    carried = pick_carried(read(a.carried), caps, rng)
    raw = read(a.mtpl)
    kept = [remap(r, sil) for r in raw if not is_formal(r)]
    if a.drop_end_dumps:
        n0, m0 = len(carried), len(kept)
        carried = [r for r in carried if not is_end_dump(r)]
        kept = [r for r in kept if not is_end_dump(r)]
        print(f"drop-end-dumps: carried {n0} -> {len(carried)}, mtpl {m0} -> {len(kept)}")
    mtpl = cap_empty(kept, a.mtpl_empty, rng)
    train = p16 + carried + mtpl
    rng.shuffle(train)
    mval = [remap(r, sil) for r in read(a.mtpl_val) if not is_formal(r)]
    val = read(a.p16_val) + rng.sample(mval, min(a.mtpl_val_n, len(mval)))

    with open(a.out, "w", encoding="utf-8") as f:
        for r in train:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(a.val_out, "w", encoding="utf-8") as f:
        for r in val:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def share(rs):
        return sum(1 for r in rs if not r["text"].strip()) / max(1, len(rs))
    print(f"p16 {len(p16)} (empty {share(p16):.2f}); carried {len(carried)} (empty {share(carried):.2f}); "
          f"mtpl raw {len(raw)} -> no-formal {len(kept)} -> kept {len(mtpl)} (empty {share(mtpl):.2f})")
    by = {}
    for r in train:
        by[r["pair"]] = by.get(r["pair"], 0) + 1
    print(f"train {len(train)} (empty {share(train):.2f}) by pair {by}; val {len(val)}")
    print("MIX_P16C_DONE")


if __name__ == "__main__":
    main()
