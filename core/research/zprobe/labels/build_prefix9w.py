"""Labels v9w = v8b with explicit terminal supervision: every label ends with the literal <WAIT> marker.

The EOS-stuck / forced-flush hallucination pair traces to one supervision artifact: every training row ends,
so EOS doubles as "sentence looks complete" (mid-stream, wrong) and "nothing more to say" (right). v9w splits
the meanings: the row's terminal token sequence becomes <WAIT> ("nothing to add *now* - evidence short, or
silence"), and EOS only ever follows it. At inference the decoder strips <WAIT>, emits nothing for a
WAIT-only step, and the stall flush + min_new floor (the hallucination manufactory over non-speech, see
torture-261105) can be removed outright - silence becomes a legal output instead of a forbidden one.

<WAIT> is a literal string, not a new vocabulary token, so the prefix8b LoRA-only recipe (frozen embeddings)
carries over unchanged - the only variable that moves is the corpus.

usage: python core/research/zprobe/labels/build_prefix9w.py"""
import argparse
import json

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="models/zprobe/data/train_prefix8b.jsonl")
ap.add_argument("--val", default="models/zprobe/data/val_big_local.jsonl")
ap.add_argument("--out", default="models/zprobe/data/train_prefix9w.jsonl")
ap.add_argument("--out-val", default="models/zprobe/data/val_prefix9w.jsonl")
a = ap.parse_args()

WAIT = "<WAIT>"


def convert(src, dst):
    n = 0
    with open(dst, "w") as out:
        for line in open(src):
            row = json.loads(line)
            row["text"] = row["text"].rstrip() + WAIT
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    print(f"{src} -> {dst}: {n} rows")


convert(a.base, a.out)
convert(a.val, a.out_val)
