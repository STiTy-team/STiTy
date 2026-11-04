"""배치 생성이 순차 생성과 같은 출력을 내는지 본다. 2.3 로컬 결과(순차)에서 요청을 골라 배치로 다시 낸다.

    PYTHONPATH=$PWD python evaluation/TranslatorPrompt/scripts/check_batch_equiv.py --n 64 --batch 16
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(HERE.parent / "LongContextMT" / "scripts"))

from tp.quality_prompts import build  # noqa: E402
from lcmt.backends import LocalBf16  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=64)
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--run-id", default="qual-20260930")
args = ap.parse_args()

rows = [json.loads(l) for l in (HERE / "results" / args.run_id / "qwen3.5-4b-bf16" / "quality.jsonl")
        .read_text(encoding="utf-8").splitlines() if l.strip()][:args.n]
m = LocalBf16("qwen3.5-4b-bf16", "Qwen/Qwen3.5-4B", 256)
m.load()
pairs = [build(r["cond"], r["src_lang"], r["tgt_lang"], r["src"]) for r in rows]
runs = []
for _ in range(2):
    out = []
    for k in range(0, len(pairs), args.batch):
        out += [x["raw_output"].strip() for x in m.translate_batch(pairs[k:k + args.batch])]
    runs.append(out)
seq = [r["hyp"] for r in rows]
same_seq = sum(a == b for a, b in zip(runs[0], seq))
same_rep = sum(a == b for a, b in zip(runs[0], runs[1]))
print(f"batch vs sequential: {same_seq}/{len(seq)} identical; batch run1 vs run2: {same_rep}/{len(seq)}")
for a, b, r in zip(runs[0], seq, rows):
    if a != b:
        print(f"  DIFF {r['id']}: seq={b!r}\n        bat={a!r}")
