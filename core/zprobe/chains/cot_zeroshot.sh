#!/bin/bash
# Zero-shot transcribe-then-translate with big2 (no training), two prompt variants, on the three directions that have
# big2 direct-translation latency (big2_t_*). COMET vs direct big2 and the Hy cascade, plus latency.
#   tmux new-session -d -s cot0 -c /home/mobility/STiTy "bash core/zprobe/chains/cot_zeroshot.sh > core/zprobe/chains/cot_zeroshot.log 2>&1"
cd /home/mobility/STiTy
export PYTHONPATH=Qwen3-ASR
PY=.venv/bin/python; CPY=.venv-tpm/bin/python
D=models/zprobe/data; H=$D/hy; LOG=models/zprobe/logs
P="en-ko ko-en zh-ko"
mark() { echo "$(date +%T) MARK $1"; }
lines() { [ -f "$1" ] && wc -l < "$1" || echo 0; }
mark "1 decode"
for v in inline turn; do for p in $P; do
  [ "$(lines $H/cot${v}_$p.jsonl)" -eq "$(lines $D/ev_big2_$p.jsonl)" ] && continue
  PROBE_ADAPTER=models/zprobe/adapters/big2 $PY -u core/zprobe/cot_zeroshot.py --pair $p --ids $D/ev_big2_$p.jsonl --variant $v \
    --out $H/cot${v}_$p.jsonl 2>&1 | grep -v -i -E "warn|pad_token"
done; done
mark "2 COMET"
items=""; for v in inline turn; do for p in $P; do items="$items cot${v}_$p=$H/cot${v}_$p.jsonl"; done; done
$CPY -u core/zprobe/comet_eval.py --n 0 --per_sent $H/scores $items 2>&1 | grep -v -i warn | tail -9
mark "3 report"
{
for v in inline turn; do
  echo "## cot $v vs big2 direct"; $CPY core/zprobe/paired_delta.py --scores $H/scores --a big2 --b cot$v --pairs $P
  echo "## cot $v vs ASR->Hy"; $CPY core/zprobe/paired_delta.py --scores $H/scores --a asr --b cot$v --pairs $P
done
$CPY - <<'EOF'
import json, numpy as np
H = "models/zprobe/data/hy"
print("\n| pair | big2 direct ms | cot ASR ms | cot translate ms | cot total ms | ASR + Hy ms |\n|---|---|---|---|---|---|")
for p in ["en-ko", "ko-en", "zh-ko"]:
    b = [json.loads(l)["ms"] for l in open(f"{H}/big2_t_{p}.jsonl")]
    c = [json.loads(l) for l in open(f"{H}/cotinline_{p}.jsonl")][1:]
    hy = [json.loads(l) for l in open(f"{H}/hy_asr_{p}.jsonl")]
    a_ = np.array([r["asr_ms"] for r in c]); t_ = np.array([r["trans_ms"] for r in c])
    print(f"| {p} | {np.median(b):.0f} | {np.median(a_):.0f} | {np.median(t_):.0f} | {np.median(a_ + t_):.0f} "
          f"| {np.median([r['asr_ms'] + r['ms'] for r in hy]):.0f} |")
r = {x["id"]: x["hyp"] for x in map(json.loads, open(f"{H}/cotinline_en-ko.jsonl"))}
for i in ["1662", "1930", "2005", "1679", "1663"]:
    print(f"case {i}: {r[i]}")
EOF
} | tee $LOG/cot_zeroshot.md
mark CHAIN_DONE
