#!/bin/bash
# Hy-MT2-1.8B (bf16) cascade vs big2 translation adapter, FLEURS test, 12 directions of en/ko/ja/zh.
# Ids per pair = ev_big2_<pair>.jsonl (already produced by the big2 eval), so every arm scores the same sentences.
#   tmux new-session -d -s hy-vs-adapter -c /home/mobility/STiTy "bash core/zprobe/chains/hy_vs_adapter.sh > core/zprobe/chains/hy_vs_adapter.log 2>&1"
# Each step is skipped when its output already has the expected number of lines, so rerunning resumes.
cd /home/mobility/STiTy
set -a; . ./.env; set +a
export PYTHONPATH=Qwen3-ASR
PY=.venv/bin/python; CPY=.venv-tpm/bin/python
D=models/zprobe/data; H=$D/hy; LOG=models/zprobe/logs
mkdir -p $H/scores
PAIRS="en-ko en-ja en-zh ko-en ko-ja ko-zh ja-en ja-ko ja-zh zh-en zh-ko zh-ja"
mark() { echo "$(date +%T) MARK $1"; }
lines() { [ -f "$1" ] && wc -l < "$1" || echo 0; }
quiet() { grep -v -E "pad_token_id|Warning|warn" ; }
until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -ge 10000 ]; do
  echo "$(date +%T) waiting for 10 GB free GPU"; sleep 60; done

mark "1 base ASR"
for l in en ko ja zh; do
  n=$($PY -c "import sys; sys.path.insert(0,'core/zprobe'); import probe_z as p; print(len(p.common_ids(['$l'])))")
  [ "$(lines $H/asr_base_$l.jsonl)" -eq "$n" ] && { echo "skip asr $l"; continue; }
  $PY -u core/zprobe/probe_z.py tagswap --pair $l-$l --out $H/asr_base_$l.jsonl 2>&1 | quiet | tail -4
done

mark "2 big2 timing rerun"
for p in en-ko ko-en zh-ko; do
  [ "$(lines $H/big2_t_$p.jsonl)" -eq "$(lines $D/ev_big2_$p.jsonl)" ] && { echo "skip big2_t $p"; continue; }
  PROBE_ADAPTER=models/zprobe/adapters/big2 $PY -u core/zprobe/probe_z.py tagswap --pair $p --out $H/big2_t_$p.jsonl 2>&1 | quiet | tail -4
done

mark "3 Hy on ASR transcripts"
todo=""; for p in $PAIRS; do [ "$(lines $H/hy_asr_$p.jsonl)" -eq "$(lines $D/ev_big2_$p.jsonl)" ] || todo="$todo $p"; done
[ -n "$todo" ] && $PY -u core/zprobe/cascade_hy.py --pair $todo --ids "$D/ev_big2_{pair}.jsonl" --asr "$H/asr_base_{src}.jsonl" --out "$H/hy_asr_{pair}.jsonl" 2>&1 | quiet

mark "4 Hy on gold transcripts"
todo=""; for p in $PAIRS; do [ "$(lines $H/hy_gold_$p.jsonl)" -eq "$(lines $D/ev_big2_$p.jsonl)" ] || todo="$todo $p"; done
[ -n "$todo" ] && $PY -u core/zprobe/cascade_hy.py --pair $todo --ids "$D/ev_big2_{pair}.jsonl" --gold --out "$H/hy_gold_{pair}.jsonl" 2>&1 | quiet

mark "5 COMET"
items=""; for p in $PAIRS; do items="$items gold_$p=$H/hy_gold_$p.jsonl asr_$p=$H/hy_asr_$p.jsonl big2_$p=$D/ev_big2_$p.jsonl"; done
$CPY -u core/zprobe/comet_eval.py --n 0 --per_sent $H/scores $items 2>&1 | quiet | tail -40

mark "6 report"
$CPY core/zprobe/hy_vs_adapter_report.py --hy $H --scores $H/scores --data $D --out $LOG/hy_vs_adapter.md
for p in en-ko ko-en zh-ko; do
  $PY -c "import json; a=[json.loads(l)['hyp'] for l in open('$D/ev_big2_$p.jsonl')]; b=[json.loads(l)['hyp'] for l in open('$H/big2_t_$p.jsonl')]; print('$p big2 rerun identical:', sum(x==y for x,y in zip(a,b)), '/', len(a))"
done
mark CHAIN_DONE
