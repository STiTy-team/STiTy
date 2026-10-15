#!/bin/bash
# B-seg arm of the Hy comparison: segft9_mix (koen-seg-mix SEG base + translation LoRA with clause <SEG> taught on
# en-en and en-ko) on the same 12 directions and ids as hy_vs_adapter.sh. Needs that chain's Hy outputs.
# COMET with <SEG> stripped (segmix_*) and raw (segmixraw_*).
#   tmux new-session -d -s hy-vs-seg -c /home/mobility/STiTy "bash core/zprobe/chains/hy_vs_segadapter.sh > core/zprobe/chains/hy_vs_segadapter.log 2>&1"
cd /home/mobility/STiTy
set -a; . ./.env; set +a
export PYTHONPATH=Qwen3-ASR
PY=.venv/bin/python; CPY=.venv-tpm/bin/python
D=models/zprobe/data; H=$D/hy; LOG=models/zprobe/logs
BASE=models/Qwen3-ASR-1.7B-koen-seg-mix-merged; AD=models/zprobe/adapters/segft9_mix
PAIRS="en-ko en-ja en-zh ko-en ko-ja ko-zh ja-en ja-ko ja-zh zh-en zh-ko zh-ja"
mark() { echo "$(date +%T) MARK $1"; }
lines() { [ -f "$1" ] && wc -l < "$1" || echo 0; }
quiet() { grep -v -E "pad_token_id|Warning|warn" ; }
until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -ge 10000 ]; do
  echo "$(date +%T) waiting for 10 GB free GPU"; sleep 60; done

mark "1 segft9_mix outputs"
for p in $PAIRS; do
  [ "$(lines $H/segmix_$p.jsonl)" -eq "$(lines $D/ev_big2_$p.jsonl)" ] && { echo "skip $p"; continue; }
  PROBE_MODEL=$BASE PROBE_ADAPTER=$AD $PY -u core/zprobe/probe_z.py tagswap --pair $p --out $H/segmix_$p.jsonl 2>&1 | quiet | tail -3
done

mark "2 COMET"
strip=""; raw=""
for p in $PAIRS; do strip="$strip segmix_$p=$H/segmix_$p.jsonl"; raw="$raw segmixraw_$p=$H/segmix_$p.jsonl"; done
$CPY -u core/zprobe/comet_eval.py --n 0 --strip_seg --per_sent $H/scores $strip 2>&1 | quiet | tail -14
$CPY -u core/zprobe/comet_eval.py --n 0 --per_sent $H/scores $raw 2>&1 | quiet | tail -14

mark "3 report"
$CPY core/zprobe/hy_vs_adapter_report.py --hy $H --scores $H/scores --data $D --b segmix \
  --b_files "$H/segmix_{p}.jsonl" --strip_seg --out $LOG/hy_vs_segadapter.md
mark CHAIN_DONE
