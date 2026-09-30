#!/bin/bash
# 2.3 에 조건을 더 돌린다 (같은 run). 두 모델 번역을 동시에 하고 끝나면 채점(캐시에 없는 것만)과 집계.
#   tmux new-session -d -s qual-s5 -c "$PWD" "bash evaluation/TranslatorPrompt/scripts/run_quality_extra.sh S5"
set -uo pipefail
root=evaluation/TranslatorPrompt
export PYTHONPATH=$PWD
# 채점·집계용 파이썬 — unbabel-comet 과 metricx24 가 있는 환경. 번역은 PATH 의 python 으로 돈다.
METRICS_PY=${METRICS_PY:-python}
export PYTORCH_ALLOC_CONF=expandable_segments:True
run_id=${RUN_ID:-qual-20260930}
python -u $root/scripts/quality_run.py --model gpt-6-luna --run-id $run_id --conditions "$@" \
  >> $root/logs/quality_gpt-6-luna.log 2>&1 &
python -u $root/scripts/quality_run.py --model qwen3.5-4b-bf16 --run-id $run_id --batch 16 --conditions "$@" \
  >> $root/logs/quality_qwen3.5-4b-bf16.log 2>&1
wait
echo "[$(date +%T)] score"
"$METRICS_PY" -u evaluation/LongContextMT/scripts/score.py --run-dir $root/results/$run_id \
  --glob "*/quality.jsonl" --only comet kiwi metricx >> $root/logs/quality_score.log 2>&1 || echo "score FAILED"
echo "[$(date +%T)] aggregate"
"$METRICS_PY" -u $root/scripts/quality_aggregate.py $run_id >> $root/logs/quality_aggregate.log 2>&1 \
  || echo "aggregate FAILED"
echo "[$(date +%T)] all done"
