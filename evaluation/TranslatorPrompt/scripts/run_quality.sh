#!/bin/bash
# 2.3: 두 모델 번역 → 채점 → 집계. 모델 번역은 각자 세션, 이 스크립트가 끝을 기다린다.
#   tmux new-session -d -s qual-api   -c "$PWD" "bash evaluation/TranslatorPrompt/scripts/run_quality.sh translate gpt-6-luna"
#   tmux new-session -d -s qual-local -c "$PWD" "bash evaluation/TranslatorPrompt/scripts/run_quality.sh translate qwen3.5-4b-bf16"
#   tmux new-session -d -s qual-post  -c "$PWD" "bash evaluation/TranslatorPrompt/scripts/run_quality.sh post"
set -uo pipefail
root=evaluation/TranslatorPrompt
m=$root/logs/markers
export PYTHONPATH=$PWD
# 채점·집계용 파이썬 — unbabel-comet 과 metricx24 가 있는 환경. 번역은 PATH 의 python 으로 돈다.
METRICS_PY=${METRICS_PY:-python}
export PYTORCH_ALLOC_CONF=expandable_segments:True
run_id=${RUN_ID:-qual-20260930}
if [ "$1" = translate ]; then
  model=$2; shift 2
  python -u $root/scripts/quality_run.py --model "$model" --run-id "$run_id" "$@" >> "$root/logs/quality_$model.log" 2>&1
  grep -q "\[$model\] finished" "$root/logs/quality_$model.log" && touch "$m/quality_${run_id}_$model.done"
  exit
fi
until [ -f $m/quality_${run_id}_gpt-6-luna.done ] && [ -f $m/quality_${run_id}_qwen3.5-4b-bf16.done ]; do sleep 30; done
echo "[$(date +%T)] score"
"$METRICS_PY" -u evaluation/LongContextMT/scripts/score.py --run-dir $root/results/$run_id \
  --glob "*/quality.jsonl" --only comet kiwi metricx >> $root/logs/quality_score.log 2>&1 || echo "score FAILED"
echo "[$(date +%T)] aggregate"
"$METRICS_PY" -u $root/scripts/quality_aggregate.py $run_id >> $root/logs/quality_aggregate.log 2>&1 \
  && touch $m/quality_${run_id}_aggregate.done || echo "aggregate FAILED"
echo "[$(date +%T)] all done"
