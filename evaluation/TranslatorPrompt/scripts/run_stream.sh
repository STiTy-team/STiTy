#!/bin/bash
# 2.1: 모델별 번역 → 채점 → 집계. 모델 번역은 각자 세션, post 가 둘의 끝을 기다린다.
#   tmux new-session -d -s stream-api   -c "$PWD" "bash evaluation/TranslatorPrompt/scripts/run_stream.sh translate gpt-6-luna"
#   tmux new-session -d -s stream-local -c "$PWD" "bash evaluation/TranslatorPrompt/scripts/run_stream.sh translate qwen3.5-4b-bf16"
#   tmux new-session -d -s stream-post  -c "$PWD" "METRICS_PY=<채점 환경>/bin/python bash evaluation/TranslatorPrompt/scripts/run_stream.sh post"
# 로컬 번역을 다른 GPU 작업 뒤에 돌리려면 WAIT_FOR=<마커 파일> 을 명령 문자열 안에 준다.
set -uo pipefail
root=evaluation/TranslatorPrompt
m=$root/logs/markers
mkdir -p $m
export PYTHONPATH=$PWD
METRICS_PY=${METRICS_PY:-python}
TRANS_PY=${TRANS_PY:-python}
export PYTORCH_ALLOC_CONF=expandable_segments:True
run_id=${RUN_ID:-stream-20260930}
if [ "$1" = translate ]; then
  model=$2; shift 2
  [ -n "${WAIT_FOR:-}" ] && until [ -f "$WAIT_FOR" ]; do sleep 30; done
  "$TRANS_PY" -u $root/scripts/stream_run.py --model "$model" --run-id "$run_id" "$@" >> "$root/logs/stream_$model.log" 2>&1
  grep -q "\[$model\] finished" "$root/logs/stream_$model.log" && touch "$m/stream_${run_id}_$model.done"
  exit
fi
until [ -f $m/stream_${run_id}_gpt-6-luna.done ] && [ -f $m/stream_${run_id}_qwen3.5-4b-bf16.done ]; do sleep 30; done
echo "[$(date +%T)] score"
"$METRICS_PY" -u $root/scripts/stream_score.py $run_id >> $root/logs/stream_score.log 2>&1 || { echo "score FAILED"; exit 1; }
echo "[$(date +%T)] aggregate"
"$METRICS_PY" -u $root/scripts/stream_aggregate.py $run_id >> $root/logs/stream_aggregate.log 2>&1 \
  && touch $m/stream_${run_id}_aggregate.done || echo "aggregate FAILED"
echo "[$(date +%T)] all done"
