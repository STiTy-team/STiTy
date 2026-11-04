#!/bin/bash
# 1.1 번역이 끝나면 1.2 지연 스윕 → 채점 → 집계. 순서는 완료 표시 파일로 잡는다.
#   tmux new-session -d -s lcmt-post -c "$PWD" "bash evaluation/LongContextMT/scripts/run_post.sh"
# GPU 를 쓰는 단계(로컬 스윕, 채점)는 로컬 번역이 끝난 뒤에만 돈다 — 로컬 지연 측정에 섞이지 않게.
set -uo pipefail
root=evaluation/LongContextMT
m=$root/logs/markers
export PYTHONPATH=$PWD
# 채점·집계용 파이썬 — unbabel-comet 과 metricx24 가 있는 환경. 번역은 PATH 의 python 으로 돈다.
METRICS_PY=${METRICS_PY:-python}
export PYTORCH_ALLOC_CONF=expandable_segments:True
log() { echo "[$(date +%FT%T)] $*"; }

wait_for() { until [ -f "$1" ]; do sleep 30; done; }

sweep() {
  local model=$1
  [ -f $m/sweep_$model.done ] && return
  log "sweep $model"
  python -u $root/scripts/latency_sweep.py --model "$model" >> $root/logs/sweep_$model.log 2>&1 \
    && touch $m/sweep_$model.done || log "sweep $model FAILED"
}

wait_for $m/translate_gpt-6-luna.done
sweep gpt-6-luna &
api_pid=$!
wait_for $m/translate_qwen3.5-4b-bf16.done
(
  SESSION=lcmt-post bash evaluation/DialogueContext/scripts/gpu_sampler.sh $root/logs/gpu_samples.csv &
  sweep qwen3.5-4b-bf16
)
wait $api_pid

if [ ! -f $m/score.done ]; then
  log "score"
  "$METRICS_PY" -u $root/scripts/score.py --run-dir $root/results/lcmt-20260930 \
    >> $root/logs/score.log 2>&1 && touch $m/score.done || log "score FAILED"
fi
log "aggregate"
"$METRICS_PY" -u $root/scripts/aggregate.py >> $root/logs/aggregate.log 2>&1 \
  && touch $m/aggregate.done || log "aggregate FAILED"
log "all done"
