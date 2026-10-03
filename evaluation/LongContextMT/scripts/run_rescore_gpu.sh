#!/bin/bash
# rescore.py GPU 단계 → 집계. GPU 가 비고 3분 뒤에도 비어 있을 때 시작한다.
#   tmux new-session -d -s lcmt-rescore -c "$PWD" "CORE_PATH=<core 사본> METRICS_PY=<채점 환경 python> bash evaluation/LongContextMT/scripts/run_rescore_gpu.sh"
# 빈 것의 기준: 여유 메모리 MIN_FREE_MIB 이상, 사용률 MAX_UTIL% 이하. 상주 프로세스(Xorg, rerun 0.4GB)는 기준 안에 든다.
set -uo pipefail
root=evaluation/LongContextMT
m=$root/logs/markers
# core 를 다른 사본에서 읽어야 할 때(작업 트리가 merge 도중 등) CORE_PATH 로 넘긴다.
export PYTHONPATH=${CORE_PATH:-$PWD}
METRICS_PY=${METRICS_PY:-python}
MIN_FREE_MIB=${MIN_FREE_MIB:-20000}
MAX_UTIL=${MAX_UTIL:-10}
CONFIRM_SEC=${CONFIRM_SEC:-180}
POLL_SEC=${POLL_SEC:-60}
export PYTORCH_ALLOC_CONF=expandable_segments:True
log() { echo "[$(date +%FT%T)] $*"; }

idle() {
  read -r free util < <(nvidia-smi --query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits | tr -d ',')
  log "gpu free ${free}MiB util ${util}%"
  [ "$free" -ge "$MIN_FREE_MIB" ] && [ "$util" -le "$MAX_UTIL" ]
}

[ -f $m/rescore_gpu.done ] && { log "already done"; exit 0; }
while true; do
  if idle; then
    log "idle — rechecking in ${CONFIRM_SEC}s"
    sleep "$CONFIRM_SEC"
    if idle; then
      log "still idle — starting"
      break
    fi
    log "busy again — keep waiting"
  fi
  sleep "$POLL_SEC"
done

log "rescore gpu"
"$METRICS_PY" -u $root/scripts/rescore.py --stage gpu || { log "rescore gpu FAILED"; exit 1; }
log "aggregate"
"$METRICS_PY" $root/scripts/aggregate_rescore.py > /dev/null && touch $m/rescore_gpu.done || log "aggregate FAILED"
log "all done"
