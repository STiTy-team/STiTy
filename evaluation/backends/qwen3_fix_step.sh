#!/usr/bin/env bash
# Qwen3 기준선 재실행. 인자: RUN UTIL N "LANGS" [EXTRA_SERVER_ARGS]
set -uo pipefail
RUN="$1"; UTIL="$2"; N="$3"; LANGS="$4"; EXTRA="${5:-}"
PORT=8796
S=/home/skkai/STiTy-team/STiTy
BE=/home/skkai/bench-wt/multi-asr-backends/evaluation/backends
CONDA=/home/skkai/miniforge3/bin/conda
mkdir -p "$RUN"
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$RUN/step.log"; }
rec(){ echo "  >> [$1] $2" | tee -a "$RUN/step.log"; }

SPID=""
cleanup(){
  log "teardown 시작"
  if [[ -n "$SPID" ]]; then
    kill -TERM -"$SPID" 2>/dev/null || kill -TERM "$SPID" 2>/dev/null
    for i in $(seq 1 20); do kill -0 "$SPID" 2>/dev/null || break; sleep 1; done
    kill -KILL -"$SPID" 2>/dev/null || kill -KILL "$SPID" 2>/dev/null
  fi
  sleep 3
  log "teardown 후 GPU: $(nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader)"
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader >> "$RUN/step.log" 2>&1
  touch "$RUN/DONE"
  log "DONE"
}
trap cleanup EXIT INT TERM

log "시작 util=$UTIL n=$N langs=$LANGS extra=[$EXTRA]"
log "시작 GPU: $(nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader)"
if ss -ltn 2>/dev/null | grep -q ":$PORT "; then rec serve "ABORT - 포트 $PORT 사용중"; exit 1; fi

( while true; do
    printf "%s  %s  free=%s\n" "$(date +%H:%M:%S)" \
      "$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | tr "\n" ";")" \
      "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader)"
    sleep 15
  done >> "$RUN/gpu_watch.log" ) &
WPID=$!

cd "$S" || exit 1
setsid $CONDA run --no-capture-output -n stity python \
  "$S/evaluation/streaming_websocket_server_ast.py" \
  --no-idle-shutdown --port "$PORT" --gpu-memory-utilization "$UTIL" $EXTRA \
  > "$RUN/server.log" 2>&1 &
SPID=$!
log "server pgid=$SPID"

READY=0
for i in $(seq 1 240); do
  kill -0 "$SPID" 2>/dev/null || { rec serve "FAILED - 프로세스 즉사 (server.log)"; kill $WPID 2>/dev/null; exit 1; }
  if ss -ltn 2>/dev/null | grep -q ":$PORT "; then READY=1; break; fi
  sleep 5
done
[[ $READY -eq 1 ]] || { rec serve "FAILED - 1200s 안에 안 뜸"; kill $WPID 2>/dev/null; exit 1; }
rec serve "ok"
log "기동 후 GPU: $(nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader)"

for L in $LANGS; do
  if ! kill -0 "$SPID" 2>/dev/null; then rec serve "중간에 죽음 (server.log)"; break; fi
  log "smoke_$L 시작 ($N 클립)"
  if timeout 5400 $CONDA run -n asr-nemotron python "$BE/smoke_client.py" \
        --ws "ws://127.0.0.1:$PORT" --lang "$L" --limit "$N" \
        --tag "qwen3_$L" --out "$RUN/qwen3_smoke_${L}.json" \
        > "$RUN/qwen3_smoke_${L}_client.log" 2>&1; then
    rec "smoke_$L" "ok"
  else
    rec "smoke_$L" "FAILED"
  fi
  tail -3 "$RUN/qwen3_smoke_${L}_client.log" | tee -a "$RUN/step.log"
done

kill $WPID 2>/dev/null
tail -120 "$RUN/server.log" > "$RUN/server_tail.log" 2>/dev/null
log "본작업 종료"
