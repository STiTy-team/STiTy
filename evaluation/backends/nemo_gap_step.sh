#!/usr/bin/env bash
# Nemotron 스윕 공백 메우기: rc0 ko + rc6 en. 인자: RUN N
set -uo pipefail
RUN="$1"; N="$2"
PORT=8795
WT=/home/skkai/bench-wt/multi-asr-backends
BE="$WT/evaluation/backends"
CONDA=/home/skkai/miniforge3/bin/conda
mkdir -p "$RUN"
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$RUN/step.log"; }
rec(){ echo "  >> [$1] $2" | tee -a "$RUN/step.log"; }

CUR=""
cleanup(){
  log "teardown 시작"
  if [[ -n "$CUR" ]]; then
    kill -TERM -"$CUR" 2>/dev/null || kill -TERM "$CUR" 2>/dev/null
    for i in $(seq 1 20); do kill -0 "$CUR" 2>/dev/null || break; sleep 1; done
    kill -KILL -"$CUR" 2>/dev/null || kill -KILL "$CUR" 2>/dev/null
  fi
  sleep 3
  log "teardown 후 GPU: $(nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader)"
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader >> "$RUN/step.log" 2>&1
  touch "$RUN/DONE"; log "DONE"
}
trap cleanup EXIT INT TERM

( while true; do
    printf "%s  %s  free=%s\n" "$(date +%H:%M:%S)" \
      "$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | tr "\n" ";")" \
      "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader)"
    sleep 15
  done >> "$RUN/gpu_watch.log" ) &
WPID=$!

run_case(){
  local rc="$1" lang="$2" n="$3" tag="$4"
  local locale="ko-KR"; [[ "$lang" == "en" ]] && locale="en-US"
  local slog="$RUN/nemotron_${tag}_server.log"
  log "케이스 rc=$rc lang=$lang locale=$locale n=$n"
  if ss -ltn 2>/dev/null | grep -q ":$PORT "; then rec "$tag" "SKIP - 포트 $PORT 사용중"; return; fi
  local free; free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
  if [[ "$free" -lt 4000 ]]; then rec "$tag" "SKIP - VRAM 부족 (${free}MiB)"; return; fi

  setsid $CONDA run --no-capture-output -n asr-nemotron python "$BE/server.py" \
      --backend nemotron --lang "$locale" --right-context "$rc" \
      --port "$PORT" --log-file "$slog" > "$slog.stdout" 2>&1 &
  CUR=$!
  local ready=0
  for i in $(seq 1 120); do
    kill -0 "$CUR" 2>/dev/null || { rec "$tag" "서버 즉사 (${tag}_server.log.stdout)"; CUR=""; return; }
    if ss -ltn 2>/dev/null | grep -q ":$PORT "; then ready=1; break; fi
    sleep 5
  done
  if [[ $ready -ne 1 ]]; then rec "$tag" "600s 안에 안 뜸"; else
    log "  서버 ok, 클라이언트 시작"
    if timeout 5400 $CONDA run -n asr-nemotron python "$BE/smoke_client.py" \
          --ws "ws://127.0.0.1:$PORT" --lang "$lang" --limit "$n" \
          --tag "nemotron_$tag" --out "$RUN/nemotron_${tag}.json" \
          > "$RUN/nemotron_${tag}_client.log" 2>&1; then
      rec "$tag" "ok"
    else
      rec "$tag" "client FAILED"
    fi
    tail -2 "$RUN/nemotron_${tag}_client.log" | tee -a "$RUN/step.log"
  fi
  kill -TERM -"$CUR" 2>/dev/null || kill -TERM "$CUR" 2>/dev/null
  for i in $(seq 1 20); do kill -0 "$CUR" 2>/dev/null || break; sleep 1; done
  kill -KILL -"$CUR" 2>/dev/null || true
  CUR=""; sleep 5
  log "  케이스 종료, GPU: $(nvidia-smi --query-gpu=memory.free --format=csv,noheader)"
}

log "시작 n=$N"
log "시작 GPU: $(nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader)"
run_case 0 ko "$N" "sweep_rc0_80ms"
run_case 6 en "$N" "sweep_rc6_560ms_en"
kill $WPID 2>/dev/null
log "본작업 종료"
