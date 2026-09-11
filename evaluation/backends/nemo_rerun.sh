#!/usr/bin/env bash
# 단어 경계 델타 수정본으로 Nemotron ko/en 만 재측정한다(다른 백엔드는 영향 없음).
set -uo pipefail
N=${N:-50}
PEAK=${PEAK:-0.5}
RUN=${RUN:-$HOME/bench-results/nemofix_$(date +%Y%m%d_%H%M%S)}
BE=$HOME/bench-wt/multi-asr-backends/evaluation/backends
CONDA=$HOME/miniforge3/bin/conda
PORT=8795
mkdir -p "$RUN"
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$RUN/step.log"; }
CUR=""
cleanup(){ [[ -n "$CUR" ]] && { kill -TERM -"$CUR" 2>/dev/null; sleep 5; kill -KILL -"$CUR" 2>/dev/null; }
           sleep 3; log "teardown GPU: $(nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader)"
           nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader >> "$RUN/step.log"
           touch "$RUN/DONE"; log DONE; }
trap cleanup EXIT INT TERM
wait_port(){ for _ in $(seq 1 "$2"); do ss -ltn 2>/dev/null | grep -q ":$1 " && return 0; sleep 5; done; return 1; }
log "시작 N=$N PEAK=$PEAK RUN=$RUN"
for L in ko en; do
  LOCALE="ko-KR"; [[ "$L" == "en" ]] && LOCALE="en-US"
  SLOG="$RUN/nemotron_${L}_server.log"
  setsid "$CONDA" run --no-capture-output -n asr-nemotron python "$BE/server.py" \
      --backend nemotron --lang "$LOCALE" --right-context 13 \
      --port "$PORT" --log-file "$SLOG" > "$SLOG.stdout" 2>&1 &
  CUR=$!
  if wait_port "$PORT" 120; then
    timeout 7200 "$CONDA" run --no-capture-output -n asr-nemotron python "$BE/smoke_client.py" \
        --ws "ws://127.0.0.1:$PORT" --lang "$L" --limit "$N" \
        --peak-normalize "$PEAK" --tag "nemotron_rc13" --out "$RUN/nemotron_rc13_${L}.json" \
        > "$RUN/nemotron_rc13_${L}_client.log" 2>&1 \
      && log "  nemotron.$L ok" || log "  nemotron.$L FAILED"
    tail -1 "$RUN/nemotron_rc13_${L}_client.log" | tee -a "$RUN/step.log"
  else
    log "  nemotron.$L 서버 안 뜸"
  fi
  kill -TERM -"$CUR" 2>/dev/null; sleep 8; kill -KILL -"$CUR" 2>/dev/null; CUR=""; sleep 5
  log "  nemotron $L 종료, GPU: $(nvidia-smi --query-gpu=memory.free --format=csv,noheader)"
done
log "본작업 종료"
