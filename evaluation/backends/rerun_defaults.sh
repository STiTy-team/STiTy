#!/usr/bin/env bash
# 세 백엔드 기본값 재실행 (2026-09-10, 09-11 갱신).
#
# 이전 표의 문제 셋을 고치고 다시 잰다:
#  1) Nemotron 만 스윕으로 최적점(rc6)을 쓰고 나머지는 기본값이었다 -> 전부 기본값.
#     Nemotron 기본값은 rc13(num_lookahead_tokens=13, 1120ms).
#  2) 지연 지표가 백엔드마다 다른 정의였다 -> smoke_client/smoke_voxtral 을 같은 정의로
#     맞췄다(ttfo_sec / completion_lag_sec / xrt, 전부 '무음 제외 실제 오디오 끝' 기준).
#  3) Voxtral 이 발화 중 출력을 안 한 건 우리가 commit 을 끝에 한 번만 보냈기 때문이다.
#     주기적 commit(1초) 구성을 정식으로 추가하고, 종전 방식도 참고로 같이 잰다.
#
# 2026-09-11 추가:
#  4) Nemotron 어댑터가 finish 에서 한 번에 뱉던 것을 streamer 로 증분화했다.
#     이전 표의 "Nemotron 지연 꼴찌"는 모델이 아니라 이 어댑터 탓이었다.
#  5) 입력 레벨을 세 백엔드 동일하게 피크 정규화한다(PEAK). FLEURS 는 클립별
#     녹음 레벨이 50dB 넘게 벌어져 있고, Voxtral 은 피크 0.005 아래에서 전사가
#     통째로 빈다(en 빈 전사 4건의 원인). Qwen3/Nemotron 은 내부 정규화가 있어
#     영향이 적지만, 공정성을 위해 같은 오디오를 먹인다.
#
# 클립 수는 세 백엔드 동일하게 ko/en 각 50 (load_fleurs 가 file_id 정렬 후 앞 N개라 같은 클립).
set -uo pipefail

N=${N:-50}
PEAK=${PEAK:-0.5}    # 0 이면 원본 레벨. 0.1/0.5/0.95 에서 전사 동일함을 확인했다.
RUN=${RUN:-$HOME/bench-results/defaults_$(date +%Y%m%d_%H%M%S)}
WT=$HOME/bench-wt/multi-asr-backends
BE="$WT/evaluation/backends"
S=$HOME/STiTy-team/STiTy
CONDA=$HOME/miniforge3/bin/conda
VOX_PORT=8010
NEMO_PORT=8795
QWEN_PORT=8796

mkdir -p "$RUN"
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$RUN/step.log"; }
rec(){ echo "  >> [$1] $2" | tee -a "$RUN/step.log"; }
gpu(){ nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader; }

CUR=""          # 현재 우리가 띄운 서버의 pgid
VOX_OURS=0      # Voxtral 을 우리가 띄웠으면 1 (남이 띄운 건 안 내린다)

cleanup(){
  log "teardown 시작"
  [[ -n "$CUR" ]] && { kill -TERM -"$CUR" 2>/dev/null || kill -TERM "$CUR" 2>/dev/null; sleep 5;
                       kill -KILL -"$CUR" 2>/dev/null || kill -KILL "$CUR" 2>/dev/null; }
  if [[ "$VOX_OURS" == "1" ]]; then
    pkill -f "Voxtral-Mini-4B-Realtime" 2>/dev/null || true
  fi
  sleep 3
  log "teardown 후 GPU: $(gpu)"
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader >> "$RUN/step.log" 2>&1
  touch "$RUN/DONE"; log "DONE"
}
trap cleanup EXIT INT TERM

( while true; do
    printf "%s  %s  free=%s\n" "$(date +%H:%M:%S)" \
      "$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | tr '\n' ';')" \
      "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader)"
    sleep 20
  done >> "$RUN/gpu_watch.log" ) &
WPID=$!

wait_port(){  # port timeout_sec
  for _ in $(seq 1 "$2"); do
    ss -ltn 2>/dev/null | grep -q ":$1 " && return 0
    [[ -n "$CUR" ]] && { kill -0 "$CUR" 2>/dev/null || return 2; }
    sleep 5
  done
  return 1
}

log "시작 N=$N PEAK=$PEAK RUN=$RUN"
log "시작 GPU: $(gpu)"

# ── PHASE 1: Voxtral (기본 지연 노브, commit 구성 2종) ─────────────────────
log "PHASE 1: Voxtral"
if ss -ltn 2>/dev/null | grep -q ":$VOX_PORT "; then
  rec voxtral.serve "이미 떠 있는 서버 사용"
else
  setsid bash "$BE/run_voxtral.sh" > "$RUN/voxtral_server.log" 2>&1 &
  CUR=$!; VOX_OURS=1
  if wait_port "$VOX_PORT" 120; then rec voxtral.serve "ok"; else rec voxtral.serve "FAILED"; CUR=""; fi
fi

if ss -ltn 2>/dev/null | grep -q ":$VOX_PORT "; then
  for L in ko en; do
    for CI in 1.0 0; do
      TAG="voxtral_ci${CI/./_}"
      log "  voxtral $L commit_interval=$CI ($N 클립)"
      timeout 7200 env LD_LIBRARY_PATH="$HOME/miniforge3/envs/asr-voxtral/lib" \
        "$CONDA" run -n asr-voxtral python "$BE/smoke_voxtral.py" \
          --base-url "http://127.0.0.1:$VOX_PORT" --lang "$L" --limit "$N" \
          --commit-interval-sec "$CI" --peak-normalize "$PEAK" --tag "$TAG" \
          --out "$RUN/${TAG}_${L}.json" > "$RUN/${TAG}_${L}_client.log" 2>&1 \
        && rec "$TAG.$L" ok || rec "$TAG.$L" FAILED
      tail -2 "$RUN/${TAG}_${L}_client.log" | tee -a "$RUN/step.log"
    done
  done
fi
[[ "$VOX_OURS" == "1" ]] && { pkill -f "Voxtral-Mini-4B-Realtime" 2>/dev/null || true; CUR=""; sleep 10; }
log "PHASE 1 종료, GPU: $(gpu)"

# ── PHASE 2: Nemotron 기본값 (rc13 = 1120ms) ──────────────────────────────
log "PHASE 2: Nemotron rc13(기본값)"
for L in ko en; do
  LOCALE="ko-KR"; [[ "$L" == "en" ]] && LOCALE="en-US"
  SLOG="$RUN/nemotron_${L}_server.log"
  setsid "$CONDA" run --no-capture-output -n asr-nemotron python "$BE/server.py" \
      --backend nemotron --lang "$LOCALE" --right-context 13 \
      --port "$NEMO_PORT" --log-file "$SLOG" > "$SLOG.stdout" 2>&1 &
  CUR=$!
  if wait_port "$NEMO_PORT" 120; then
    timeout 7200 "$CONDA" run -n asr-nemotron python "$BE/smoke_client.py" \
        --ws "ws://127.0.0.1:$NEMO_PORT" --lang "$L" --limit "$N" \
        --peak-normalize "$PEAK" --tag "nemotron_rc13" --out "$RUN/nemotron_rc13_${L}.json" \
        > "$RUN/nemotron_rc13_${L}_client.log" 2>&1 \
      && rec "nemotron.$L" ok || rec "nemotron.$L" FAILED
    tail -2 "$RUN/nemotron_rc13_${L}_client.log" | tee -a "$RUN/step.log"
  else
    rec "nemotron.$L" "서버 안 뜸"
  fi
  kill -TERM -"$CUR" 2>/dev/null || kill -TERM "$CUR" 2>/dev/null; sleep 8
  kill -KILL -"$CUR" 2>/dev/null || true; CUR=""; sleep 5
  log "  nemotron $L 종료, GPU: $(gpu)"
done

# ── PHASE 3: Qwen3 기준선 (기본 청크/커밋 정책) ───────────────────────────
log "PHASE 3: Qwen3 기준선"
cd "$S" || exit 1
setsid "$CONDA" run --no-capture-output -n stity python \
  "$S/evaluation/streaming_websocket_server_ast.py" \
  --no-idle-shutdown --port "$QWEN_PORT" --gpu-memory-utilization 0.60 \
  --trans-backend gtx --trans-retries 1 > "$RUN/qwen3_server.log" 2>&1 &
CUR=$!
if wait_port "$QWEN_PORT" 240; then
  rec qwen3.serve ok
  for L in ko en; do
    kill -0 "$CUR" 2>/dev/null || { rec qwen3.serve "중간에 죽음"; break; }
    timeout 7200 "$CONDA" run -n asr-nemotron python "$BE/smoke_client.py" \
        --ws "ws://127.0.0.1:$QWEN_PORT" --lang "$L" --limit "$N" \
        --peak-normalize "$PEAK" --tag "qwen3" --out "$RUN/qwen3_${L}.json" \
        > "$RUN/qwen3_${L}_client.log" 2>&1 \
      && rec "qwen3.$L" ok || rec "qwen3.$L" FAILED
    tail -2 "$RUN/qwen3_${L}_client.log" | tee -a "$RUN/step.log"
  done
else
  rec qwen3.serve "FAILED - 서버 안 뜸"
fi
kill -TERM -"$CUR" 2>/dev/null || kill -TERM "$CUR" 2>/dev/null; sleep 8
kill -KILL -"$CUR" 2>/dev/null || true; CUR=""

kill $WPID 2>/dev/null

# ── 요약 ──────────────────────────────────────────────────────────────────
python3 - "$RUN" > "$RUN/summary.txt" <<'PY'
import json, sys, glob, os
run = sys.argv[1]
print("%-22s %-4s %8s %8s %9s %9s %7s %6s" %
      ("backend", "lang", "err", "ttfo", "compLag", "xrt", "pieces", "empty"))
for p in sorted(glob.glob(os.path.join(run, "*.json"))):
    try:
        s = json.load(open(p))["summary"]
    except Exception:
        continue
    unit = s.get("unit", "")
    err = s.get("avg_" + unit)
    rows = json.load(open(p)).get("rows", [])
    pieces = round(sum(r.get("num_finals", 0) for r in rows) / max(len(rows), 1), 1)
    print("%-22s %-4s %8s %8s %9s %9s %7s %6s" % (
        s.get("backend"), s.get("lang"),
        ("%.4f" % err) if err is not None else "NA",
        s.get("avg_ttfo_sec"), s.get("avg_completion_lag_sec"),
        s.get("avg_xrt"), pieces, s.get("empty_transcripts")))
PY
cat "$RUN/summary.txt" | tee -a "$RUN/step.log"
log "본작업 종료"
