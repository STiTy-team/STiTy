#!/usr/bin/env bash
# Voxtral / Nemotron / Qwen3 백엔드 벤치마크 - 2026-09-08 재실행판 (v2).
#
# 20260907_233539 런의 실패를 고친다:
#   1) nemotron probe : librosa / accelerate 누락        -> 시작 시 명시 설치
#   2) voxtral serve  : 시스템 libstdc++ 가 먼저 로드     -> LD_LIBRARY_PATH 로 env 우선
#   3) 산출물 소실    : results/ 가 워크트리 안이라 clean 에 쓸림 -> $HOME/bench-results
#   4) 워크트리 점유  : 공용 체크아웃(main)을 안 건드리게 git worktree 사용
#
# v2 에서 추가로 잡은 것:
#   5) 영어 스모크가 ko-KR 로 돌 뻔했다. smoke_client 는 항상 lang="auto" 를 보내고
#      (smoke_client.py:88) 엔진은 auto 면 서버 기동 시 로케일로 떨어진다
#      (engine_nemotron.py:82-84). -> 서버에 --lang 을 언어별로 넘긴다.
#   6) 지난 런은 죽은 서버를 40분간 기다렸다. -> 서버 PID 가 죽으면 즉시 중단.
#   7) vLLM 의 EngineCore 는 cmdline 이 "VLLM::EngineCore" 로 바뀌어 이름 기반
#      pkill 로는 안 잡힌다. -> 프로세스 트리로 죽인다.
#   8) GPU 반납은 "우리가 시작한 뒤 새로 생긴 컴퓨트 프로세스"만 대상으로 한다.
#      남이 그 사이 띄운 작업은 건드리지 않는다.
#   9) qwen3 기준선의 빈 전사 원인을 알 수 있게 원본 메시지를 먼저 덤프한다.

set -uo pipefail

REPO="$HOME/STiTy-team/STiTy"
WT="$HOME/bench-wt/multi-asr-backends"
BRANCH="feat/multi-asr-backends"
BE="$WT/evaluation/backends"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN="$HOME/bench-results/$STAMP"
CONDA_SH="$HOME/miniforge3/etc/profile.d/conda.sh"
STATUS="$RUN/status.json"
VOX_LIB="$HOME/miniforge3/envs/asr-voxtral/lib"
VOX_MODEL="mistralai/Voxtral-Mini-4B-Realtime-2602"
RAW_PROBE="$HOME/bin/qwen3_raw_probe.py"

# 포트는 남과 겹치지 않게 잡는다. 2026-09-08 23:50 부터 다른 세션이 8765 에
# 자기 AST 서버를 띄웠다. 겹치면 wait_ws 가 "남의 서버"에 붙어 성공으로 오판하고
# 그 결과가 우리 백엔드 이름으로 기록된다 - 조용히 틀린 숫자가 나온다.
NEMO_PORT=8795
VOX_PORT=8010
QWEN_PORT=8796

# 단계별 최소 VRAM (MiB). 부족하면 기다렸다가, 그래도 없으면 건너뛴다.
NEED_NEMO=4000
NEED_VOX=15000
NEED_QWEN=8000
GATE_WAIT=7200          # 최대 2시간까지 기다린다

SMOKE_N=20
SWEEP_N=50
while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke-n) SMOKE_N="$2"; shift 2 ;;
    --sweep-n) SWEEP_N="$2"; shift 2 ;;
    *) echo "unknown arg $1"; exit 2 ;;
  esac
done

mkdir -p "$RUN"
exec > >(tee -a "$RUN/driver.log") 2>&1

echo "=============================================================="
echo " backend bench (재실행 v2)  $STAMP"
echo " run dir: $RUN"
echo "=============================================================="

# shellcheck disable=SC1090
source "$CONDA_SH"

# 우리가 시작하기 전부터 GPU 를 쓰고 있던 프로세스. 이것들은 끝까지 건드리지 않는다.
PRE_GPU_PIDS=" $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr '\n' ' ') "
echo " 시작 시점 GPU 점유 프로세스(보호 대상): ${PRE_GPU_PIDS:-없음}"

declare -A RESULT
record() { RESULT["$1"]="$2"; echo "  >> [$1] $2"; }

write_status() {
  { echo "{"
    echo "  \"run\": \"$STAMP\","
    echo "  \"updated_at\": \"$(date -Is)\","
    echo "  \"phases\": {"
    local first=1
    for k in "${!RESULT[@]}"; do
      [[ $first -eq 0 ]] && echo ","
      printf '    "%s": "%s"' "$k" "${RESULT[$k]}"
      first=0
    done
    echo ""
    echo "  }"
    echo "}"
  } > "$STATUS"
}

# 프로세스 트리 사살. conda run -> vllm -> VLLM::EngineCore 처럼 cmdline 이
# 바뀌는 자식까지 확실히 잡는다.
kill_tree() {
  local pid="${1:-}" child
  [[ -z "$pid" ]] && return 0
  for child in $(ps -o pid= --ppid "$pid" 2>/dev/null); do
    kill_tree "$child"
  done
  kill -9 "$pid" 2>/dev/null || true
}

# 우리가 만든 GPU 잔여물만 정리한다.
#  - 시작 전부터 GPU 를 쓰던 PID 는 보호한다.
#  - 그 사이 남이 띄운 것도 죽이지 않는다. 우리 워크로드 cmdline 패턴에만 손댄다.
reap_our_gpu() {
  local waited=0 p cmd leftover i
  for i in $(seq 1 45); do
    leftover=""
    for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
      [[ "$PRE_GPU_PIDS" == *" $p "* ]] && continue
      leftover="$leftover $p"
    done
    if [[ -z "$leftover" ]]; then
      echo "   우리 GPU 프로세스 없음 (${i}x2s)"
      return 0
    fi
    waited=$((waited + 2))
    if [[ $waited -ge 30 ]]; then
      for p in $leftover; do
        cmd=$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null)
        if echo "$cmd" | grep -qE "VLLM::EngineCore|vllm|backends/server\.py|streaming_websocket_server|Voxtral"; then
          echo "   잔여 정리: pid=$p  ${cmd:0:100}"
          kill -9 "$p" 2>/dev/null || true
        else
          echo "   !! 우리 것이 아닌 GPU 프로세스는 남긴다: pid=$p  ${cmd:0:100}"
        fi
      done
      waited=0
    fi
    sleep 2
  done
  return 1
}

stop_gpu_watch() {
  if [[ -n "${GPU_WATCH_PID:-}" ]]; then
    kill "$GPU_WATCH_PID" 2>/dev/null || true
    GPU_WATCH_PID=""
  fi
}

# 어젯밤 런은 Phase D 도중 남의 프로세스가 19.28GiB 를 잡는 바람에 AST 서버의
# translation 이 CUDA OOM 재시도로 3700 초를 블록했고, final 이 클라이언트가 끊긴
# 뒤에 나가 전 클립이 빈 전사로 채점됐다. 시작 시점 게이트만으로는 못 잡는다.
# 30 초마다 점유 상황을 남겨, 결과를 나중에 해석할 수 있게 한다.
start_gpu_watch() {
  ( while true; do
      printf '%s  ' "$(date +%H:%M:%S)"
      nvidia-smi --query-compute-apps=pid,used_gpu_memory --format=csv,noheader \
        | tr '\n' ';'
      printf '  free=%s\n' "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader)"
      sleep 30
    done ) > "$RUN/gpu_watch.log" 2>&1 &
  GPU_WATCH_PID=$!
}

teardown() {
  stop_gpu_watch
  local rc=$?
  echo ""
  echo "--------------------------------------------------------------"
  echo " teardown: 우리가 띄운 것만 내리고 GPU 반납"
  echo "--------------------------------------------------------------"
  kill_tree "${NEMO_PID:-}"
  kill_tree "${VOX_PID:-}"
  kill_tree "${Q_PID:-}"
  pkill -9 -f "backends/server.py"                2>/dev/null || true
  pkill -9 -f "$VOX_MODEL"                        2>/dev/null || true
  pkill -9 -f "streaming_websocket_server_ast.py" 2>/dev/null || true
  reap_our_gpu
  echo "  최종 GPU 상태:"
  nvidia-smi --query-gpu=memory.used,memory.free --format=csv | sed 's/^/    /'
  nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv | sed 's/^/    /'
  record "teardown" "done(exit=$rc)"
  write_status
  date -Is > "$RUN/DONE"
  echo ""
  echo "=============================================================="
  echo " 결과: $RUN"
  cat "$STATUS"
  echo "=============================================================="
}
# HUP 포함: tmux kill-session 은 SIGHUP 을 보낸다. 트랩에 없으면 GPU 를 문 채 죽는다.
trap teardown EXIT INT TERM HUP

# 포트가 비어 있는가. 남이 쓰는 포트에는 절대 서버를 올리지 않는다.
port_free() {
  ! (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null
}

# VRAM 이 빌 때까지 기다린다. 남의 작업을 밀어내지 않기 위한 장치다.
#   0 = 확보됨, 1 = 시간 내 확보 실패
wait_vram() {   # <needed_mib> <max_wait_sec>
  local need="$1" max="$2" waited=0 free
  while :; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null)
    if [[ -n "$free" && "$free" -ge "$need" ]]; then
      echo "   VRAM ${free}MiB >= ${need}MiB 확보 (대기 ${waited}s)"
      return 0
    fi
    if [[ "$waited" -ge "$max" ]]; then
      echo "   VRAM 부족: ${free:-?}MiB < ${need}MiB, ${max}s 기다린 뒤 포기"
      return 1
    fi
    [[ $((waited % 600)) -eq 0 ]] && echo "   VRAM 대기 중: ${free:-?}MiB < ${need}MiB (${waited}s 경과)"
    sleep 60
    waited=$((waited + 60))
  done
}

# wait_ws <host> <port> <timeout_sec> [watch_pid]
#   0 = 열림, 1 = 타임아웃, 2 = 서버가 죽음
wait_ws() {
  local host="$1" port="$2" to="$3" pid="${4:-}" i
  for ((i = 0; i < to; i++)); do
    if (exec 3<>"/dev/tcp/$host/$port") 2>/dev/null; then
      return 0
    fi
    if [[ -n "$pid" ]] && ! kill -0 "$pid" 2>/dev/null; then
      echo "  !! 서버 프로세스($pid)가 ${i}초 만에 죽었다 - 기다리지 않는다"
      return 2
    fi
    sleep 1
  done
  return 1
}

# =========================================================================
# PHASE 0 - 워크트리 + 의존성
# =========================================================================
echo ""; echo "### PHASE 0: 워크트리 / 의존성"

if [[ ! -d "$BE" ]]; then
  mkdir -p "$(dirname "$WT")"
  git -C "$REPO" worktree add "$WT" "$BRANCH" > "$RUN/worktree.log" 2>&1 \
    && record "worktree" "생성됨 ($WT)" \
    || record "worktree" "FAILED (worktree.log)"
else
  record "worktree" "기존 것 사용 ($WT)"
fi
if [[ ! -d "$BE" ]]; then
  echo "!! 백엔드 코드가 없다. 중단."; write_status; exit 1
fi
echo "  HEAD: $(git -C "$WT" log --oneline -1 2>/dev/null)"

conda run -n asr-nemotron pip install -q librosa accelerate > "$RUN/deps_nemotron.log" 2>&1 \
  && record "deps.nemotron" "ok" || record "deps.nemotron" "FAILED (deps_nemotron.log)"

if LD_LIBRARY_PATH="$VOX_LIB" conda run -n asr-voxtral python -c "import vllm; print(vllm.__version__)" \
     > "$RUN/deps_voxtral.log" 2>&1; then
  record "deps.voxtral" "ok (vllm $(tail -1 "$RUN/deps_voxtral.log"))"
  VOX_OK=1
else
  record "deps.voxtral" "FAILED - vllm 임포트 불가 (deps_voxtral.log)"
  VOX_OK=0
fi
write_status

KO_WAV=$(ls "$HOME/STiTy-team/datasets/fleurs/data/ko_kr/audio/test/"*.wav 2>/dev/null | head -1)
EN_WAV=$(ls "$HOME/STiTy-team/datasets/fleurs/data/en_us/audio/test/"*.wav 2>/dev/null | head -1)
echo "  probe 오디오: ko=$KO_WAV"
echo "                en=$EN_WAV"
if [[ -z "$KO_WAV" || -z "$EN_WAV" ]]; then
  record "data" "FAILED - FLEURS wav 를 못 찾음"
  write_status; exit 1
fi
record "data" "ok (ko/en wav 확인)"
echo "  시작 시점 GPU:"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader | sed 's/^/    /'

# =========================================================================
# PHASE B - Nemotron
# =========================================================================
start_gpu_watch
echo ""; echo "### PHASE B: Nemotron"

timeout 900 conda run -n asr-nemotron python "$BE/engine_nemotron.py" \
    --probe "$KO_WAV" --lang ko-KR --right-context 13 \
    > "$RUN/nemotron_probe_ko.json" 2> "$RUN/nemotron_probe_ko.err"
NEMO_PROBE=$?
if [[ $NEMO_PROBE -eq 0 ]]; then
  record "nemotron.probe" "ok"
else
  record "nemotron.probe" "FAILED (nemotron_probe_ko.json/.err)"
fi
write_status
kill_tree "${NEMO_PID:-}" 2>/dev/null || true
NEMO_PID=""

run_nemotron_case() {   # <right_context> <lang(ko|en)> <n> <tag>
  local rc="$1" lang="$2" n="$3" tag="$4"
  local log="$RUN/nemotron_${tag}_server.log"
  # smoke_client 는 언제나 lang="auto" 를 보낸다. 로케일은 서버에서 정해야 한다.
  local locale="ko-KR"
  [[ "$lang" == "en" ]] && locale="en-US"
  echo ""; echo "-- nemotron rc=$rc lang=$lang locale=$locale n=$n port=$NEMO_PORT"
  if ! port_free "$NEMO_PORT"; then
    record "nemotron.$tag" "SKIP - 포트 $NEMO_PORT 를 남이 쓰고 있다"
    write_status; return
  fi
  if ! wait_vram "$NEED_NEMO" "$GATE_WAIT"; then
    record "nemotron.$tag" "SKIP - VRAM 확보 실패"
    write_status; return
  fi
  conda run --no-capture-output -n asr-nemotron python "$BE/server.py" --backend nemotron \
      --lang "$locale" --right-context "$rc" --port "$NEMO_PORT" --log-file "$log" \
      > "$log.stdout" 2>&1 &
  NEMO_PID=$!
  wait_ws 127.0.0.1 "$NEMO_PORT" 600 "$NEMO_PID"
  case $? in
    0)
      timeout 5400 conda run -n asr-nemotron python "$BE/smoke_client.py" \
          --ws "ws://127.0.0.1:$NEMO_PORT" --lang "$lang" --limit "$n" \
          --tag "nemotron_$tag" --out "$RUN/nemotron_${tag}.json" \
          > "$RUN/nemotron_${tag}_client.log" 2>&1 \
        && record "nemotron.$tag" "ok" || record "nemotron.$tag" "client FAILED"
      ;;
    2) record "nemotron.$tag" "서버가 즉시 죽음 (${tag}_server.log.stdout)" ;;
    *) record "nemotron.$tag" "600s 안에 안 뜸" ;;
  esac
  kill_tree "$NEMO_PID"
  NEMO_PID=""
  pkill -9 -f "backends/server.py" 2>/dev/null || true
  sleep 5
  write_status
}

if [[ $NEMO_PROBE -eq 0 ]]; then
  run_nemotron_case 13 ko "$SMOKE_N" "smoke_ko"
  run_nemotron_case 13 en "$SMOKE_N" "smoke_en"
  run_nemotron_case 3  ko "$SWEEP_N" "sweep_rc3_320ms"
  run_nemotron_case 6  ko "$SWEEP_N" "sweep_rc6_560ms"
  run_nemotron_case 13 ko "$SWEEP_N" "sweep_rc13_1120ms"
else
  record "nemotron.smoke" "skipped (probe 실패)"
fi
reap_our_gpu

# =========================================================================
# PHASE C - Voxtral
# =========================================================================
echo ""; echo "### PHASE C: Voxtral"

VOX_PID=""
if [[ "$VOX_OK" -ne 1 ]]; then
  record "voxtral.serve" "skipped (vllm 임포트 실패)"
elif ! port_free "$VOX_PORT"; then
  record "voxtral.serve" "SKIP - 포트 $VOX_PORT 를 남이 쓰고 있다"
elif ! wait_vram "$NEED_VOX" "$GATE_WAIT"; then
  record "voxtral.serve" "SKIP - VRAM ${NEED_VOX}MiB 확보 실패 (남의 작업을 밀어내지 않는다)"
else
  VOX_LOG="$RUN/voxtral_vllm.log"
  # 2026-09-09 확정된 두 가지:
  #  - util 0.60 + 기본 max_model_len 131072 이면 KV cache 예산이 -5.25GiB 가 되어
  #    "No available memory for the cache blocks" 로 죽는다. 0.85 + 16384 로 +9.87GiB.
  #  - 이 박스에 nvcc 가 없다. flashinfer 의 top-k/top-p 샘플러가 JIT 로 CUDA 커널을
  #    빌드하려 해서 EngineCore 가 죽는다. 샘플러만 끄면 되고 컴파일/CUDA 그래프는
  #    정상 동작한다.
  LD_LIBRARY_PATH="$VOX_LIB" VLLM_DISABLE_COMPILE_CACHE=1 VLLM_USE_FLASHINFER_SAMPLER=0 \
  conda run --no-capture-output -n asr-voxtral \
    vllm serve "$VOX_MODEL" \
      --tokenizer-mode mistral \
      --gpu-memory-utilization 0.85 \
      --max-model-len 16384 \
      --compilation_config '{"cudagraph_mode": "PIECEWISE"}' \
      --port "$VOX_PORT" > "$VOX_LOG" 2>&1 &
  VOX_PID=$!

  # 가중치 ~9GB 최초 다운로드가 포함될 수 있어 넉넉히 준다.
  # 프로세스가 죽으면 기다리지 않고 즉시 빠진다 (지난 런은 여기서 40분을 버렸다).
  wait_ws 127.0.0.1 "$VOX_PORT" 3600 "$VOX_PID"
  case $? in
    0)
      record "voxtral.serve" "ok"
      sleep 10
      for L in ko en; do
        if [[ "$L" == "ko" ]]; then W="$KO_WAV"; else W="$EN_WAV"; fi
        timeout 900 env LD_LIBRARY_PATH="$VOX_LIB" conda run -n asr-voxtral python "$BE/probe_voxtral.py" \
            --audio "$W" --lang "$L" --base-url "http://127.0.0.1:$VOX_PORT" \
            --out "$RUN/voxtral_probe_${L}.json" \
            > "$RUN/voxtral_probe_${L}.log" 2>&1 \
          && record "voxtral.probe.$L" "ok (전사 성공)" \
          || record "voxtral.probe.$L" "전사 실패 - 이벤트 덤프는 voxtral_probe_${L}.json"
      done
      write_status
      # 실제 숫자. smoke_voxtral 은 smoke_client 의 채점/데이터 로더를 그대로 쓰므로
      # 요약 JSON 형식이 nemotron/qwen3 과 같다 -> 요약 표에서 한 줄로 비교된다.
      for L in ko en; do
        timeout 5400 env LD_LIBRARY_PATH="$VOX_LIB" conda run --no-capture-output -n asr-voxtral \
          python "$BE/smoke_voxtral.py" \
            --base-url "http://127.0.0.1:$VOX_PORT" --model "$VOX_MODEL" \
            --lang "$L" --limit "$SMOKE_N" --tag "voxtral" \
            --out "$RUN/voxtral_smoke_${L}.json" \
            > "$RUN/voxtral_smoke_${L}_client.log" 2>&1 \
          && record "voxtral.smoke_$L" "ok" \
          || record "voxtral.smoke_$L" "FAILED (voxtral_smoke_${L}_client.log)"
        write_status
      done
      ;;
    2) record "voxtral.serve" "FAILED - 프로세스가 즉시 죽음 (voxtral_vllm.log)" ;;
    *) record "voxtral.serve" "FAILED - 3600s 안에 안 뜸 (voxtral_vllm.log)" ;;
  esac
  tail -80 "$VOX_LOG" > "$RUN/voxtral_vllm_tail.log" 2>/dev/null || true
  kill_tree "$VOX_PID"
  VOX_PID=""
  pkill -9 -f "$VOX_MODEL" 2>/dev/null || true
  reap_our_gpu
fi
write_status

# =========================================================================
# PHASE D - Qwen3 기준선 (공용 체크아웃에서 기동. 여긴 .env/모델이 있다)
# =========================================================================
echo ""; echo "### PHASE D: Qwen3 기준선"

Q_LOG="$RUN/qwen3_server.log"
Q_PID=""
cd "$REPO" || exit 1
# 다른 세션이 8765 에 같은 AST 서버를 돌리고 있다. 우리 것은 별도 포트에 띄우고,
# 포트가 막혀 있으면 붙지 않는다 - 남의 서버 결과를 우리 것으로 적으면 안 된다.
if ! port_free "$QWEN_PORT"; then
  record "qwen3.serve" "SKIP - 포트 $QWEN_PORT 를 남이 쓰고 있다"
elif ! wait_vram "$NEED_QWEN" "$GATE_WAIT"; then
  record "qwen3.serve" "SKIP - VRAM ${NEED_QWEN}MiB 확보 실패"
else
conda run --no-capture-output -n stity python "$REPO/evaluation/streaming_websocket_server_ast.py" \
    --no-idle-shutdown --port "$QWEN_PORT" > "$Q_LOG" 2>&1 &
Q_PID=$!
wait_ws 127.0.0.1 "$QWEN_PORT" 900 "$Q_PID"
case $? in
  0)
    record "qwen3.serve" "ok"
    # 지난 런의 빈 전사 원인 규명용. 채점 전에 원본 메시지 스키마부터 남긴다.
    if [[ -f "$RAW_PROBE" ]]; then
      conda run -n asr-nemotron python "$RAW_PROBE" \
          --ws "ws://127.0.0.1:$QWEN_PORT" --out "$RUN/qwen3_raw_messages.json" \
          > "$RUN/qwen3_raw_probe.log" 2>&1 \
        && record "qwen3.raw_probe" "ok" || record "qwen3.raw_probe" "FAILED (qwen3_raw_probe.log)"
    fi
    for L in ko en; do
      if ! kill -0 "$Q_PID" 2>/dev/null; then
        record "qwen3.serve" "중간에 죽음 (qwen3_server.log)"
        break
      fi
      timeout 5400 conda run -n asr-nemotron python "$BE/smoke_client.py" \
          --ws "ws://127.0.0.1:$QWEN_PORT" --lang "$L" --limit "$SMOKE_N" \
          --tag "qwen3_$L" --out "$RUN/qwen3_smoke_${L}.json" \
          > "$RUN/qwen3_smoke_${L}_client.log" 2>&1 \
        && record "qwen3.smoke_$L" "ok" || record "qwen3.smoke_$L" "FAILED"
    done
    tail -60 "$Q_LOG" > "$RUN/qwen3_server_tail.log" 2>/dev/null || true
    ;;
  2) record "qwen3.serve" "FAILED - 프로세스가 즉시 죽음 (qwen3_server.log)" ;;
  *) record "qwen3.serve" "FAILED - 900s 안에 안 뜸" ;;
esac
kill_tree "$Q_PID"
Q_PID=""
fi
write_status

# =========================================================================
echo ""; echo "### 요약 표"
python3 - "$RUN" <<'PY'
import json, sys, glob, os
run = sys.argv[1]
rows = []
for p in sorted(glob.glob(os.path.join(run, "*.json"))):
    if os.path.basename(p) in ("status.json", "qwen3_raw_messages.json"):
        continue
    try:
        d = json.load(open(p, encoding="utf-8"))
    except Exception:
        continue
    s = d.get("summary")
    if s:
        rows.append(s)
if not rows:
    print("  (요약 가능한 결과 없음)")
else:
    hdr = "%-28s %-5s %-5s %8s %8s %7s %6s %8s" % (
        "backend", "lang", "unit", "score", "1stTok", "FSL", "RTF", "n")
    print(hdr)
    print("-" * len(hdr))
    for s in rows:
        unit = s.get("unit", "")
        def num(v, f):
            return (f % v) if isinstance(v, (int, float)) else "-"
        print("%-28s %-5s %-5s %8s %8s %7s %6s %8s" % (
            s.get("backend", ""), s.get("lang", ""), unit,
            num(s.get("avg_" + unit), "%.4f"),
            num(s.get("avg_first_token_latency_sec"), "%.3f"),
            num(s.get("avg_fsl_sec"), "%.3f"),
            num(s.get("rtf"), "%.2f"),
            "%s/%s" % (s.get("n_ok"), s.get("n_total"))))
PY

record "summary" "done"
echo ""
echo "완료. teardown 이 이어서 GPU 를 반납한다."
