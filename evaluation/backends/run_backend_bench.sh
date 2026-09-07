#!/usr/bin/env bash
# Voxtral / Nemotron / Qwen3 백엔드 무인 벤치마크.
#
# 설계 원칙 셋:
#   1) 한 번 기동되면 추가 승인 없이 끝까지 간다 (사용자가 자리에 없다).
#   2) 백엔드 하나가 죽어도 나머지는 계속한다.
#   3) 어떤 경로로 끝나든 GPU 를 반납한다  -> trap EXIT.
#
#   bash run_backend_bench.sh [--skip-setup] [--smoke-n 20] [--sweep-n 50]

set -uo pipefail          # -e 는 쓰지 않는다. 실패를 삼키지 않고 기록하며 계속 간다.

REPO="$HOME/STiTy-team/STiTy"
BE="$REPO/evaluation/backends"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN="$REPO/evaluation/backends/results/$STAMP"
CONDA_SH="$HOME/miniforge3/etc/profile.d/conda.sh"
STATUS="$RUN/status.json"

SMOKE_N=20
SWEEP_N=50
SKIP_SETUP=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-setup) SKIP_SETUP=1; shift ;;
    --smoke-n)    SMOKE_N="$2"; shift 2 ;;
    --sweep-n)    SWEEP_N="$2"; shift 2 ;;
    *) echo "unknown arg $1"; exit 2 ;;
  esac
done

mkdir -p "$RUN"
exec > >(tee -a "$RUN/driver.log") 2>&1

echo "=============================================================="
echo " backend bench  $STAMP"
echo " run dir: $RUN"
echo "=============================================================="

# shellcheck disable=SC1090
source "$CONDA_SH"

# ── 상태 기록 ────────────────────────────────────────────────────────────
declare -A RESULT
record() { RESULT["$1"]="$2"; echo "  >> [$1] $2"; }

write_status() {
  { echo "{"
    echo "  \"run\": \"$STAMP\","
    echo "  \"finished_at\": \"$(date -Is)\","
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

# ── GPU 반납 (무슨 일이 있어도) ──────────────────────────────────────────
teardown() {
  local rc=$?
  echo ""
  echo "──────────────────────────────────────────────────────────────"
  echo " teardown: 모델 내리고 GPU 반납"
  echo "──────────────────────────────────────────────────────────────"
  pkill -9 -f "vllm"                        2>/dev/null || true
  pkill -9 -f "backends/server.py"          2>/dev/null || true
  pkill -9 -f "streaming_websocket_server"  2>/dev/null || true
  pkill -9 -f "VLLM::EngineCore"            2>/dev/null || true
  for i in $(seq 1 45); do
    n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
    if [[ "$n" -eq 0 ]]; then echo "  GPU 해제 완료 (${i}x2s)"; break; fi
    sleep 2
  done
  echo "  최종 GPU 상태:"
  nvidia-smi --query-gpu=memory.used,memory.free --format=csv | sed 's/^/    /'
  nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv | sed 's/^/    /'
  record "teardown" "done(exit=$rc)"
  write_status
  echo ""
  echo "=============================================================="
  echo " 결과: $RUN"
  cat "$STATUS"
  echo "=============================================================="
}
trap teardown EXIT INT TERM

wait_ws() {   # wait_ws <host:port> <timeout_sec> <logfile>
  local hp="$1" to="$2" lf="${3:-}"
  local host="${hp%%:*}" port="${hp##*:}"
  for i in $(seq 1 "$to"); do
    if python3 -c "import socket,sys; s=socket.socket(); s.settimeout(1); sys.exit(0 if s.connect_ex(('$host',$port))==0 else 1)" 2>/dev/null; then
      return 0
    fi
    if [[ -n "$lf" && -f "$lf" ]] && grep -qiE "traceback|error while|failed to|out of memory" "$lf" 2>/dev/null; then
      echo "  !! 서버 로그에 오류 감지 (계속 대기)"
    fi
    sleep 1
  done
  return 1
}

# ══════════════════════════════════════════════════════════════════════════
# PHASE A - 환경 준비
# ══════════════════════════════════════════════════════════════════════════
if [[ "$SKIP_SETUP" -eq 0 ]]; then
  echo ""; echo "### PHASE A: env 준비"

  if ! conda env list | grep -q "^asr-nemotron "; then
    echo "-- asr-nemotron 생성"
    conda create -y -n asr-nemotron python=3.12 > "$RUN/setup_nemotron.log" 2>&1
  fi
  conda run -n asr-nemotron pip install -q --upgrade \
      "transformers>=5.13.0" torch soundfile numpy websockets aiohttp \
      >> "$RUN/setup_nemotron.log" 2>&1 \
    && record "setup.nemotron" "ok" || record "setup.nemotron" "FAILED (see setup_nemotron.log)"

  if ! conda env list | grep -q "^asr-voxtral "; then
    echo "-- asr-voxtral 생성"
    conda create -y -n asr-voxtral python=3.12 > "$RUN/setup_voxtral.log" 2>&1
  fi
  conda run -n asr-voxtral pip install -q --upgrade \
      "vllm>=0.20.0" "mistral-common[audio]>=1.9.0" transformers \
      soundfile numpy websockets aiohttp \
      >> "$RUN/setup_voxtral.log" 2>&1 \
    && record "setup.voxtral" "ok" || record "setup.voxtral" "FAILED (see setup_voxtral.log)"
else
  record "setup" "skipped"
fi
write_status

KO_WAV=$(ls "$HOME/STiTy-team/datasets/fleurs/data/ko_kr/audio/test/"*.wav 2>/dev/null | head -1)
EN_WAV=$(ls "$HOME/STiTy-team/datasets/fleurs/data/en_us/audio/test/"*.wav 2>/dev/null | head -1)
echo "  probe 오디오: ko=$KO_WAV"
echo "                en=$EN_WAV"

# ══════════════════════════════════════════════════════════════════════════
# PHASE B - Nemotron: probe -> smoke -> sweep
# ══════════════════════════════════════════════════════════════════════════
echo ""; echo "### PHASE B: Nemotron"

conda run -n asr-nemotron python "$BE/engine_nemotron.py" \
    --probe "$KO_WAV" --lang ko-KR --right-context 13 \
    > "$RUN/nemotron_probe_ko.json" 2> "$RUN/nemotron_probe_ko.err"
NEMO_PROBE=$?
if [[ $NEMO_PROBE -eq 0 ]]; then
  record "nemotron.probe" "ok"
else
  record "nemotron.probe" "FAILED (nemotron_probe_ko.json/.err 에 API 표면 기록됨)"
fi
write_status

run_nemotron_case() {   # run_nemotron_case <right_context> <lang> <n> <tag>
  local rc="$1" lang="$2" n="$3" tag="$4"
  local log="$RUN/nemotron_${tag}_server.log"
  echo ""; echo "-- nemotron rc=$rc lang=$lang n=$n"
  conda run -n asr-nemotron python "$BE/server.py" --backend nemotron \
      --right-context "$rc" --port 8765 --log-file "$log" > "$log.stdout" 2>&1 &
  local pid=$!
  if wait_ws "127.0.0.1:8765" 600 "$log"; then
    conda run -n asr-nemotron python "$BE/smoke_client.py" \
        --ws ws://127.0.0.1:8765 --lang "$lang" --limit "$n" \
        --tag "nemotron_$tag" --out "$RUN/nemotron_${tag}.json" \
        > "$RUN/nemotron_${tag}_client.log" 2>&1 \
      && record "nemotron.$tag" "ok" || record "nemotron.$tag" "client FAILED"
  else
    record "nemotron.$tag" "server did not start in 600s"
  fi
  kill -9 "$pid" 2>/dev/null || true
  pkill -9 -f "backends/server.py" 2>/dev/null || true
  sleep 5
  write_status
}

if [[ $NEMO_PROBE -eq 0 ]]; then
  run_nemotron_case 13 ko "$SMOKE_N" "smoke_ko"
  run_nemotron_case 13 en "$SMOKE_N" "smoke_en"
  # 지연 스윕 (한국어). 160ms / 560ms / 1.12s
  run_nemotron_case 1  ko "$SWEEP_N" "sweep_rc1_160ms"
  run_nemotron_case 6  ko "$SWEEP_N" "sweep_rc6_560ms"
  run_nemotron_case 13 ko "$SWEEP_N" "sweep_rc13_1120ms"
else
  record "nemotron.smoke" "skipped (probe 실패)"
fi

# ══════════════════════════════════════════════════════════════════════════
# PHASE C - Voxtral: vllm serve -> realtime 스키마 프로브
# ══════════════════════════════════════════════════════════════════════════
echo ""; echo "### PHASE C: Voxtral"

VOX_LOG="$RUN/voxtral_vllm.log"
VLLM_DISABLE_COMPILE_CACHE=1 conda run -n asr-voxtral \
  vllm serve mistralai/Voxtral-Mini-4B-Realtime-2602 \
    --tokenizer-mode mistral \
    --compilation_config '{"cudagraph_mode": "PIECEWISE"}' \
    --port 8000 > "$VOX_LOG" 2>&1 &
VOX_PID=$!

if wait_ws "127.0.0.1:8000" 2400 "$VOX_LOG"; then
  record "voxtral.serve" "ok"
  sleep 10
  for L in ko en; do
    W=$([[ "$L" == ko ]] && echo "$KO_WAV" || echo "$EN_WAV")
    conda run -n asr-voxtral python "$BE/probe_voxtral.py" \
        --audio "$W" --lang "$L" --base-url http://127.0.0.1:8000 \
        --out "$RUN/voxtral_probe_${L}.json" \
        > "$RUN/voxtral_probe_${L}.log" 2>&1 \
      && record "voxtral.probe.$L" "ok (전사 성공)" \
      || record "voxtral.probe.$L" "전사 실패 - 이벤트 덤프는 voxtral_probe_${L}.json 에 있음"
  done
else
  record "voxtral.serve" "FAILED - 2400s 안에 안 뜸 (voxtral_vllm.log)"
  tail -50 "$VOX_LOG" > "$RUN/voxtral_vllm_tail.log" 2>/dev/null || true
fi
kill -9 "$VOX_PID" 2>/dev/null || true
pkill -9 -f "vllm" 2>/dev/null || true
for i in $(seq 1 45); do
  n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
  [[ "$n" -eq 0 ]] && break
  sleep 2
done
write_status

# ══════════════════════════════════════════════════════════════════════════
# PHASE D - Qwen3 기준선 (기존 서버, stity env)
# ══════════════════════════════════════════════════════════════════════════
echo ""; echo "### PHASE D: Qwen3 기준선"

Q_LOG="$RUN/qwen3_server.log"
cd "$REPO" || exit 1
conda run -n stity python "$REPO/evaluation/streaming_websocket_server_ast.py" \
    --no-idle-shutdown --port 8765 > "$Q_LOG" 2>&1 &
Q_PID=$!
if wait_ws "127.0.0.1:8765" 900 "$Q_LOG"; then
  record "qwen3.serve" "ok"
  for L in ko en; do
    conda run -n asr-nemotron python "$BE/smoke_client.py" \
        --ws ws://127.0.0.1:8765 --lang "$L" --limit "$SMOKE_N" \
        --tag "qwen3_$L" --out "$RUN/qwen3_smoke_${L}.json" \
        > "$RUN/qwen3_smoke_${L}_client.log" 2>&1 \
      && record "qwen3.smoke_$L" "ok" || record "qwen3.smoke_$L" "FAILED"
  done
else
  record "qwen3.serve" "FAILED - 900s 안에 안 뜸"
fi
kill -9 "$Q_PID" 2>/dev/null || true
write_status

# ══════════════════════════════════════════════════════════════════════════
# 요약
# ══════════════════════════════════════════════════════════════════════════
echo ""; echo "### 요약 표"
python3 - "$RUN" <<'PY'
import json, sys, glob, os
run = sys.argv[1]
rows = []
for p in sorted(glob.glob(os.path.join(run, "*.json"))):
    name = os.path.basename(p)
    if name == "status.json":
        continue
    try:
        d = json.load(open(p, encoding="utf-8"))
    except Exception:
        continue
    s = d.get("summary")
    if not s:
        continue
    rows.append(s)
if not rows:
    print("  (요약 가능한 결과 없음)")
else:
    hdr = f"{'backend':28} {'lang':5} {'unit':5} {'score':>8} {'1stTok':>8} {'FSL':>7} {'RTF':>6} {'n':>6}"
    print(hdr); print("-" * len(hdr))
    for s in rows:
        unit = s.get("unit", "")
        print(f"{s.get('backend',''):28} {s.get('lang',''):5} {unit:5} "
              f"{(s.get('avg_'+unit) if s.get('avg_'+unit) is not None else float('nan')):>8.4f} "
              f"{(s.get('avg_first_token_latency_sec') or float('nan')):>8.3f} "
              f"{(s.get('avg_fsl_sec') or float('nan')):>7.3f} "
              f"{(s.get('rtf') or float('nan')):>6.2f} "
              f"{str(s.get('n_ok'))+'/'+str(s.get('n_total')):>6}")
PY

record "summary" "done"
echo ""
echo "완료. teardown 이 이어서 GPU 를 반납한다."
