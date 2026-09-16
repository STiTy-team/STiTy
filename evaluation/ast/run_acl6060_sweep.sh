#!/usr/bin/env bash
# ACL 60/60 — 분절 정책 8개 config 를 한 태그로 쓸어 담는다.
#
#   tmux new-session -d -s acl_sweep -c ~/STiTy \
#     'bash evaluation/ast/run_acl6060_sweep.sh eval 2>&1 | tee /tmp/acl_sweep.log'
#
# config 하나가 죽어도 나머지는 이어서 돈다
# ----------------------------------------
# 카드가 한 장이라 config 는 순서대로 돌 수밖에 없다. 그래서 "따로 띄운다"가 아니라
# **"앞 config 의 실패가 뒤 config 를 막지 않는다"** 로 격리한다. 셋을 지킨다.
#
#   1. config 하나를 서브셸에서 돌리고 실패해도 스윕은 다음으로 넘어간다.
#   2. config 마다 시간 상한을 둔다 — 서버가 매달리면 그 config 만 버리고 넘어간다.
#      상한에 걸려 죽은 자리에는 vLLM 서버가 남으므로 매번 stop_server.sh 로 회수한다.
#   3. 끝난 config 는 마커 파일로 남긴다. 스윕 자체가 죽어도 **다시 띄우면 남은 것만**
#      돈다. PID 로 순서를 잡지 않는 이유와 같다 — 세션이 죽으면 PID 는 의미가 없다.
#
# 8 config (타깃은 셋 다 de/ja/zh)
# ------------------------------
#   seg-c0.5   <SEG> 커밋, 0.5초 청크
#   seg-c1     <SEG> 커밋, 1초 청크
#   segdot-c1  <SEG> + 구두점 둘 다 커밋 트리거, 1초 청크
#              (`--ast-hide-seg` 를 빼는 유일한 축 — run_acl6060.sh 참조)
#   static-c3..c6  매 청크 커밋, 3/4/5/6초
#   punct-c1   구두점 전용 커밋, 1초 청크 (`<SEG>` 는 파싱 단계에서 숨긴다).
#              segdot-c1 과 청크가 같아 차이가 "`<SEG>` 를 트리거로 쓰는가" 로 좁혀진다
#              — 다만 이 축만 `--no-rep-dedup` 이 붙는다(기존 punct 축 정의 그대로)
#
# 환경변수
#   STAMP     기본 실행 시각. 8 config 가 이 태그 하나를 공유한다
#   MODEL     ASR 가중치 경로
#   LANGS     기본 "de ja zh"
#   PORT      기본 8765
#   GPU_UTIL  기본 0.5 (로컬 MADLAD 번역기 7.2GiB 와 나눠 쓴다)
#   TIMEOUT   config 하나의 시간 상한(초). 기본 10800 = 3시간
#   ONLY      돌릴 config 라벨을 공백으로 나열하면 그것만 돈다 (재실행용)
#   SWEEP_CONFIGS  "축:청크" 목록을 통째로 바꾼다 (예: "seg:1.0 static:8.0")
#   EXTRA_SERVER_ARGS  서버에 그대로 넘긴다 (예: --speech-start-gate)
set -u

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
STOP="$REPO/evaluation/LibriSpeech/paper_result/ASR/scripts/stop_server.sh"

SPLIT="${1:-eval}"
export STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
export MODEL="${MODEL:-$REPO/models/Qwen3-ASR-1.7B-en-covost2-dailytalk-mix-c200-merged}"
export LANGS="${LANGS:-de ja zh}"
export PORT="${PORT:-8765}"
export GPU_UTIL="${GPU_UTIL:-0.5}"
TIMEOUT="${TIMEOUT:-10800}"
ONLY="${ONLY:-}"

# ── 서버 수정 스위치 — 셋 다 기본이 꺼져 있어 여기서 명시적으로 켠다 ──────────────
# 전부 "정책"이 아니라 **버그 수정**이고, 8 config 에 똑같이 적용되므로 축 간 비교를
# 왜곡하지 않는다. 끄고 돌리려면 각각 0 을 주면 된다.
#
#   AST_CAP_FREEZE           128토큰 상한에서 생성이 교착하면 prefix 를 굳힌다.
#                            끄면 이 데이터셋에서 꼬리가 통째로 사라진다
#                            (실측 ACL60/60 talk110: 결손 210.6초 → 1.0초).
#   AST_AUDIO_END_AT_COMMIT  SEG 커밋의 audio_end 역추정을 "직전 커밋 끝~지금" 창과
#                            글자 비율로 다시 잡는다. 끄면 seg 축 지연이 **체계적으로
#                            과소평가**된다(실측 de 3.544s → 6.075s). 지금 비교하려는
#                            것이 seg 대 static 이므로 끄면 seg 가 공짜로 좋아 보인다.
#                            seg 사유 커밋에만 걸리므로 static/punct 는 영향이 없다.
#   AST_TRANS_ANTI_REPEAT    번역기 반복 생성 가드. ja 의 음수 지연 비율을
#                            11.82% → 1.63% 로 낮췄다. 타깃에 ja 가 있으므로 켠다.
export AST_CAP_FREEZE="${AST_CAP_FREEZE:-1}"
export AST_AUDIO_END_AT_COMMIT="${AST_AUDIO_END_AT_COMMIT:-1}"
export AST_TRANS_ANTI_REPEAT="${AST_TRANS_ANTI_REPEAT:-1}"

LOGDIR="$REPO/evaluation/ast/results/_runlogs/acl_${SPLIT}_$STAMP"
STATE="$LOGDIR/state"
mkdir -p "$STATE"

# "축:청크" — 라벨은 run_acl6060.sh 의 axis_label 이 만든다(청크 2.0 이면 접미사 없음).
CONFIGS=(
  "seg:0.5"
  "seg:1.0"
  "segdot:1.0"
  "static:3.0"
  "static:4.0"
  "static:5.0"
  "static:6.0"
  "punct:1.0"
)
# `SWEEP_CONFIGS="seg:1.0 static:6.0"` 처럼 주면 위 목록 대신 그것만 돈다. 같은 STAMP 로
# 나중에 config 를 덧붙일 때 쓴다(완료 마커가 같은 자리에 쌓인다).
if [ -n "${SWEEP_CONFIGS:-}" ]; then
  read -r -a CONFIGS <<< "$SWEEP_CONFIGS"
fi

label_of() {  # 축, 청크 → 결과 폴더 라벨
  local axis="$1" chunk="$2"
  if [ "$chunk" = "2.0" ]; then echo "$axis"; else echo "$axis-c${chunk%.0}"; fi
}

# ── 사전 점검 ─────────────────────────────────────────────────────────────
# 도중에 알면 몇 시간을 버린다. 시작 전에 전부 확인한다.
fail=0
[ -d "$MODEL" ] || { echo "!! 모델 경로가 없습니다: $MODEL"; fail=1; }
for lang in $LANGS; do
  man="$REPO/evaluation/ast/manifests/acl6060_${SPLIT}_en-${lang}.jsonl"
  [ -s "$man" ] || { echo "!! 매니페스트가 없습니다: $man"; fail=1; }
done
# 파이썬은 run_acl6060.sh 의 pick_python 과 같은 규칙으로 고른다 — `.venv` 가 평가용이
# 아닌 머신이 있어서 경로를 못 박으면 안 된다.
for c in "${PY:-}" "$REPO/.venv/bin/python" "$HOME/miniforge3/envs/stity/bin/python"; do
  [ -n "$c" ] && [ -x "$c" ] || continue
  "$c" -c "import vllm, websockets" 2>/dev/null && { export PY="$c"; break; }
done
if [ -z "${PY:-}" ] || ! "$PY" -c "import vllm, websockets" 2>/dev/null; then
  echo "!! vllm 과 websockets 를 갖춘 파이썬을 못 찾았습니다 (PY= 로 지정)"; fail=1
fi
[ "$fail" -eq 0 ] || exit 2

echo "═══════════════════════════════════════════════════════"
echo " ACL 60/60 스윕  split=$SPLIT  tag=$STAMP"
echo " 모델   : $(basename "$MODEL")"
echo " 언어   : $LANGS      포트: $PORT   GPU: $GPU_UTIL"
echo " 파이썬 : $PY"
echo " 스위치 : CAP_FREEZE=$AST_CAP_FREEZE AUDIO_END=$AST_AUDIO_END_AT_COMMIT ANTI_REPEAT=$AST_TRANS_ANTI_REPEAT"
echo " 서버인자: ${EXTRA_SERVER_ARGS:-(없음)}"
echo " config : ${#CONFIGS[@]}개, 하나당 상한 ${TIMEOUT}초"
echo " 로그   : $LOGDIR"
echo " 시작   : $(date '+%F %T')"
echo "═══════════════════════════════════════════════════════"
echo

for cfg in "${CONFIGS[@]}"; do
  axis="${cfg%%:*}"; chunk="${cfg##*:}"
  label="$(label_of "$axis" "$chunk")"

  if [ -n "$ONLY" ] && ! printf '%s\n' $ONLY | grep -qx "$label"; then
    echo "── [$label] ONLY 밖 — 건너뜀"; continue
  fi
  if [ -f "$STATE/$label.done" ]; then
    echo "── [$label] 이미 완료 — 건너뜀 ($(cat "$STATE/$label.done"))"; continue
  fi

  echo "╔══ [$label] 시작 $(date '+%F %T')  (축=$axis 청크=${chunk}s)"
  date '+%F %T' > "$STATE/$label.running"
  rm -f "$STATE/$label.failed"

  # 여기서 실패해도 스윕은 안 죽는다. 상한에 걸리면 timeout 이 끊는다.
  # 이 스크립트는 errexit 를 쓰지 않는다(`set -u` 만). 그러니 `set +e`/`set -e` 로
  # 감싸면 안 된다 — 복구 쪽이 없던 errexit 를 **켜서**, 뒤 config 의 사소한 nonzero
  # 하나에 스윕 전체가 죽는다. 격리하려던 것과 정반대가 된다.
  rc=0
  AXES="$axis" CHUNK="$chunk" \
    timeout --signal=TERM --kill-after=60 "$TIMEOUT" \
    bash "$REPO/evaluation/ast/run_acl6060.sh" "$SPLIT" || rc=$?

  # 서버 회수는 성공·실패와 무관하게 한다. timeout 으로 끊긴 자리에는 vLLM 이
  # 남아 다음 config 가 포트도 VRAM 도 못 잡는다.
  bash "$STOP" "$PORT" 2>&1 | tail -3
  sleep 10

  # run_acl6060.sh 는 축이 실패해도 종료코드 0 으로 끝난다(마지막 명령이 echo 다).
  # 그래서 rc 만 믿으면 서버가 한 번도 안 뜬 config 도 "완료" 로 찍힌다. 언어마다
  # metric.json 이 실제로 생겼는지 본다.
  for lang in $LANGS; do
    m="$REPO/evaluation/ast/results/ACL6060/$label/${SPLIT}-${lang}/$STAMP/metric.json"
    if [ ! -s "$m" ]; then
      echo "   !! [$label/$lang] metric.json 이 없다: $m"
      [ "$rc" -eq 0 ] && rc=65
    fi
  done

  rm -f "$STATE/$label.running"
  if [ "$rc" -eq 0 ]; then
    date '+%F %T' > "$STATE/$label.done"
    echo "╚══ [$label] 완료 $(date '+%F %T')"
  else
    echo "rc=$rc $(date '+%F %T')" > "$STATE/$label.failed"
    if [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; then
      echo "╚══ [$label] !! 시간 상한(${TIMEOUT}초) 초과 — 이 config 만 버리고 계속한다"
    else
      echo "╚══ [$label] !! 실패 rc=$rc — 이 config 만 버리고 계속한다"
    fi
  fi
  echo
done

echo "═══════════════ 스윕 요약 ═══════════════ $(date '+%F %T')"
for cfg in "${CONFIGS[@]}"; do
  axis="${cfg%%:*}"; chunk="${cfg##*:}"
  label="$(label_of "$axis" "$chunk")"
  if   [ -f "$STATE/$label.done" ];   then printf '  %-12s 완료   %s\n' "$label" "$(cat "$STATE/$label.done")"
  elif [ -f "$STATE/$label.failed" ]; then printf '  %-12s 실패   %s\n' "$label" "$(cat "$STATE/$label.failed")"
  else                                     printf '  %-12s 안 돌림\n' "$label"
  fi
done
echo
echo "결과: $REPO/evaluation/ast/results/ACL6060/*/${SPLIT}-*/$STAMP/metric.json"
echo "채점: .venv-streamlaal/bin/python evaluation/ast/score_acl6060.py --tag $STAMP --split $SPLIT"
echo "재실행(실패분만): STAMP=$STAMP bash evaluation/ast/run_acl6060_sweep.sh $SPLIT"
