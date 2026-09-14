#!/bin/bash
# judge01 — 판단형 루프 첫 실행.
#   v0 를 손으로 쓰지 않고 생성한다: Profiler 가 소스 문장 20개에서 언어 특징을 뽑고 Writer 가
#   판단형 프롬프트를 쓴다. 손으로 쓴 v0(v0_coh4.txt)는 영어 부정어 목록과 영어 예시가 박혀
#   있어 다른 소스 언어로 못 옮긴다 — 자동화 주장이 거기서 깨진다.
#   목표는 그리디 오라클(경계별 H 상위 k), 판정은 (문장,k) 짝 H_set 을 문장 단위로 재추출한
#   부트스트랩 CI 하한. min_gap 1 — 짧은 조각은 H_set 이 알아서 벌한다.
set -u
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)" || exit 1
set -a; [ -f ./.env ] && . ./.env; set +a
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=${PY:-.venv/bin/python}   # COMET·langcodes 가 다른 venv 에 있는 기계는 PY=.venv-autoseg/bin/python
RUN=${RUN:-judge01}          # 설정을 바꿔 다시 돌릴 때는 새 이름으로 — 이전 런 산출물·로그를 덮지 않는다
BUDGET=${BUDGET:-35}         # 런 전체 상한. RESUME=1 이면 state.json 에서 잇고 앞선 지출을 뺀다
V0ARGS=${V0:+--prompt $V0}   # V0=<프롬프트 파일> 이면 v0 를 생성하지 않고 그걸로 시작한다
V0ARGS=${V0ARGS:---generate-v0 --v0-candidates 2}
LOG=core/meaning_segmentator/experiment/artifacts/en2x/logs/$RUN.log
mkdir -p "$(dirname "$LOG")"
echo "== $(date '+%F %T') $RUN start" >> $LOG
$PY -u -m core.meaning_segmentator.autoseg.loop_judge \
    --from-run en2x/en-multi/run25 --run-id en2x/en-multi/$RUN \
    $V0ARGS \
    --dev-a 150 --min-gap 1 --min-chunk 2 --max-k 10 \
    --iterations 5 --n-cases 12 --checkpoint-every 2 \
    --provider openai --model gpt-5-mini --budget $BUDGET ${RESUME:+--resume} >> $LOG 2>&1
echo "== $(date '+%F %T') exit=$? DONE" >> $LOG
