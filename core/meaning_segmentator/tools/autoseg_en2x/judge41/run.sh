#!/bin/bash
# judge41 — **Writer 가 지금 골격을 쓸 수 있는지만** 본다. 채점은 하지 않는다.
#
# 왜 이것만 보나
#   지금까지 홀드아웃에서 잰 이득 둘(combo +0.0084, graded +0.0066)은 **둘 다 사람이 손으로 쓴
#   것**이다. combo 는 Writer 출력을 한 글자도 안 바꾸고 `[Scoring Rules]` 만 다시 쓴 것이고,
#   graded 는 Writer 가 쓴 질문 일곱을 그대로 두고 그 뒤 정도절을 내가 붙인 것이다. 루프의 편집
#   서른 몇 개는 전부 ≤0 이다.
#
#   그래서 다음 물음이 "Writer 가 그 구조를 스스로 쓸 수 있는가" 다. 지시문에 넣어 두었지만
#   (`Each line has two parts`, Order 는 예외 전용, 일반 선호 금지) **한 번도 생성해 본 적이 없다.**
#   못 쓰면 정도절은 사람이 계속 박아야 하는 것이고, 루프의 출발점 자체가 사람 손을 탄다.
#
# 예산 $2 가 안전장치다
#   Writer 호출은 후보당 약 $0.01 이고 프로파일러가 조금 더 쓴다. 후보 파일은 **채점 전에** 써지므로
#   train 채점(한 벌 약 $3)에 들어가는 순간 `BudgetExceeded` 로 죽고, 그때 후보 파일은 이미 남아
#   있다. 보고 싶은 것만 사고 그 이상은 못 쓰게 하는 구성이다.
#
# 무엇을 볼 것인가 (생성된 후보 파일에서)
#   1. `[Core Principles]` 의 줄마다 질문(물음표) + 정도 축(심한 쪽·가벼운 쪽 둘 다)이 있는가.
#      `ungraded_principles()` 가 센다. 하나라도 비면 그 원칙은 등급 배정만 하고 등급 안 서열에는
#      기여하지 못한다.
#   2. `[Order Principles]` 가 **좁은 예외**인가, 아니면 일반 선호("prefer the later marker")인가.
#      후자는 새 지시문이 어느 칸에도 금지한 것이고, 종전 Writer 는 O1·O3 으로 정확히 그것을 썼다.
#   3. 정도가 **소스 표면의 구성**으로 적혔는가, 아니면 느낌의 정도("somewhat bad")인가.
#   4. 줄 중간 대괄호 헤더, 환경 의존 상수(이 평가 환경에서만 참인 수치), 골격 검사.
#
#   tmux new-session -d -s judge41 -c <저장소> "bash core/meaning_segmentator/tools/autoseg_en2x/judge41/run.sh"
set -u
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)" || exit 1
set -a; [ -f ./.env ] && . ./.env; set +a
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LANGSMITH_TRACING=0
LOG=core/meaning_segmentator/experiment/artifacts/en2x/logs/judge41.log

.venv-autoseg/bin/python -u -m core.meaning_segmentator.autoseg.loop_judge \
    --from-run en2x/en-multi/run30 --run-id en2x/en-multi/judge41 \
    --generate-v0 --v0-candidates 3 --resume \
    --candidate-roles severity,single_small --candidates-cross \
    --candidates-cap 4 --findings-max 2 --full-score-max 4 \
    --pe-verbatim --critic-both-kinds --screen-n 0 \
    --baseline-draws 3 --confirm-draws 3 --final-draws 3 \
    --gate-rule mean --gate-min 0.003 --adopt-rule lo --guard-bin 0.01 --no-near-miss \
    --labeled-examples --growth-per-iter 0.15 --n-cases 100 --inversions-max 15 \
    --case-exclude bin --case-alloc loss \
    --min-gap 1 --min-chunk 2 --max-k 99 --k-samples 1 --iterations 3 \
    --workers 720 --score-workers 8 \
    --provider openai --model gpt-5-mini --budget "${BUDGET:-2}" >> "$LOG" 2>&1
rc=$?
echo "== $(date '+%F %T') judge41 exit=$rc" >> "$LOG"
