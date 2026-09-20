#!/bin/bash
# judge42 — Writer 가 **계속** 못 쓰는지, 한 번 그런 것인지.
#
# 왜 이것만 보나
#   새 골격·새 지시문으로 Writer 후보 셋을 뽑았더니 구조는 전부 맞았다(정도 축 8/8, Order 는 좁은
#   예외, 느낌의 정도·줄중간 헤더·환경 상수 없음, 골격 검사 통과). **그런데 점수가 낮았다** —
#   train 0.5460 / 0.5459 / 0.5377 이고, 옛 골격에서 뽑은 후보 둘은 0.5588 / 0.5589 였다.
#   세 개가 모두 두 개보다 낮으니 우연으로 보기 어렵지만 표본이 3 대 2 다.
#
#   사람이 손으로 만든 프롬프트로 루프를 돌리는 것은 Writer 를 빼는 것이라 뜻이 없다. 그러니
#   **Writer 를 더 돌려 분포를 본다.** 여덟 개를 놓고 최고값과 폭을 보면 "계속 못 쓰는가" 에 답이 된다.
#
# 구성
#   `--run-id judge41` 을 그대로 쓴다 — 기존 후보 셋의 **파일과 분절 캐시가 재사용**되므로 그 셋의
#   재채점은 공짜다. 다섯 개를 새로 뽑아 여덟 개로 만든다.
#   `--v0-candidates 8` 이 그것이고, 예산이 v0 선택 직후 기준선에서 죽게 잡혀 있다 — 보려는 것만
#   사고 그 이상은 못 쓰게 하는 구성이다(judge41 의 앞선 실행에서 $22.27 을 이미 썼고 `--resume` 이
#   그것을 차감한다).
#
# 읽는 법
#   여덟 개 중 최고가 0.556 근처면 Writer 는 쓸 수 있고 **폭이 큰 것**이다 — `--v0-candidates` 를
#     올려 고르면 된다(`V0_TIE_BAND` 가 동점대에서 포맷으로 가른다).
#   여덟 개가 전부 0.546 아래면 **새 지시문이 Writer 를 나쁘게 만든다.** 그러면 무엇이 원인인지
#     좁혀야 한다 — 길이(정도 축을 여덟 줄에 붙이면 1,000~1,800자 늘어난다), 원칙 수(8 대 7),
#     아니면 정도 축 자체.
#
#   구조는 이미 여덟 개 전부 볼 것이다 — 정도 축 개수, Order 의 폭, 길이, 1차 포맷 통과율.
#
#   tmux new-session -d -s judge42 -c <저장소> "bash core/meaning_segmentator/tools/autoseg_en2x/judge42/run.sh"
set -u
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)" || exit 1
set -a; [ -f ./.env ] && . ./.env; set +a
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LANGSMITH_TRACING=0
LOG=core/meaning_segmentator/experiment/artifacts/en2x/logs/judge42.log

.venv-autoseg/bin/python -u -m core.meaning_segmentator.autoseg.loop_judge \
    --from-run en2x/en-multi/run30 --run-id en2x/en-multi/judge42 \
    --generate-v0 --v0-candidates 8 --resume \
    --candidate-roles severity,replace,fallback,narrow_rule --candidates-cross \
    --candidates-cap 6 --findings-max 2 --full-score-max 6 \
    --pe-verbatim --critic-both-kinds --screen-n 0 \
    --baseline-draws 3 --confirm-draws 3 --final-draws 3 \
    --gate-rule mean --gate-min 0.003 --adopt-rule lo --guard-bin 0.01 --no-near-miss \
    --labeled-examples --growth-per-iter 0.15 --n-cases 100 --inversions-max 15 \
    --case-exclude bin --case-alloc loss \
    --min-gap 1 --min-chunk 2 --max-k 99 --k-samples 1 --iterations 3 \
    --workers 720 --score-workers 8 \
    --provider openai --model gpt-5-mini --budget "${BUDGET:-20}" >> "$LOG" 2>&1
rc=$?
echo "== $(date '+%F %T') judge42 exit=$rc" >> "$LOG"
