#!/bin/bash
# judge42 — 편집 표면을 옮긴 뒤 처음 끝까지 도는 루프 런.
#
# v0 는 후보 여덟에서 골랐다 (train 0.5352~0.5560, 평균 0.5467, 폭 0.0208)
#   상위 셋이 0.0018 안에 뭉쳐 `V0_TIE_BAND` 가 점수 대신 포맷으로 갈랐고, 후보 3 이 채택됐다 —
#   1차 포맷 통과율 0.788(셋 중 최고), 원칙 일곱 전부 정도 축, 길이 10,433자(여덟 중 최소).
#
#   분포에서 알게 된 것 둘
#     (a) Writer 는 폭이 크다. 최고 0.5560 과 최저 0.5352 가 0.0208 차이다. 다만 여덟 개 **전부**
#         옛 골격의 v0(train 0.5588)보다 낮다 — 분포가 0.003~0.024 아래로 평행 이동했다.
#     (b) 정도 축 누락은 점수의 원인이 아니다. 완전한 다섯이 평균 0.5478, 1~2줄 빠진 셋이 0.5448 로
#         차이 0.0030 이고 한 벌 sd(0.0035) 안이다. 위생 문제지 점수 문제가 아니다.
#         그러니 "길이가 차서 뒤를 줄이니 점수가 나쁘다" 는 연결은 근거가 없었다.
#
# 이 런이 묻는 것
#   **루프가 등급 안 서열을 개선할 수 있는가.** 지금까지 홀드아웃에서 잰 이득 둘(combo +0.0084,
#   정도 축 +0.0066)은 둘 다 사람이 손으로 쓴 것이고, 루프의 편집 서른 몇 개는 전부 ≤0 이다.
#   표면을 이득이 있는 자리로 옮긴 뒤 처음 돌린다.
#
# 역할 넷이 세 일에 갈린다 (이터당 후보 5개)
#   severity    ×order   원칙 하나의 정도 축만. 질문은 첫 물음표까지 바이트 보존.
#   fallback    ×order   [Order Principles] 에 좁은 예외 하나.
#   narrow_rule ×check   [Core Principles] 에 새 질문 + 정도 축.
#   replace     ×둘      원칙 하나를 발견으로 교체. 비교 팔.
#
# 앞선 실행에서 고친 것 — **문면 되돌림이 C 단위를 망치고 있었다**
#   Critic 발견 문면은 질문 + 정도 축 형태가 아니라 평서문·금지문이다. `--pe-verbatim` 이 PE 가 옳게
#   쓴 줄을 그 문면으로 갈아치우면 모양 검사나 금지문 검사가 **반드시** 거부한다. 실측으로 이터
#   하나에서 후보 다섯 중 둘을 그렇게 잃었다. 면제를 역할이 아니라 **칸** 으로 정했다(C 는 되돌리지
#   않고 O 는 그대로). 이 런에서 그 수리가 실전에서 확인된다.
#
# 볼 것
#   (a) `severity` 후보의 Δ — 이 런의 유일한 질문이다.
#   (b) 후보 다섯이 다섯 다 채점까지 가는가(되돌림 수리가 들었는가).
#   (c) 모양 검사가 **정당하게만** 거부하는가. 표지는 실제 생성물로 맞췄다(정도 축 31/31 통과,
#       방향만 말하는 절 23개 오탐 0).
#   (d) 세 이터가 같은 규칙을 되풀이하는가.
#
# 예산 $150 — 앞선 실행 지출($26.69)을 `--resume` 이 차감한다. 채택 없으면 총 $100 안팎,
# 채택이 나면 최종 test 까지 $140 쯤이다. 예산 가드는 런을 죽이므로 최악을 덮는 값으로 잡는다.
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
    --provider openai --model gpt-5-mini --budget "${BUDGET:-150}" >> "$LOG" 2>&1
rc=$?
echo "== $(date '+%F %T') judge42 exit=$rc" >> "$LOG"
