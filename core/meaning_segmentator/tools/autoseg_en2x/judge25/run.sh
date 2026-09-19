#!/bin/bash
# judge25 — judge24 에서 갈린 셋을 반영한다. run30 을 그대로 쓴다.
#
# judge24 결과: 최종 test 560 Δ +0.0017 [−0.0097, +0.0125]. 유의하지 않지만 세 런 연속
# 올라왔다(judge21 −0.0089 → judge23 −0.0047 → judge24 +0.0017). ≤3 은 +0.0085 로 여덟 런
# 중 가장 컸다. 부호를 뒤집은 변경은 **train 300 → 500(사례 45 → 100)** 이다 — 확인 관문
# (judge23)은 가짜를 걸러냈지만 홀드아웃을 음수로 남겼다.
#
# 이번에 바꾸는 셋:
#
#   prune 은 남기고 반복만 막는다
#     judge23·24 에서 일곱 번 전부 같은 원칙을 지웠다 — finding 을 갈라 줘도, 채택으로 프롬프트가
#     바뀐 뒤에도 그랬다(기각되면 프롬프트가 안 바뀌고, 같으면 "가장 근거 약한 원칙" 도 같다).
#     그래도 역할을 지우지는 않는다: **규칙을 계속 더하면 짧은 지연과 긴 지연의 이득이 충돌하는
#     지점이 반드시 생기고, 어느 쪽을 버릴지 고를 수 있는 것은 prune 뿐이다.** judge21 이터 1 이
#     그 예다 — C2 를 지워 ≤3 을 +0.0117 벌고 ≤10 을 −0.005 내줬다(부검: "술어–보어를 넓게
#     보호하던 것이 약해졌다").
#     그래서 `enforce_role` 이 **이미 시도한 삭제를 코드로 거부한다.** 보여주기(`sibling_candidates`)
#     만으로는 무시했다. id 로 추적하면 안 된다 — 이터마다 다시 매겨져 `C2` 가 매번 다른 원칙이다.
#     `edit_summary` 가 남기는 본문 앞 80자로 본다. 거부되면 재시도 경로가 사유를 주고 다시 부른다.
#
#   induce 는 사례를 8 → 24건, 조각 번역까지 준다
#     judge24 에서 네 이터 내내 중간 성적이었는데 증거량이 원인일 수 있다 — Critic 은 사례 100개
#     (91K 토큰)를 읽고 induce 는 8개만 봤다. 실측으로 PE 호출 하나가 입력 37,700 토큰이고 그중
#     사례는 2,851 뿐이었다(나머지 34,849 가 시스템·단위·이력·예시). 24건에 조각 번역까지 넣으면
#     17~20K 이고 총 53K — 컨텍스트 400K 에 여유가 크다. 조각 번역이 "왜 그 자리가 나쁜가" 의
#     증거이므로 같이 준다.
#
#   induce 는 사례 상위 절반에서만 뽑는다 (`induce_picks`)
#     gap 은 그 자리에서 오라클 대비 잃은 양이라 하위 절반은 프롬프트가 거의 맞힌 자리다.
#     거기서 규칙을 귀납하면 설명할 차이가 없는데 설명을 만들어야 하고 그 잡음이 굳는다.
#     judge24 네 이터가 전부 그랬다 — 상위/하위 묶음이 +0.0056/+0.0048, −0.0028/−0.0058,
#     −0.0076/−0.0126, −0.0015/−0.0146. 절대 하한(gap ≥ 0.4)은 못 쓴다: 프롬프트가 나아지면
#     사례가 얕아져 이터 4 에서 0.4 이상이 3개뿐이었다.
#
#   finding 2 × 역할 4 = 후보 8
#     역할이 넷이라 finding 3 이면 12 후보·이터당 $24 로 예산을 넘는다. judge24 에서 Critic 이
#     매 이터 둘만 냈으니 실질 손실도 작다. --findings-max 는 상한이다.
#
# 선별은 껐다(--screen-n 0) — 필터는 judge20 에서 폐지했고, 문장을 이터마다 무작위로 흩어
# 뽑아(`screen_indices`) 본채점 배치와 경계가 안 맞으니 캐시도 안 살아난다. 이터당 5분·$1.5 순증이다.
# 워커 720 · --score-workers 8: 84콜 × 8 = 672콜. judge24 가 같은 값에서 OpenAI 429 를 0건으로
# 끝냈다(네 이터 본채점 405~479초).
#
#   tmux new-session -d -s judge25 -c <저장소> "bash core/meaning_segmentator/tools/autoseg_en2x/judge25/run.sh"
set -u
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)" || exit 1
set -a; [ -f ./.env ] && . ./.env; set +a
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
V0=core/meaning_segmentator/experiment/artifacts/en2x/en-multi/judge21/prompt_v0.txt
LOG=core/meaning_segmentator/experiment/artifacts/en2x/logs/judge25.log

.venv-autoseg/bin/python -u -m core.meaning_segmentator.autoseg.loop_judge \
    --from-run en2x/en-multi/run30 --run-id en2x/en-multi/judge25 \
    --prompt "$V0" \
    --candidate-roles fallback,single_small,induce,prune --candidates-cross \
    --candidates-cap 8 --findings-max 2 --full-score-max 8 --induce-cases 24 \
    --screen-n 0 \
    --adopt-rule lo --adopt-strong 0 --guard-bin 0.01 --no-near-miss \
    --labeled-examples --growth-per-iter 0.15 --n-cases 100 --inversions-max 15 \
    --case-exclude bin --case-alloc loss \
    --min-gap 1 --min-chunk 2 --max-k 99 --k-samples 1 --iterations "${ITER:-3}" \
    --workers 720 --score-workers 8 \
    --provider openai --model gpt-5-mini --budget "${BUDGET:-70}" >> "$LOG" 2>&1
echo "== $(date '+%F %T') judge25 exit=$?" >> "$LOG"
