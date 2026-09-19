#!/bin/bash
# judge23 — 채택을 두 단계로 가른다 (고르기 / 확인).
#
# judge21 이 남긴 문제: dev 600 하나로 후보를 12번 재고 최고를 골라 채택했는데 최종 홀드아웃에서
# Δ −0.0089 로 뒤집혔다. 효과가 0 인 개정도 12번 재면 하나쯤 하한이 0 을 넘는다(2.5% × 12 ≈ 30%).
# run29 는 그 dev 600 을 test_a 300(고르기) / test_b 300(확인) 으로 갈랐다. 확인 단계에는
# 고르기가 없으므로 선택 편향이 없다 — 우연 통과가 2.5% × 50% = 1.25% 로 내려간다.
#
# 역할은 실측으로 둘만 남겼다:
#   single_small  결속문. judge22 +0.0057, judge16 채택 둘의 형식, judge17·18 에서 네 번 최선
#   fallback      순서문. judge21·22 네 번 다 양수, 홀드아웃에서 겨눈 구간을 맞힌 유일한 것
#   prune         원칙 삭제. dev 에서 +0.0049/+0.0068 (나머지 두 번은 같은 delete C2 의 캐시
#                 재측정이다). 홀드아웃 단독 검증을 받은 적이 없어 이번에 확인 분할을 통과하는지 본다.
# 뺀 것: narrow_rule(네 번 전부 음수) / examples_only(홀드아웃에서 겨눈 구간까지 뒤집힘) /
#   free(judge17·18 최악) / rewrite(길이 천장 초과).
# finding 을 2개로 올리고 교차곱(--candidates-cross)으로 역할 3 × finding 2 = 후보 6 을 만든다.
# cap 은 min(역할수 × finding수, cap) 이므로 6 이어야 prune 이 잘리지 않는다.
# 교차곱 없이 두면 주기가 겹쳐 같은 (역할, finding) 짝이 두 번 반복된다. judge21 은 finding 1개라
# prune 이 세 이터 연속 같은 편집을 냈다.
#
# PE 문면에 번짐 금지(Containment)를 넣었다. 규칙이 겨냥 밖 문장의 순위를 바꾸면 안 된다.
#
# OPENAI_API_KEY_2 는 2026-09-19 11:42 에 크레딧이 떨어져 429 insufficient_quota 로 런을
# 죽였다. 키 하나로만 돈다 — 라운드로빈에서 죽은 키가 순번을 받으면 재시도 5회를 다 태우고 끝난다.
#
# 2026-09-19 13:15 컨테이너 재시작으로 tmux 데몬째 죽었다 — claude 세션 종료는 tmux 가
# 버티지만 컨테이너 재시작은 못 버틴다. --resume 으로 이터 4 부터 잇는다(이터 3 채택본이
# state.json·best_prompt.txt 에 남아 있다).
#
#   tmux new-session -d -s judge23 -c <저장소> "bash core/meaning_segmentator/tools/autoseg_en2x/judge23/run.sh"
set -u
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)" || exit 1
set -a; [ -f ./.env ] && . ./.env; set +a
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
V0=core/meaning_segmentator/experiment/artifacts/en2x/en-multi/judge21/prompt_v0.txt
LOG=core/meaning_segmentator/experiment/artifacts/en2x/logs/judge23.log

.venv-autoseg/bin/python -u -m core.meaning_segmentator.autoseg.loop_judge \
    --from-run en2x/en-multi/run29 --run-id en2x/en-multi/judge23 --resume \
    --prompt "$V0" \
    --candidate-roles single_small,fallback,prune --candidates-cross \
    --candidates-cap 6 --findings-max 2 --full-score-max 6 \
    --adopt-rule lo --adopt-strong 0 --guard-bin 0.01 --no-near-miss \
    --verify-dev-b --verify-min 0 --checkpoint-every 0 \
    --labeled-examples --growth-per-iter 0.15 --n-cases 45 --inversions-max 15 \
    --case-exclude bin --case-alloc loss \
    --min-gap 1 --min-chunk 2 --max-k 99 --k-samples 1 --iterations 4 \
    --workers 128 \
    --provider openai --model gpt-5-mini --budget "${BUDGET:-60}" >> "$LOG" 2>&1
echo "== $(date '+%F %T') judge23 exit=$?" >> "$LOG"
