#!/bin/bash
# judge44 — 길이 상한을 고친 뒤 루프를 같은 v0 위에서 다시 돌린다.
#
# 왜 다시 도나 — 앞 런에서 `replace` 는 **한 번도 채점까지 못 갔다**
#   세 이터 전부 같은 자리에서 죽었다: "순증가 171자 > 80자", "133자 > 105자", "216자 > 101자".
#   `severity` 도 한 이터를 그렇게 잃었다("347자 늘었다 — 200자까지만 된다"). 이터당 후보 다섯 중
#   하나나 둘이 채점 전에 사라졌으므로, 그 런의 "개선 없음" 은 역할에 대한 판정이 아니었다 —
#   후보 절반쯤이 측정되지 않은 채 끝났다.
#   원인은 하나다. 원칙 한 줄이 **질문 + 정도 축** 두 부분이 되면서 300~400자가 됐는데, 상한 셋이
#   그 전 크기(187자)에 맞춰 절대값으로 박혀 있었다. 지금은 전부 대상 크기에 비례한다.
#     replace  순증가 대신 **문장 수** 를 지킨다 — 한 줄 나가고 한 줄 들어오고, 길이는 두 배까지.
#     severity 지우는 줄의 1.2배까지.
#     single_small 60단어 → 80단어(실제 원칙 줄이 40~57단어다).
#
# v0 와 기준선 벌을 앞 런과 **같게** 둔다
#   `--prompt judge42/prompt_v0.txt --draw-tag judge42` 로 기준선 3벌을 그대로 읽는다. 돈이 안 들고
#   (캐시는 `../run30/cache` 를 공유한다) 더 중요하게는 **기준선 추출 운이 두 런에서 동일해진다** —
#   한 벌의 오차는 그 런의 후보 Δ 전부에 같은 방향으로 얹히므로, 벌을 공유하면 judge42 의 Δ 와
#   이 런의 Δ 를 직접 비교할 수 있다. v0 를 새로 뽑으면 Writer 분산(train 폭 0.0208)이 끼어든다.
#
# 역할 다섯 — 후보 6
#   severity ×order / fallback ×order / narrow_rule ×check / replace ×둘 / prune ×1.
#   `prune` 의 대상이 이 판에서는 `[Order Principles]` 세 줄뿐이다. C 일곱은 전부 정도 축을 갖고,
#   정도 축을 든 줄을 지우는 것은 등급 안 서열 재료를 지우는 일이라 코드가 거부한다(그런 줄만 남은
#   판에서는 `usable_roles` 가 역할을 아예 뺀다). O 세 줄은 반대로 지울 값어치가 있다 — 그 칸의
#   계약이 "예외만, 적게, 좁게" 인데 Writer 가 셋을 썼으니 겹치는 것이 있을 수 있다.
#
# 볼 것
#   (a) 후보 다섯이 다섯 다 채점까지 가는가. 그것이 이 런의 1차 판정이다.
#   (b) `replace` 의 Δ — 처음으로 측정된다.
#   (c) `severity` 의 Δ 가 앞 런(+0.0013, 2차 +0.0003)과 같은 방향인가. 기준선 벌이 같으므로
#       그 비교가 이번에는 성립한다.
#   (d) 세 이터가 같은 규칙을 되풀이하는가.
#
# 예산 $100 — 기준선이 캐시라 이터당 $8~10, 3이터에 $30 안팎을 본다. 채택이 나면 최종 test 3벌이
# $12 더 붙는다. 예산 가드는 런을 죽이므로 최악을 덮는 값으로 잡는다. 끊기면 `--resume` 이 이어받고
# 이미 쓴 돈을 차감한다.
#
#   tmux new-session -d -s judge44 -c <저장소> "bash core/meaning_segmentator/tools/autoseg_en2x/judge44/run.sh"
set -u
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)" || exit 1
set -a; [ -f ./.env ] && . ./.env; set +a
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LANGSMITH_TRACING=0
V0=core/meaning_segmentator/experiment/artifacts/en2x/en-multi/judge42/prompt_v0.txt
LOG=core/meaning_segmentator/experiment/artifacts/en2x/logs/judge44.log

.venv-autoseg/bin/python -u -m core.meaning_segmentator.autoseg.loop_judge \
    --from-run en2x/en-multi/run30 --run-id en2x/en-multi/judge44 \
    --prompt "$V0" --draw-tag judge42 --resume \
    --candidate-roles severity,replace,fallback,narrow_rule,prune --candidates-cross \
    --candidates-cap 6 --findings-max 2 --full-score-max 6 \
    --pe-verbatim --critic-both-kinds --screen-n 0 \
    --baseline-draws 3 --confirm-draws 3 --final-draws 3 \
    --gate-rule mean --gate-min 0.003 --adopt-rule lo --guard-bin 0.01 --no-near-miss \
    --labeled-examples --growth-per-iter 0.15 --n-cases 100 --inversions-max 15 \
    --case-exclude bin --case-alloc loss \
    --min-gap 1 --min-chunk 2 --max-k 99 --k-samples 1 --iterations 3 \
    --workers 720 --score-workers 8 \
    --provider openai --model gpt-5-mini --budget "${BUDGET:-100}" >> "$LOG" 2>&1
rc=$?
echo "== $(date '+%F %T') judge44 exit=$rc" >> "$LOG"
