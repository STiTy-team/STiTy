#!/bin/bash
# judge33 — [Order Principles] 칸을 처음 쓰는 루프 런. v0 을 새로 생성한다.
#
# **왜 --generate-v0 인가.** 판단 칸이 둘로 갈렸고(등급을 정하는 단항 절, 등급 안을 가르는 이항 절)
# Writer 가 그 둘을 처음부터 채워야 한다. 옛 v0 들은 칸이 하나뿐인 시절의 글이라 이항 절이 비거나
# 손으로 박은 한 줄로 시작하게 되는데, 사람이 쓴 문장을 실측 없이 고정하는 셈이다.
#
# **골격은 사람이 정하고 코드가 주입한다.** [Scoring Rules] 와 [Output Rules] 를 v0 생성 직후
# 덮어쓴다(`aj.scoring_rules(targets)`). Writer 지시문은 선언형 순위 규칙만 알아서, 맡겨 두면
# test 560 에서 확인한 절차형 골격의 이득(+0.0084)을 그대로 버린다.
#
# 이번에 처음 켜는 것 넷:
#
#   --pe-verbatim        PE 가 쓴 문면을 Critic 의 발견 문면으로 되돌린다. judge31 이터 1·2 에서
#                        PE 가 단항 금지문("Do not cut …")을 서열문("rank … higher")으로 고쳐 썼고,
#                        그 형태가 역할 배분의 근거이므로 말로만 금지하면 새어 나간다.
#   --critic-both-kinds  check·order 를 각각 최소 하나 내게 한다. judge31 이터 1 은 발견이 `check`
#                        하나뿐이어서 이항 편집을 쓰는 역할이 짝을 못 찾고 후보가 9개 계획에서
#                        2개로 줄었다. 사유를 적어 비운 것은 그대로 받는다.
#   replace 역할         원칙 하나를 지우고 그 자리에 발견을 넣는다(순증가 80자 이내). 문장을 더한
#                        스물세 번 중 스물두 번이 −0.006~−0.011 이었는데, 지운 편집(C4)은 −0.0001
#                        이고 CI 가 0 을 정중앙에 뒀다. 손해가 내용이 아니라 길이라면 길이 중립
#                        편집은 0 에서 출발한다. [Order Principles] 단위도 대상이다.
#   prune 제외           삭제 단독을 세 번 시험해 C7 −0.0088 / C8 −0.0092 / C4 −0.0001 로 전부
#                        이득이 없었고, replace 가 삭제를 포함한다. 후보 슬롯을 아낀다.
#
# 후보는 발견 3개(order 1 + check 2)에 역할 3개를 형태로 짝지어 6개다 — fallback×order,
# single_small×check 둘, replace×셋(양쪽 kind 와 짝지어진다).
#
# 볼 것: (a) order 발견이 매 이터 나오는가 (judge31 은 세 번 중 한 번 없었다)
#        (b) replace 후보가 0 근처에서 출발하는가 — 길이 가설의 시험이다
#        (c) [Order Principles] 에 들어간 편집이, judge31 에서 같은 이항 문장을 원칙 칸에 넣었을
#            때(−0.0060 / −0.0077)보다 나은가
#        (d) PE 문면 되돌림이 몇 번 발동하는가. 많으면 PE 가 계속 형태를 바꾸려 한다는 뜻이다
#
# v0 을 새로 생성하므로 이 런의 기준선은 judge31·32 와 직접 비교할 수 없다(원칙 문면이 새 글이다).
# 런 안의 후보 Δ 와 위 (c) 가 비교 가능한 값이다.
#
#   tmux new-session -d -s judge33 -c <저장소> "bash core/meaning_segmentator/tools/autoseg_en2x/judge33/run.sh"
set -u
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)" || exit 1
set -a; [ -f ./.env ] && . ./.env; set +a
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=core/meaning_segmentator/experiment/artifacts/en2x/logs/judge33.log

.venv-autoseg/bin/python -u -m core.meaning_segmentator.autoseg.loop_judge \
    --from-run en2x/en-multi/run30 --run-id en2x/en-multi/judge33 \
    --generate-v0 --v0-candidates 2 \
    --candidate-roles fallback,single_small,replace --candidates-cross \
    --candidates-cap 8 --findings-max 3 --full-score-max 8 \
    --pe-verbatim --critic-both-kinds \
    --screen-n 0 \
    --baseline-draws 3 --confirm-draws 3 --final-draws 3 \
    --gate-rule mean --adopt-rule lo --guard-bin 0.01 --no-near-miss \
    --labeled-examples --growth-per-iter 0.15 --n-cases 100 --inversions-max 15 \
    --case-exclude bin --case-alloc loss \
    --min-gap 1 --min-chunk 2 --max-k 99 --k-samples 1 --iterations "${ITER:-3}" \
    --workers 720 --score-workers 8 \
    --provider openai --model gpt-5-mini --budget "${BUDGET:-70}" >> "$LOG" 2>&1
rc=$?
echo "== $(date '+%F %T') judge33 exit=$rc" >> "$LOG"
