#!/bin/bash
# judge33b — [Order Principles] 칸을 처음 쓰는 루프 런. v0 을 새로 생성한다.
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
# judge33 을 $5.47 에서 멈추고 다시 시작한 것이다. 그 런의 v0 후보 둘이 [Role] 본문에서 골격을
# 대괄호째 참조해("must feed the banding-and-ordering procedure in [Scoring Rules]") 그 문자열이
# 섹션 경계로 잡혔고, 골격 주입이 그 문장 중간부터 다음 경계까지를 갈아 [Role] 뒷부분이 유실됐다.
# 골격 검사는 그것을 못 잡는다 — 문자열이 있으니 "섹션 없음" 이 아니다. 복원이 불가능해(유실된
# 텍스트가 무엇이었는지 알 수 없다) v0 을 새로 생성한다.
#
# 고친 것: `strip_inline_headers` 가 주입 **전에** 본문 중간 헤더의 대괄호를 뗀다(참조 의도는
# 살리고 경계만 없앤다). Writer 지시문에도 그 금지를 명시했다.
#
# 예산 $120 의 근거 (judge27 이 같은 구성으로 $49.99):
#   v0 생성 2후보 $5.5 + 기준선 3벌 $7.5 + 후보 18개와 에이전트 $37 + both-kinds 재호출 $1.5
#   + 프롬프트가 길어진 몫 +5%($2.5) = **$54**. 1차 통과 후보가 1~2개면 2차 3벌 확인에 $5~10 이
#   붙어 $59~64, 채택이 나면 최종 test 3벌 × 2프롬프트에 $16 이 더 붙어 $75~80 이다. 예산 가드는
#   `BudgetExceeded` 로 런을 **죽이므로**(gateway.py) 최악을 덮는 값으로 잡는다.
#
# `--resume` 은 첫 실행을 $2.81 에서 멈추고 예산만 올려 이은 것이다 — v0 후보 파일 둘이 남아
# Writer 를 다시 부르지 않고, 후보 0 의 train 채점은 분절 캐시에서 그대로 읽는다.
#
#   tmux new-session -d -s judge33b -c <저장소> "bash core/meaning_segmentator/tools/autoseg_en2x/judge33b/run.sh"
set -u
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)" || exit 1
set -a; [ -f ./.env ] && . ./.env; set +a
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=core/meaning_segmentator/experiment/artifacts/en2x/logs/judge33b.log

.venv-autoseg/bin/python -u -m core.meaning_segmentator.autoseg.loop_judge \
    --from-run en2x/en-multi/run30 --run-id en2x/en-multi/judge33b \
    --generate-v0 --v0-candidates 2 --resume \
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
    --provider openai --model gpt-5-mini --budget "${BUDGET:-120}" >> "$LOG" 2>&1
rc=$?
echo "== $(date '+%F %T') judge33b exit=$rc" >> "$LOG"
