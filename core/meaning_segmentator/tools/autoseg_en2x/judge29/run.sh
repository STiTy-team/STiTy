#!/bin/bash
# judge29 — 깊은 순위를 겨눈 네 기제를 한 자에서 비교한다. 루프 없이 프롬프트만 잰다.
#
# 배경 (judge27·28)
#   순위 깊이 곡선이 프롬프트 내용과 무관했다 — v0 를 통째로 새로 써도 1위 6.83배 / 8위 1.28 /
#   13위 1.06 으로 그대로였다. 그런데 **한 번 잘못 고르는 손해는 깊이와 무관하게 일정하다**:
#   라벨로 재니 d=1 에서 0.1424, d=8 에서 0.1150, d=13 에서 0.1185 다. 좋은 자리가 소진되는 게
#   아니라 나쁜 자리가 압도적으로 많아서, 깊은 순위에서도 계속 피해야 하는 선택이 남는다.
#   손실의 55%가 ≤3 이고 그 구간은 평균 k 8.4·p90 k 13 을 쓴다.
#
#   judge28 에서 `[Scoring Rules]` 에 **절차문** 한 줄(“한 번에 뿌리지 말고 남은 것 중 최선을
#   하나씩 골라라”)을 넣으니 dev 500·3벌에서 Δ +0.0060 [+0.0009, +0.0114] 이었다. 열두 런 만의
#   첫 통과다. 깊이 적중도 예측한 모양으로 움직였다 — 1~3위는 소수점까지 그대로고 4위부터 올라
#   7위 +0.023 이 최대, 9위부터 옅어진다. 즉 **방향은 맞는데 8~13위를 아직 못 덮는다.**
#
# 재는 것 (기준선 = judge27 v0, 전부 dev 500 · 3벌 · 같은 자)
#   proc    이미 잰 절차문. 캐시에 있어 공짜 — 이번 표의 기준점으로 같이 싣는다.
#   screen  **먼저 걸러내고 살아남은 것만 서열.** 22방향 정밀 서열을 22방향 이진 판정 +
#           10방향 서열로 바꾼다. 원칙 8개가 이미 "여기는 자르면 안 된다" 형태라 원래 용도로
#           쓰인다. 남는 풀이 절반이면 8~13위가 "걸러진 풀의 5~8위" 가 되어 깊이가 얕아진다.
#   bands   **다섯 등급으로 해상도 요구를 낮춘다.** "서로 다른 정수 22개를 0~100 에 흩어라"는
#           장부 부담이 분절 판단과 무관하게 주의를 먹고, 뒤쪽은 채움수가 된다. 등급을 먼저
#           정하고 등급 안에서만 구별하면 깊은 순위가 등급에서 파생된다.
#   order   **서열을 출력 규약으로 만든다.** 절차문은 부탁이었고 이건 강제다 — 답을
#           `ORDER: 7 2 11 ... ; <문장>` 으로 시작하게 하고, 점수는 코드가 그 서열에서 만든다
#           (`agents_distill.split_order`/`order_scores`). 중간·끝 항목을 쓰려면 그 시점에
#           실제로 비교해야 하므로 텍스트 순서로 도망갈 자리가 없다. 서열이 1..m 의 순열이
#           아니면 본문 숫자로 되돌아간다 — 모델이 마커 수를 잘못 세는 일이 있다.
#           한 줄 안에 `;` 로 끊는 이유: 배치 규약이 문장당 한 줄을 요구한다(`_BATCH_LINE`).
#           규약 자체를 고치면 캐시에 남은 v0·proc 의 입력 조건이 달라져 비교가 조용히 깨진다.
#
# 넷을 **한 프로세스에서** 잰다(`--score-prompts`). 따로 띄우면 기준선 3벌을 변형마다 다시 뽑고
# GPU 를 프로세스끼리 다툰다. `--draw-tag judge28` 로 judge28 의 벌 캐시를 그대로 물려
# v0·proc 은 공짜가 된다 — 새로 뽑는 것은 screen·bands·order 의 9벌뿐이다.
#
# 후보를 고르는 단계가 없으므로 고르기 편향이 없다. 3벌 양쪽이면 Δ 의 반폭이 0.0055 라
# 하한 > 0 문턱이 +0.0055 다(judge28 실측).
#
# 볼 것: 전체 Δ, ≤3·≤5 구간 Δ, 그리고 끝난 뒤 캐시에서 다시 재는 **깊이 적중 8~13위**.
# order 는 ORDER 줄이 실제로 쓰였는지(순열 통과율)도 같이 봐야 한다 — 안 쓰였으면 효과가 아니다.
#
#   tmux new-session -d -s judge29 -c <저장소> "bash core/meaning_segmentator/tools/autoseg_en2x/judge29/run.sh"
set -u
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)" || exit 1
set -a; [ -f ./.env ] && . ./.env; set +a
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
A=core/meaning_segmentator/experiment/artifacts/en2x/en-multi
LOG=core/meaning_segmentator/experiment/artifacts/en2x/logs/judge29.log

.venv-autoseg/bin/python -u -m core.meaning_segmentator.autoseg.loop_judge \
    --from-run en2x/en-multi/run30 --run-id en2x/en-multi/judge29 \
    --score-only --final-split dev --final-draws 3 --draw-tag judge28 \
    --prompt "$A/judge27/prompt_v0.txt" \
    --score-baseline "$A/judge27/prompt_v0.txt" \
    --score-prompts "$A/judge27/prompt_v0_proc.txt,$A/judge27/prompt_v0_screen.txt,$A/judge27/prompt_v0_bands.txt,$A/judge27/prompt_v0_order.txt" \
    --min-gap 1 --min-chunk 2 --max-k 99 --k-samples 1 \
    --workers 720 --score-workers 5 \
    --provider openai --model gpt-5-mini --budget "${BUDGET:-45}" >> "$LOG" 2>&1
rc=$?
echo "== $(date '+%F %T') judge29 exit=$rc" >> "$LOG"
