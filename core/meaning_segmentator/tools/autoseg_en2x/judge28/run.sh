#!/bin/bash
# judge28 — 채점 "절차" 를 주면 깊은 순위가 움직이나. 루프를 돌지 않고 두 프롬프트만 잰다.
#
# judge27 에서 나온 진단: 순위 깊이 곡선이 **프롬프트 내용과 무관하다.** v0 를 통째로 새로
# 써도 1위 6.83배 / 8위 1.28 / 13위 1.06 으로 옛 v0(6.73 / 1.32 / 1.08)와 같다. 손실의 55%가
# ≤3 이고 그 구간이 쓰는 깊이가 바로 그 무작위 구간이다. 후보 편집들의 부검 깊이 변화도
# 8위·13위에서 ±0.01 이었다 — 규칙 문장을 더하는 개입은 깊은 쪽을 못 건드린다.
#
# 원인 가설: `[Scoring Rules]` 가 **선언형**이다. 두 v0 모두
#
#     "The integer you write is a RANK ... order the positions by the product (target)
#      and assign distinct integers spread across the full 0-100 range"
#
# 라고만 쓴다. 결과가 순위여야 한다고 말하고 **순위를 만드는 절차는 주지 않는다.** 마커 20개에
# 숫자를 한 번에 뿌려도 지시 위반이 아니므로 비교가 한 번도 강제되지 않는다. 그리고 루프의
# 역할들은 `[Core Principles]` 항목만 넣고 빼므로 이 문장은 열한 런 동안 한 번도 수정된 적이 없다.
#
# 시험: v0 에 **절차문 한 줄만** 더한다(그 외 글자 하나 안 건드렸다, +481자).
#
#     "Build the ranking by repeated selection, not in one pass. ... among the markers you have
#      not numbered yet, choose the ONE position that is the best cut at that moment ... Remove
#      that position from consideration and choose again among the rest ... Each step is one
#      comparison among the positions that remain."
#
# 양쪽을 dev 500 에서 **3벌씩** 잰다(--final-draws 3). 판정 문턱은 judge27 2차와 같은 반폭
# 0.0055 다. --score-only 라 후보를 만들지 않으니 고르기 편향이 없다.
#
# **볼 것은 전체 Δ 가 아니라 깊이 곡선이다.** 끝나면 캐시에서 깊이를 다시 잰다(API·GPU 0):
#
#   PYTHONPATH=. .venv-autoseg/bin/python -m core.meaning_segmentator.tools.autoseg_en2x.diag.rank_depth \
#     --run <run30> --cache-dir <run30>/cache --split dev --tag rank_depth_proc \
#     --prompt <judge27>/prompt_v0_proc.txt \
#     --caches segment.json,segment_fin2_judge28.json,segment_fin3_judge28.json
#
# 8위·13위 배수가 오르면 방향이 맞다. 안 움직이면 글로 주는 절차로는 안 되고 호출 구조(쌍
# 비교·토너먼트)를 바꿔야 한다는 뜻이다.
#
# 위험: 비교를 스무 번 하려면 추론 토큰이 늘어 포맷 1차 통과율(judge27 은 0.74~0.82)이 떨어질 수
# 있다. 떨어지면 재정렬이 늘어 비용이 조금 오른다 — 로그의 `fmt` 를 같이 본다.
#
#   tmux new-session -d -s judge28 -c <저장소> "bash core/meaning_segmentator/tools/autoseg_en2x/judge28/run.sh"
set -u
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)" || exit 1
set -a; [ -f ./.env ] && . ./.env; set +a
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
A=core/meaning_segmentator/experiment/artifacts/en2x/en-multi
LOG=core/meaning_segmentator/experiment/artifacts/en2x/logs/judge28.log

.venv-autoseg/bin/python -u -m core.meaning_segmentator.autoseg.loop_judge \
    --from-run en2x/en-multi/run30 --run-id en2x/en-multi/judge28 \
    --score-only --final-split dev --final-draws 3 \
    --prompt "$A/judge27/prompt_v0_proc.txt" \
    --score-baseline "$A/judge27/prompt_v0.txt" \
    --min-gap 1 --min-chunk 2 --max-k 99 --k-samples 1 \
    --workers 720 --score-workers 2 \
    --provider openai --model gpt-5-mini --budget "${BUDGET:-30}" >> "$LOG" 2>&1
rc=$?
echo "== $(date '+%F %T') judge28 exit=$rc" >> "$LOG"
