#!/bin/bash
# judge35 — 프롬프트가 모델에게 **"몇 개가 채택되는지 생각하지 말라"** 고 말하고 있던 것을 잰다.
#
# 무엇을 발견했나
#
#   `[Output Rules]` 에 이 문장이 있다 (`agents_distill.output_rules` 85~86줄, 사람이 쓴 골격):
#
#     "A deterministic step guarantees a minimum distance between kept cuts, so you never need to
#      reason about spacing or about how many to keep."
#
#   앞 절(간격)은 맞다 — 최소 간격은 코드가 보장한다. **뒤 절이 틀렸다.** 손실의 57% 가 ≤3 구간이고
#   그 구간은 문장당 절단을 약 8.4개 채택한다. 몇 개가 쓰이는지 모르면 4~10위를 공들여 가를 이유가
#   없고, `rank_depth` 가 정확히 그 모양이다: 1위는 무작위 대비 6.16배인데 8위 1.27배, 13위 1.07배다.
#   **순위가 깊이에서 무너지는 것을 프롬프트가 허락하고 있었다.**
#
# 왜 루프가 아니라 직접 측정인가
#
#   judge27·31·33b·34 에서 단일 편집 후보를 서른 번쯤 쟀고 전부 ≤0 이다. judge34 iter 2 는 같은
#   finding 을 판단 줄과 절차 줄에 각각 넣어 비교했는데 절차 줄이 +0.0091 나았지만 여전히 −0.0028 다.
#   유일하게 양수였던 것은 combo — **없던 기제를 새로 넣은** 사람의 개정이고, dev +0.0068 /
#   홀드아웃 560문장 +0.0084 였다. 이득이 구간의 절단 수 순서를 따랐다(≤3 +0.0087, ≤99 −0.0000).
#
#   가설: 레버는 "절차를 고친다" 가 아니라 **"빠진 기제를 넣는다"** 다. 위 문장은 빠진 것보다
#   나쁘다 — 틀린 것을 적어 두었다. 그러면 고치는 값이 combo 급일 수 있고, 그건 루프가 찾을 수 있는
#   종류가 아니다(`[Output Rules]` 는 `FROZEN` 이고 그래야 한다 — 출력 규약이다).
#
# 두 변형으로 원인을 가른다
#
#   nodepthclaim  틀린 절만 뗀다(−26자). 새 주장이 없다. **지시가 해로웠는가** 만 본다.
#   depth         뗀 자리에 얼마나 깊이 읽히는지를 넣는다(+322자): "짧은 예산에서 문장당 약 여덟
#                 자리가 채택되므로 여덟째로 고른 자리가 첫째만큼 자주 읽힌다. 깊은 쪽도 위쪽과
#                 같은 정성으로 정하라 — 운에 맡긴 자리는 그대로 방송된다."
#
#   둘을 같이 재야 읽힌다. A 만 올라가면 문장이 해로웠던 것이고, B 가 A 보다 더 올라가면 깊이
#   정보 자체가 값을 하는 것이다. A 가 0 이고 B 만 올라가도 같은 결론이다.
#
# 분할과 벌 수
#   dev 500 · 3벌씩. Δ 의 sd 가 0.0032 라 하한 > 0 문턱이 +0.0062 이고, combo 급(+0.0068)이면 잡힌다.
#   기준선은 judge34 의 v0 그대로다(dev 2벌 실측 0.5639, 벌 간 폭 0.0007). 여기서는 같은 draw-tag 로
#   3벌을 새로 뽑아 세 프롬프트가 같은 조건에서 붙는다.
#   양수가 나오면 test 560 으로 확인한다 — 그때가 judge36 이다.
#
# 비용: dev 500 한 벌 $3.9 × 3프롬프트 × 3벌 = $35. 예산 $55 로 덮는다.
#
#   tmux new-session -d -s judge35 -c <저장소> "bash core/meaning_segmentator/tools/autoseg_en2x/judge35/run.sh"
set -u
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)" || exit 1
set -a; [ -f ./.env ] && . ./.env; set +a
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LANGSMITH_TRACING=0
A=core/meaning_segmentator/experiment/artifacts/en2x/en-multi/judge34
LOG=core/meaning_segmentator/experiment/artifacts/en2x/logs/judge35.log

.venv-autoseg/bin/python -u -m core.meaning_segmentator.autoseg.loop_judge \
    --from-run en2x/en-multi/run30 --run-id en2x/en-multi/judge35 \
    --score-only --final-split dev --final-draws 3 --draw-tag judge35d \
    --prompt "$A/prompt_v0.txt" \
    --score-baseline "$A/prompt_v0.txt" \
    --score-prompts "$A/prompt_v0_nodepthclaim.txt,$A/prompt_v0_depth.txt" \
    --min-gap 1 --min-chunk 2 --max-k 99 --k-samples 1 \
    --workers 720 --score-workers 9 \
    --provider openai --model gpt-5-mini --budget "${BUDGET:-55}" >> "$LOG" 2>&1
rc=$?
echo "== $(date '+%F %T') judge35 exit=$rc" >> "$LOG"
