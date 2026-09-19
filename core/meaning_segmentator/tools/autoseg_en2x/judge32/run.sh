#!/bin/bash
# judge32 — 골격을 **test 560 에서** 확정한다. 루프는 돌지 않는다(--score-only).
#
# 세 프롬프트를 같은 홀드아웃에서 3벌씩 잰다:
#
#   prompt_v0.txt              judge27 이 생성한 v0. 선언형 [Scoring Rules] — 열한 런의 기준선
#   prompt_v0_combo.txt        + 다섯 등급과 등급 내 반복 선택 (dev 500 에서 +0.0068)
#   prompt_v0_combo_ordersec.txt  + [Order Rules] 칸과, 골격이 그 칸을 2단계에서 읽게 한 문장
#
# **왜 test 인가.** combo 의 +0.0068 은 dev 500 값이다. judge25 에서 dev 이득(+0.0138)이 test 에서
# −0.0070 으로 사라진 일을 겪었고, 그 원인이 기준선 한 벌의 편향이었다. 이번에는 양쪽 다 3벌이라
# 그 편향은 없지만, 홀드아웃에서 확인하지 않은 골격을 앞으로 쓸 고정 골격으로 삼을 수는 없다.
#
# **[Order Rules] 를 여기서 재는 이유.** 칸을 하나 늘리고 골격 문장을 고친 것 자체가 점수를 깎을
# 수 있다. 깎으면(combo 대비 −0.005 이상) 루프를 돌릴 의미가 없다 — 루프가 그 칸에 문장을 넣는
# 것이 다음 런의 개입인데, 칸을 만든 값이 이미 마이너스면 개입의 이득과 섞여 못 가린다.
#
# 볼 것: (a) combo 대 v0 의 test Δ 가 dev 의 +0.0068 과 같은 방향·크기인가
#        (b) ordersec 대 combo 가 0 근처인가 (칸을 만든 값)
#        (c) 구간별 이득이 그 구간이 쓰는 k 순서를 따르는가 — dev 에서는 ≤3 +0.0090 → ≤99 −0.0003
#        (d) 포맷 1차 통과율. combo 는 dev 에서 0.634 까지 떨어진 벌이 있었고 ordersec 은 더 길다
#
# 벌 태그를 새로 준다(--draw-tag judge32t) — judge28 태그는 dev 문장만 들어 있어 섞으면 나중에
# 어느 분할의 벌인지 못 가린다.
#
#   tmux new-session -d -s judge32 -c <저장소> "bash core/meaning_segmentator/tools/autoseg_en2x/judge32/run.sh"
set -u
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)" || exit 1
set -a; [ -f ./.env ] && . ./.env; set +a
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
A=core/meaning_segmentator/experiment/artifacts/en2x/en-multi
LOG=core/meaning_segmentator/experiment/artifacts/en2x/logs/judge32.log

.venv-autoseg/bin/python -u -m core.meaning_segmentator.autoseg.loop_judge \
    --from-run en2x/en-multi/run30 --run-id en2x/en-multi/judge32 \
    --score-only --final-split test --final-draws 3 --draw-tag judge32t \
    --prompt "$A/judge27/prompt_v0.txt" \
    --score-baseline "$A/judge27/prompt_v0.txt" \
    --score-prompts "$A/judge27/prompt_v0_combo.txt,$A/judge27/prompt_v0_combo_ordersec.txt" \
    --min-gap 1 --min-chunk 2 --max-k 99 --k-samples 1 \
    --workers 720 --score-workers 9 \
    --provider openai --model gpt-5-mini --budget "${BUDGET:-30}" >> "$LOG" 2>&1
rc=$?
echo "== $(date '+%F %T') judge32 exit=$rc" >> "$LOG"
