#!/bin/bash
# judge32b — 새 골격을 test 560 에서 잰다. judge32 와 같은 벌 태그를 쓰므로 한 줄로 비교된다.
#
# judge32 가 확정한 것: 절차형 골격(combo)이 선언형 v0 대비 test 560 에서 +0.0084 이고, 이득이
# 그 구간이 쓰는 절단 수 순서를 따른다(≤3 +0.0087 … ≤99 −0.0000). 그 위에 `[Order Rules]` 칸을
# 더한 판은 −0.0031 이었다(잡음 0.67σ 안이지만 여섯 구간 중 다섯이 음수).
#
# 그 뒤 골격을 세 가지 바꿨고 이 런이 그 셋을 합쳐서 잰다:
#
#   1. 칸 이름 [Order Rules] → [Order Principles]. [Output Rules] 와 헷갈린다(사람이 실제로
#      헷갈렸다). [Core Principles] 와 짝이 된다.
#   2. 섹션 순서를 Role → Scoring Rules → Core Principles → Order Principles → Output Rules →
#      Examples 로. 골격이 목표·절차·두 칸의 정의를 말하므로 판단 절보다 먼저 읽어야 한다.
#   3. 두 칸의 정의를 골격에 명시(+560자). 종전에는 절차 문장 안에 섞여 있어서 채점 모델이
#      "하나는 한 자리를 묻고 하나는 두 자리를 견준다" 를 그 자리에서 읽지 못했다.
#
# 그리고 사람이 박아 둔 서열 규칙 한 줄을 골격(동결)으로 옮겨 [Order Principles] 를 **비웠다** —
# 그 칸은 Writer 가 v0 을 쓸 때 채우고 루프가 개정한다. 칸 비용이 461자에서 헤더뿐으로 줄었다.
#
# 볼 것: (a) skel2 가 combo(0.5630) 대비 어디인가. 0 근처면 골격 확장이 공짜다 — 그러면 다음
#            런은 이 골격으로 간다. −0.005 아래면 세 변경 중 무엇이 깎는지 갈라야 한다.
#        (b) 구간별 모양이 combo 의 서명(깊은 구간 +, ≤99 0)을 지키는가
#        (c) 포맷 1차 통과율. 골격이 2,867자로 커졌다(combo 2,268자)
#
# combo 와 v0 의 벌은 judge32 가 뽑아 캐시에 있다(--draw-tag judge32t) — 새로 뽑는 것은 skel2
# 3벌뿐이라 $8 쯤이다. 그래서 세 판이 같은 홀드아웃 위에서 한 줄로 비교된다.
#
#   tmux new-session -d -s judge32b -c <저장소> "bash core/meaning_segmentator/tools/autoseg_en2x/judge32b/run.sh"
set -u
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)" || exit 1
set -a; [ -f ./.env ] && . ./.env; set +a
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
A=core/meaning_segmentator/experiment/artifacts/en2x/en-multi
LOG=core/meaning_segmentator/experiment/artifacts/en2x/logs/judge32b.log

.venv-autoseg/bin/python -u -m core.meaning_segmentator.autoseg.loop_judge \
    --from-run en2x/en-multi/run30 --run-id en2x/en-multi/judge32b \
    --score-only --final-split test --final-draws 3 --draw-tag judge32t \
    --prompt "$A/judge27/prompt_v0.txt" \
    --score-baseline "$A/judge27/prompt_v0.txt" \
    --score-prompts "$A/judge27/prompt_v0_combo.txt,$A/judge27/prompt_v0_skel2.txt" \
    --min-gap 1 --min-chunk 2 --max-k 99 --k-samples 1 \
    --workers 720 --score-workers 9 \
    --provider openai --model gpt-5-mini --budget "${BUDGET:-20}" >> "$LOG" 2>&1
rc=$?
echo "== $(date '+%F %T') judge32b exit=$rc" >> "$LOG"
