#!/bin/bash
# judge39 — `graded` 를 홀드아웃 560문장에서 확인한다.
#
# dev 500 에서 잰 값 (judge38): Δ **+0.0076 [+0.0028, +0.0127]**, 이득이 ≤3 구간에 +0.0143.
# 벌별 0.5685 / 0.5657 / 0.5671 로 세 벌 전부가 기준선 최고 벌(0.5636)보다 높았다.
#
# 무엇을 바꾼 프롬프트인가
#   `[Core Principles]` C1~C7 은 전부 예/아니오 위험 질문이었고 전부 한 방향이었다 — 등급 **배정**
#   에는 맞지만 등급 **안** 서열에는 쓸 수 없다("얼마나 나쁜가" 를 말하지 않는다). 캐시된 출력을
#   보면 모델은 아래 등급에서도 99.7%가 서로 다른 번호를 주고 등급 폭을 77% 벌려 쓰는데, 순위
#   깊이 배수는 8위 1.27배다 — **근거 없는 순서를 만들어 내고 있었다.**
#   graded 는 일곱 질문에 각각 "무엇이 더 심하고 무엇이 더 가벼운가" 를 붙이고, 골격의 등급 안
#   기본 서열을 "Core Principles 의 우려가 가장 가벼운 쪽" 으로 바꿨다. 같은 질문이 등급 배정과
#   등급 안 서열을 **둘 다** 하게 된다.
#
# 두 판을 같이 잰다
#   graded        dev 에서 +0.0076 로 잰 그대로. 홀드아웃 재현을 본다.
#   graded_clean  거기서 "아래 등급이 보통 가장 많은 자리를 담는다" 를 뺀 판(−46자). 그 문장은 이
#                 데이터의 분포이므로 배포 프롬프트에 두지 않는다(코드의 골격에서는 이미 뺐다).
#                 **배포할 판이 이것이므로 이것도 홀드아웃에서 재야 한다.**
#
# 비교 기준
#   combo 는 dev +0.0068 → 홀드아웃 560 **+0.0084** 로 재현됐고 이득이 구간의 절단 수 순서를
#   따랐다. graded 가 같은 모양이면 채택한다. test 560 은 한 벌 sd 0.0049 라 3벌 대 3벌 Δ 의
#   sd 가 0.0040 이고 하한 > 0 문턱이 약 +0.0078 이다 — dev 값(+0.0076)이 그 문턱 바로 위다.
#   그러니 **하한이 0 을 살짝 밟아도 구간 모양(≤3 이 가장 크고 ≤99 가 0 근처)이 맞으면 근거로 본다.**
#
# 비용: test 560 한 벌 약 $4.4 × 3프롬프트 × 3벌 = $40. 예산 $60.
#
#   tmux new-session -d -s judge39 -c <저장소> "bash core/meaning_segmentator/tools/autoseg_en2x/judge39/run.sh"
set -u
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)" || exit 1
set -a; [ -f ./.env ] && . ./.env; set +a
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LANGSMITH_TRACING=0
A=core/meaning_segmentator/experiment/artifacts/en2x/en-multi/judge34
LOG=core/meaning_segmentator/experiment/artifacts/en2x/logs/judge39.log

.venv-autoseg/bin/python -u -m core.meaning_segmentator.autoseg.loop_judge \
    --from-run en2x/en-multi/run30 --run-id en2x/en-multi/judge39 \
    --score-only --final-split test --final-draws 3 --draw-tag judge39t \
    --prompt "$A/prompt_v0.txt" \
    --score-baseline "$A/prompt_v0.txt" \
    --score-prompts "$A/prompt_v0_graded.txt,$A/prompt_v0_graded_clean.txt" \
    --min-gap 1 --min-chunk 2 --max-k 99 --k-samples 1 \
    --workers 720 --score-workers 9 \
    --provider openai --model gpt-5-mini --budget "${BUDGET:-60}" >> "$LOG" 2>&1
rc=$?
echo "== $(date '+%F %T') judge39 exit=$rc" >> "$LOG"
