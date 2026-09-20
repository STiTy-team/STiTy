#!/bin/bash
# judge38 — 에이전트가 고칠 수 있는 표면이 목적함수와 **격리돼 있었는지** 잰다.
#
# 무엇을 알아냈나
#
#   combo 골격(다섯 등급 + 등급 안 반복 선택)이 +0.0084 를 벌었다. 그 위에서 에이전트가 낸 개정
#   서른 몇 개는 전부 ≤0 이다. judge37 이 목적함수 탓은 아니라고 했다(max→mean 으로 SNR 0.62배).
#   그래서 골격이 요구하는 것과 각 칸이 주는 것을 맞춰 봤다.
#
#   기제는 **완벽하게 돌고 있다.** 캐시된 출력에서 문장의 76%가 다섯 등급을 전부 쓰고, 아래 등급
#   에서도 99.7%가 서로 다른 번호를 받고 등급 폭을 77% 벌려 쓴다.
#
#   그런데 순위 깊이 배수가 1위 6.16 → 8위 1.27 → 13위 1.07 이다. **모델은 근거 없는 순서를
#   만들어 내고 있다** — "모든 자리에 다른 번호를 줘라" 를 지키려고 구별을 발명한다.
#
#   어디서 끊기는지 계산된다. 문장당 자리 19.8개, 짧은 예산에서 채택 8.4개 = 상위 42%.
#   등급 누적은 90–99 9.6% → 70–89 22.6% → 40–69 45.6% 다. **채택 문턱이 40–69 "위험" 등급 안에
#   떨어지고 거기가 1.27배다.** 아래 두 등급(54%)은 애초에 채택되지 않는다.
#
#   그리고 두 칸이 골격과 어긋난다.
#     [Core Principles] C1~C7 은 전부 예/아니오 위험 질문이고 전부 한 방향이다. 등급 **배정**에는
#       맞지만 등급 **안** 서열에는 쓸 수 없다 — "얼마나 나쁜가" 를 말하지 않는다.
#     [Order Principles] O1·O3 은 골격 S7 이 스스로 정한 기본 서열("앞이 홀로 설 수 있는 쪽,
#       둘 다면 늦은 쪽")과 **같은 말이다.** 골격은 먼저 읽히고 못 고친다. 그래서 그 칸을 편집해도
#       실효가 없다 — fallback 추가 세 번(−0.0130/−0.0018/−0.0028)과 O1 교체 세 번
#       (−0.0017/−0.0041/−0.0022)이 사실상 같은 값인 이유다.
#
#   즉 **에이전트가 만질 수 있는 표면이 목적함수와 격리돼 있었다.** Core 쪽 편집은 이미 잘 되는
#   등급 배정을 다듬고, Order 쪽 편집은 골격이 이미 하는 말을 되풀이한다.
#
# 세 변형
#
#   nodup        골격 S7 의 기본 서열 두 문장을 뗀다. 등급 안 서열은 **오직** [Order Principles] 가
#                정한다(매달린 "that default" 참조도 고쳤다). 중복이 편집을 무효화했는지 본다.
#   graded       C1~C7 에 **정도**를 붙인다 — 각 질문에 "무엇이 더 심하고 무엇이 더 가벼운가" 를
#                더하고, 골격의 기본 서열을 "Core Principles 의 우려가 가장 가벼운 쪽" 으로 바꾼다.
#                같은 질문이 등급 배정과 등급 안 서열을 **둘 다** 하게 된다.
#   graded_edit  graded 위에 **이미 실측된 에이전트 편집**을 얹는다 — judge34 iter 2 후보 0 의
#                O_end 삽입이고 v0 위에서 −0.0015 였다. 같은 편집이 여기서 달라지면 격리가 확인된다.
#                이것이 이 런의 핵심 질문이다: **표면을 목적함수에 이으면 에이전트 편집이 처음으로
#                숫자를 움직이는가.**
#
# 읽는 법
#   nodup 이 0 근처면 중복 자체는 비용이 아니었다는 뜻이다(그래도 O 칸을 살리는 값은 있다).
#   graded 가 오르면 등급 안 서열 재료가 빠져 있던 것이 맞다.
#   graded_edit − graded 를 −0.0015(같은 편집이 v0 위에서 낸 값)와 비교한다. 그 차이가 격리의 크기다.
#
# 기준선을 다시 뽑지 않는다 — `--draw-tag judge35d` 로 v0 세 벌을 캐시에서 읽는다
# (0.5636 / 0.5573 / 0.5578, 평균 0.5596, 벌 하나 sd 0.0035). 3벌 대 3벌 하한 > 0 문턱 약 +0.0057.
#
# 주의: 변형들은 judge34 의 v0 에서 파생됐고 그 v0 의 S7 에는 "아래 등급이 보통 가장 많은 자리를
# 담는다" 는 분포 주장이 남아 있다(코드에서는 뺐다). 기준선과 세 변형에 **똑같이** 들어 있어
# Δ 에서는 상쇄된다. 채택하게 되면 그 문장을 뺀 판으로 다시 만든다.
#
# 비용: 변형 3개 × 3벌 = 9벌 × $3.9 = $35. 예산 $55.
#
#   tmux new-session -d -s judge38 -c <저장소> "bash core/meaning_segmentator/tools/autoseg_en2x/judge38/run.sh"
set -u
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)" || exit 1
set -a; [ -f ./.env ] && . ./.env; set +a
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LANGSMITH_TRACING=0
A=core/meaning_segmentator/experiment/artifacts/en2x/en-multi/judge34
LOG=core/meaning_segmentator/experiment/artifacts/en2x/logs/judge38.log

.venv-autoseg/bin/python -u -m core.meaning_segmentator.autoseg.loop_judge \
    --from-run en2x/en-multi/run30 --run-id en2x/en-multi/judge38 \
    --score-only --final-split dev --final-draws 3 --draw-tag judge35d \
    --prompt "$A/prompt_v0.txt" \
    --score-baseline "$A/prompt_v0.txt" \
    --score-prompts "$A/prompt_v0_nodup.txt,$A/prompt_v0_graded.txt,$A/prompt_v0_graded_edit.txt" \
    --min-gap 1 --min-chunk 2 --max-k 99 --k-samples 1 \
    --workers 720 --score-workers 9 \
    --provider openai --model gpt-5-mini --budget "${BUDGET:-55}" >> "$LOG" 2>&1
rc=$?
echo "== $(date '+%F %T') judge38 exit=$rc" >> "$LOG"
