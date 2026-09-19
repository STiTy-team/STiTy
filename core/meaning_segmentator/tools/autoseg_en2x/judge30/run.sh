#!/bin/bash
# judge30 — 살아 있는 재료를 합친다. 루프 없이 프롬프트만 잰다.
#
# judge29 결과 (dev 500 · 3벌 · 기준선 v0 0.5636, 관문 반폭 0.0053)
#   proc   (절차문)        Δ +0.0037 [−0.0015, +0.0091]   다섯 구간 전부 양수
#   bands  (등급)          Δ +0.0026 [−0.0027, +0.0081]   다섯 구간 전부 양수
#   screen (실격 먼저)     Δ −0.0043 [−0.0110, +0.0031]   **그런데 순위는 최고** (Spearman +0.309)
#   order  (서열 출력규약) Δ −0.0052 [−0.0116, +0.0013]   포맷 1차 0.55~0.71 로 붕괴, 순위도 나빠짐
#
# screen 의 역설이 이 런의 근거다. 순위를 가장 잘 매기면서 점수가 떨어졌다. 동점 때문이 아니다
# (고유 점수 비율 0.998, 최대 동점 1.04). 원인은 프롬프트 문면이었다 — screen 은 위치의 **65%**
# 를 실격시켜(20 미만 비율 0.650) 살아남는 게 문장당 7.7개인데 **≤3 구간이 쓰는 k 는 8.4** 다.
# 거기에 "실격된 것들끼리의 순서는 중요하지 않다"고 써 놨다. 그런데 라벨 실측은 반대다 —
# **한 번 잘못 고르는 손해가 깊이와 무관하게 일정하다**(d=1 0.1424 / d=8 0.1150 / d=13 0.1185).
# 게다가 H_set 은 조각 중 최악의 모순으로 곱해지므로 집합 평균이 아니라 **최악의 절단 하나가**
# 값을 지배한다. 오라클과의 겹침은 올라가면서 점수가 떨어지는 경로가 그것이다.
#
# 재는 것 (기준선 = judge27 v0, dev 500 · 3벌 · 같은 자. proc·bands 는 캐시라 공짜)
#   combo   등급 + 절차문. 둘을 나란히 붙이면 어긋나므로(등급은 "등급 먼저 그 안에서 구별",
#           절차문은 "전체에서 하나씩") **등급으로 굵게 가르고 각 등급 안에서 하나씩 고르게**
#           맞물린다. 맨 아래 등급도 서열을 매기라고 명시한다 — screen 에서 버릴 부분이 그
#           한 문장뿐이었으므로 여기서는 반대로 못 박는다.
#   combo3  combo + 실격 판정. screen 의 신호(순위 최고)를 살리고 "순서는 아무래도 좋다"만
#           뺐다. 실격된 것은 아래 두 등급으로 보내되 그들끼리도 서열을 갖는다.
#
# 개별 효과가 +0.003 자리이고 문턱이 +0.0053 이라 **두세 개를 합쳐야 넘는 크기**다. 이 런은
# 합이 더해지는지(가산적인지) 아니면 서로 잡아먹는지를 본다 — judge21 의 합본 실험에서는
# 두 편집이 같은 자리를 반대로 당겨 더 나빠진 전례가 있다.
#
#   tmux new-session -d -s judge30 -c <저장소> "bash core/meaning_segmentator/tools/autoseg_en2x/judge30/run.sh"
set -u
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)" || exit 1
set -a; [ -f ./.env ] && . ./.env; set +a
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
A=core/meaning_segmentator/experiment/artifacts/en2x/en-multi
LOG=core/meaning_segmentator/experiment/artifacts/en2x/logs/judge30.log

.venv-autoseg/bin/python -u -m core.meaning_segmentator.autoseg.loop_judge \
    --from-run en2x/en-multi/run30 --run-id en2x/en-multi/judge30 \
    --score-only --final-split dev --final-draws 3 --draw-tag judge28 \
    --prompt "$A/judge27/prompt_v0.txt" \
    --score-baseline "$A/judge27/prompt_v0.txt" \
    --score-prompts "$A/judge27/prompt_v0_proc.txt,$A/judge27/prompt_v0_bands.txt,$A/judge27/prompt_v0_combo.txt,$A/judge27/prompt_v0_combo3.txt" \
    --min-gap 1 --min-chunk 2 --max-k 99 --k-samples 1 \
    --workers 720 --score-workers 5 \
    --provider openai --model gpt-5-mini --budget "${BUDGET:-35}" >> "$LOG" 2>&1
rc=$?
echo "== $(date '+%F %T') judge30 exit=$rc" >> "$LOG"
