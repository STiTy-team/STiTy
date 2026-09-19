#!/bin/bash
# judge36 — 프롬프트가 **집합 수준 구조**를 말하지 않는 것을 잰다.
#
# 무엇을 발견했나
#
#   `hset.py:150` 이 이렇다:
#
#     worst = max((contra_of(i, j) for j in c), default=0.0)
#     out.append(q * (1 - worst))
#
#   **채택된 절단 집합에서 contra 가 가장 큰 하나가 문장 전체 점수에 곱해진다.** 그런데 프롬프트의
#   `[Scoring Rules]` 는 자리마다의 `target = cohesion x (1 - contra)` 만 말하고 이 구조를 한 번도
#   말하지 않는다. 모델은 자리를 하나씩 채점하라고만 듣는다.
#
#   이것이 밤새 본 패턴을 설명한다 — "개정이 절단을 바꾸지만 방향이 없고, 방향 없는 흔들림은 음수로
#   쏠린다". 아무 재배열이나 하면 contra 높은 자리가 채택 집합에 들어갈 위험이 생기고 그 하나가
#   문장을 곱으로 깎는다. 부검에서 나빠진 짝과 좋아진 짝이 매번 반반인데 합은 음수인 이유다.
#   **기제를 알고 있었으면서 모델에게 말하지 않았다.**
#
#   깊이와 합쳐진다: ≤3 구간은 문장당 절단을 평균 8.42개 채택하고(dev 500 실측, 중앙값 8, 짝 1947,
#   전체의 43%) 손실의 57% 가 거기 있다. 여덟을 고르는데 그중 최악 하나가 문장을 지배한다.
#
# 무엇을 재나
#
#   worstcut  두 오류가 대등하지 않다는 것을 `[Scoring Rules]` 에 넣는다(+687자). 위험한 자리를
#             안전한 자리 위로 올리면 문장 전체가 걸리고, 안전한 자리를 너무 내리면 그 한 선택만
#             손해다. 목표값이 비슷하면 뒷부분에 반박당할 가능성이 낮은 쪽을 택하라.
#   both      worstcut + judge35 의 depth. judge35 에서 depth 는 +0.0017 [−0.0032, +0.0068] 로
#             문턱(+0.0062)에 못 미쳤지만 **이득이 ≤3 구간(+0.0033)에 몰렸다** — combo 와 같은
#             모양이다. 두 기제가 가산적이면(combo 에서 등급 +0.0026 과 절차문 +0.0037 이 합 +0.0063
#             으로 실측 +0.0064 였다) 합쳐서 문턱을 넘을 수 있다.
#
# 기준선을 다시 뽑지 않는다
#   `--draw-tag judge35d` 를 그대로 쓴다. v0 의 세 벌이 그 태그로 이미 캐시에 있어 공짜로 읽히고
#   (`segment.json`, `segment_fin2_judge35d.json`, `segment_fin3_judge35d.json`), 세 프롬프트가
#   **문자 그대로 같은 기준선**에 붙는다. judge35 의 Δ 와 직접 비교된다.
#   v0 3벌 실측: 0.5636 / 0.5573 / 0.5578, 평균 **0.5596**, 벌 하나 sd 0.0035.
#   3벌 대 3벌 Δ 의 sd 가 0.0029 라 하한 > 0 문턱이 약 +0.0057 이다.
#
# 비용: 새로 뽑는 것은 변형 둘 × 3벌 = 6벌 × $3.9 = $23. 예산 $45 로 덮는다.
#
#   tmux new-session -d -s judge36 -c <저장소> "bash core/meaning_segmentator/tools/autoseg_en2x/judge36/run.sh"
set -u
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)" || exit 1
set -a; [ -f ./.env ] && . ./.env; set +a
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LANGSMITH_TRACING=0
A=core/meaning_segmentator/experiment/artifacts/en2x/en-multi/judge34
LOG=core/meaning_segmentator/experiment/artifacts/en2x/logs/judge36.log

.venv-autoseg/bin/python -u -m core.meaning_segmentator.autoseg.loop_judge \
    --from-run en2x/en-multi/run30 --run-id en2x/en-multi/judge36 \
    --score-only --final-split dev --final-draws 3 --draw-tag judge35d \
    --prompt "$A/prompt_v0.txt" \
    --score-baseline "$A/prompt_v0.txt" \
    --score-prompts "$A/prompt_v0_worstcut.txt,$A/prompt_v0_both.txt" \
    --min-gap 1 --min-chunk 2 --max-k 99 --k-samples 1 \
    --workers 720 --score-workers 9 \
    --provider openai --model gpt-5-mini --budget "${BUDGET:-45}" >> "$LOG" 2>&1
rc=$?
echo "== $(date '+%F %T') judge36 exit=$rc" >> "$LOG"
