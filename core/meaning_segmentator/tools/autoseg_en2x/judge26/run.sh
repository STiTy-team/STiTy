#!/bin/bash
# judge26 — judge25 에서 **딱 하나만** 바꾼다: 채택 전 재추출 확인을 실제로 켠다.
#
# judge25 결과: dev 500 에서 Δ +0.0138 [+0.0032, +0.0240] 로 채택 → test 560 에서 −0.0070
# [−0.0180, +0.0036]. 아홉 런째 뒤집힘이다.
#
# 원인을 사후에 가렸다(`judge25/redraw.sh`, $4.45). **같은 dev 500 문장·같은 두 프롬프트를
# 분절만 새로 뽑아** 다시 쟀다:
#
#   추출 1 (판정에 쓴 것)  v0 0.5514 / 채택본 0.5652 → Δ +0.0138 [+0.0032, +0.0240]  통과
#   추출 2 (새로 뽑음)     v0 0.5578 / 채택본 0.5615 → Δ +0.0037 [−0.0082, +0.0152]  탈락
#
# 문장이 같으므로 "dev 분포에 피팅됐다" 는 배제된다. 사라진 0.0101 중 **0.0064 가 기준선 v0 쪽**
# 이다 — 기준선 한 벌은 후보 전부가 공유하므로 낮게 뽑히면 후보가 다 같이 올라간다. judge25
# 이터 2 에서 서로 다른 편집 넷 중 **셋이 동시에** 하한 > 0 이었던 게 그 자국이다.
#
# 부트스트랩이 틀린 게 아니다. 두 추출 Δ 의 차 0.0101 은 각 Δ 의 sd 0.0058 로 1.35σ 다.
# 문제는 쫓는 효과(≈+0.005)가 반폭(±0.011)과 같은 크기인데 후보를 16번 들여다보고 최고의
# 하한만 본다는 것이다.
#
# 고친 것:
#
#   --adopt-strong 0.05  (judge20~25 는 전부 0)
#     `decide()` 는 하한이 0 과 이 값 사이면 'confirm' 을 내고, 그때 **양쪽을 새 분절로 다시
#     재서** 거기서도 하한 > 0 이어야 채택한다. 이 경로는 코드에 내내 있었는데 문턱이 0 이라
#     `lo > strong` 이 곧바로 accept 로 빠져 **여섯 런 동안 한 번도 안 탔다.** 0.05 는 실측
#     분포상 도달 불가능한 값이라 모든 채택이 확인을 거친다.
#
#   확인이 기준선도 다시 뽑는다 (loop_judge.py)
#     종전 구현은 후보만 새로 뽑고 기준선은 원래 추출을 그대로 썼다. 위 분해대로 오차의 더
#     큰 몫이 기준선 쪽이라 반쪽짜리였다. 이제 둘을 2워커로 동시에 새로 뽑는다.
#     확인 캐시 파일 이름에 런 이름이 들어간다 — `cache/` 가 from-run 으로의 심볼릭 링크라
#     이터 번호만으로는 다음 런이 같은 파일을 읽어 "새 추출" 이 아니게 된다.
#
# 비용은 채택될 뻔한 때만 늘어난다(확인 1회 ≈ $4.4, 5분). judge25 라면 한 번이었다.
# 나머지는 judge25 와 같다 — v0·분할(run30)·역할·사례 수·선별 끔·워커까지 통제한다.
#
#   tmux new-session -d -s judge26 -c <저장소> "bash core/meaning_segmentator/tools/autoseg_en2x/judge26/run.sh"
set -u
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)" || exit 1
set -a; [ -f ./.env ] && . ./.env; set +a
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
V0=core/meaning_segmentator/experiment/artifacts/en2x/en-multi/judge21/prompt_v0.txt
LOG=core/meaning_segmentator/experiment/artifacts/en2x/logs/judge26.log

.venv-autoseg/bin/python -u -m core.meaning_segmentator.autoseg.loop_judge \
    --from-run en2x/en-multi/run30 --run-id en2x/en-multi/judge26 \
    --prompt "$V0" \
    --candidate-roles fallback,single_small,induce,prune --candidates-cross \
    --candidates-cap 8 --findings-max 2 --full-score-max 8 --induce-cases 24 \
    --screen-n 0 \
    --adopt-rule lo --adopt-strong 0.05 --guard-bin 0.01 --no-near-miss \
    --labeled-examples --growth-per-iter 0.15 --n-cases 100 --inversions-max 15 \
    --case-exclude bin --case-alloc loss \
    --min-gap 1 --min-chunk 2 --max-k 99 --k-samples 1 --iterations "${ITER:-3}" \
    --workers 720 --score-workers 8 \
    --provider openai --model gpt-5-mini --budget "${BUDGET:-70}" >> "$LOG" 2>&1
echo "== $(date '+%F %T') judge26 exit=$?" >> "$LOG"
