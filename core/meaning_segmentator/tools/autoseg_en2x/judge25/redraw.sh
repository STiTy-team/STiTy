#!/bin/bash
# judge25 재추출 확인 — 채택본의 dev 이득이 "분절을 다시 뽑아도" 남는지 본다.
#
# judge25 는 dev 500 에서 Δ +0.0138 [+0.0032, +0.0240] 로 채택됐는데 test 560 에서
# −0.0070 [−0.0180, +0.0036] 이었다. 가를 것은 둘이다:
#
#   (a) dev 고유 피팅   → 새 추출에서도 dev Δ 가 +0.014 근처로 다시 나온다
#   (b) 후보 선택 잡음  → 새 추출에서 0 근처로 주저앉는다
#
# (b) 쪽 근거가 이미 있다. `noise/sample_noise.json` 은 **같은 프롬프트**를 세 번 독립으로
# 분절해 서로 비교한 것인데 Δ 가 +0.0144 / +0.0040 / −0.0105 로 나왔다(200문장, 반폭 0.0178).
# 문장 500 이면 이 흔들림의 sd 가 약 0.0057 이라 +0.0138 은 2.4σ 다. 부트스트랩은 문장만
# 다시 뽑으므로 후보 넷 중 최고를 고르는 데서 오는 편향은 못 덮는다.
#
# **캐시는 런 디렉토리가 아니라 --from-run 쪽에 있다.** 루프가 `<런>/cache` 를
# `../<from-run>/cache` 로 심볼릭 링크하므로 run-id 만 바꿔서는 같은 분절을 그대로 읽는다
# (첫 시도 judge25r2 가 59초·$0.00 에 Δ +0.0138 을 소수점까지 재현했다. 그래서 judge24 의
# v0 test 줄과 judge25 의 것이 완전히 같았던 것이다).
# 그래서 `run30f` 를 만들어 쓴다 — 데이터·라벨은 run30 으로의 링크고 **분절 캐시만 비었다.**
# 번역 캐시는 링크로 살려 둔다(조각 텍스트로 키를 잡으므로 겹치는 만큼 비용이 줄고, 분절이
# 새로 뽑히는 것과는 무관하다).
# --score-only 라 후보를 만들지 않고 두 프롬프트만 잰다 — 선택이 없으니 선택 편향도 없다.
#
#   tmux new-session -d -s judge25r2 -c <저장소> "bash core/meaning_segmentator/tools/autoseg_en2x/judge25/redraw.sh"
set -u
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)" || exit 1
set -a; [ -f ./.env ] && . ./.env; set +a
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
A=core/meaning_segmentator/experiment/artifacts/en2x/en-multi
LOG=core/meaning_segmentator/experiment/artifacts/en2x/logs/judge25r3.log

.venv-autoseg/bin/python -u -m core.meaning_segmentator.autoseg.loop_judge \
    --from-run en2x/en-multi/run30f --run-id en2x/en-multi/judge25r3 \
    --score-only --final-split dev \
    --prompt "$A/judge25/best_prompt.txt" \
    --score-baseline "$A/judge21/prompt_v0.txt" \
    --min-gap 1 --min-chunk 2 --max-k 99 --k-samples 1 \
    --workers 720 --score-workers 2 \
    --provider openai --model gpt-5-mini --budget "${BUDGET:-12}" >> "$LOG" 2>&1
echo "== $(date '+%F %T') judge25r3 exit=$?" >> "$LOG"
