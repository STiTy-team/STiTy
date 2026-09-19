#!/bin/bash
# judge24 — 귀납 기반을 두 배로 키우고 dev 를 단일로 되돌린다.
#
# judge23 진단: 확인 분할(test-B 300)이 **고르기 편향**은 걸러냈지만(이터 1 −0.0014 탈락)
# 개정 자체의 **과적합**은 못 막았다 — 이터 3 은 두 단계를 다 통과하고 최종 홀드아웃에서
# −0.0047 이었다. 규칙 한 줄이 사례 45개에서 귀납되는데 그 표본이 좁은 것이 원인이다.
# 그래서 확인 관문을 늘리는 대신 **귀납 기반**을 키운다.
#
#   run30    train 500 (사례 100) / dev 500 단일 / test 560.  dev 를 쪼개면 구간 반폭이
#            0.0102 → 0.0147 로 벌어져 효과 크기와 같은 자릿수가 된다. 500 단일이면 0.0112.
#   사례 100 train 500 이면 이터당 100 을 써도 4이터를 버틴다(judge16 은 train 200 에 100 을
#            써서 2이터에 말랐다).
#   finding 2 개, 역할 4 개로 후보 8. 첫 시도(finding 1개, 후보 4)에서는 **네 역할이 거의 같은
#            규칙을 냈다** — single_small·fallback·induce 가 다 "유한동사와 짧은 보어" 를 겨눴고
#            예시 목록까지 겹쳤다. 모두가 같은 한 자리를 보기 때문이다. finding 을 둘로 늘려
#            탐색 폭을 넓힌다. 비용은 이터당 약 \$16 (dev 500 본채점 한 번이 \$2).
#
# 역할에 `induce` 를 새로 넣었다. 다른 역할은 Critic 이 사례를 한 문장으로 줄인 진단만 읽는데,
# induce 는 **압축 전 자료**(어디를 잘랐고 어디를 잘라야 했는지)를 직접 보고 규칙을 귀납한다 —
# 진단은 주지 않는다. 사례 8개는 `induce_picks` 로 고른다: 구간 비중을 사례 배분과 맞추고
# (gap 절대값은 긴 구간에서 크게 나와 그냥 상위를 뽑으면 ≤3 이 밀린다), 프롬프트가 예시로 든
# 문장을 빼고, 모순으로 죽은 사례를 절반으로 제한한다(그것만 보여주면 "모순 회피" 하나만 배운다).
# 나머지 셋은 실측 성적으로 남겼다 — single_small(결속문, judge23 이터3 채택),
# fallback(순서문, 네 번 다 양수), prune(원칙 삭제, v0 위에서 +0.004~+0.011).
#
# --workers 720 --score-workers 8: 후보 하나는 dev 500/batch 6 = 84콜밖에 안 써서 워커 128 을
# 못 채웠다(활용률 66%). 여덟을 한꺼번에 던지면 672콜이고 워커 720 이 그걸 한 라운드에 받는다 —
# `max_connections` 는 `max(16, workers)` 로 자동 동기화된다.
# 이득이 큰 근거: 캐시 적중 후보(분절 API 0콜)가 16초에 끝났다. 후보 하나 281초 중 GPU 채점이
# 16초(6%)뿐이고 나머지가 API 대기다. GPU 는 락으로 직렬화했으니 8개면 GPU 가 128초 쌓이지만
# API 구간은 겹쳐서, 본채점이 37분 → 약 7분이 된다.
# **이터 1 에서 429 를 확인하는 것이 이번 실행의 목적이다** — rate limit 은 조회할 수 없고
# (프로젝트 키로 /v1/organization/* 가 403), 키도 11:42 에 2번이 소진돼 하나뿐이다. 지금까지
# 이 저장소의 최대 기록은 워커 128 이고 그때 OpenAI 429 는 0건이었다.
#
#   tmux new-session -d -s judge24 -c <저장소> "bash core/meaning_segmentator/tools/autoseg_en2x/judge24/run.sh"
set -u
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)" || exit 1
set -a; [ -f ./.env ] && . ./.env; set +a
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
V0=core/meaning_segmentator/experiment/artifacts/en2x/en-multi/judge21/prompt_v0.txt
LOG=core/meaning_segmentator/experiment/artifacts/en2x/logs/judge24.log

.venv-autoseg/bin/python -u -m core.meaning_segmentator.autoseg.loop_judge \
    --from-run en2x/en-multi/run30 --run-id en2x/en-multi/judge24 --resume \
    --prompt "$V0" \
    --candidate-roles single_small,fallback,prune,induce --candidates-cross \
    --candidates-cap 8 --findings-max 2 --full-score-max 8 --induce-cases 8 \
    --adopt-rule lo --adopt-strong 0 --guard-bin 0.01 --no-near-miss \
    --labeled-examples --growth-per-iter 0.15 --n-cases 100 --inversions-max 15 \
    --case-exclude bin --case-alloc loss \
    --min-gap 1 --min-chunk 2 --max-k 99 --k-samples 1 --iterations 4 \
    --workers 720 --score-workers 8 \
    --provider openai --model gpt-5-mini --budget "${BUDGET:-80}" >> "$LOG" 2>&1
echo "== $(date '+%F %T') judge24 exit=$?" >> "$LOG"
