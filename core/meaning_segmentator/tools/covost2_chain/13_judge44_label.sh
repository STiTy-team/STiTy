#!/bin/bash
# CoVoST2 en test 전체(15,530)를 judge44 의 **v0 와 채택본**으로 라벨링한다. 여기서 돈이 든다.
#
# 왜 두 프롬프트인가 — judge44 가 FLEURS 에서 올린 폭(+0.0187)이 **다른 코퍼스에서도 남는지**
# 보려는 것이다. covost2 는 루프가 한 번도 본 적 없고 문장이 짧다(어절 9.1 대 FLEURS 21).
#
# 비용·시간 (같은 코퍼스 실측 `covost2_full_run13_mg1` 에서: 15,430문장 $38.53 / 워커 12 로 4시간)
#   출력 토큰이 비용의 97% 다. 어절당 141토큰(judge44 프롬프트 실측)이면 15,530 × 9.1 × 141
#   = 19.9M → 약 $40. 프롬프트당 $40~60, 둘이면 $80~120.
#   시간은 워커가 정한다. 호출 하나가 약 63초이고 총 2,700호출이므로 워커 720 이면 4웨이브
#   = 5분 안쪽이다. **단 그 농도로 부어 본 적이 없다** — 500동시까지만 검증됐다.
#   그래서 429 를 센다(`gw.rate_limited`, 진행 줄에 찍힌다). 백오프가 붙으면 웨이브가 늘어진다.
#
# 키 둘을 라운드로빈으로 쓴다(`OPENAI_API_KEY`, `OPENAI_API_KEY_2`). 진행 줄과 최종 요약에
# **키별 지출이 이름으로** 찍힌다 — 인덱스만 남으면 나중에 대시보드와 맞춰 볼 수 없다.
#
#   tmux new-session -d -s covost44 -c <저장소> "bash core/meaning_segmentator/tools/covost2_chain/13_judge44_label.sh"
set -u
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)" || exit 1
set -a; [ -f ./.env ] && . ./.env; set +a
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# LangSmith 추적을 끈다 — 켜 두면 자기 쪽 월 한도에 걸려 429 를 쏟아내고(로그가 그 메시지로
# 덮인다) 호출마다 지연이 붙는다. 분절 결과와는 무관하다.
export LANGSMITH_TRACING=0
PY=${PY:-.venv-autoseg/bin/python}
J=core/meaning_segmentator/experiment/artifacts/en2x/en-multi/judge44
F=core/meaning_segmentator/experiment/artifacts/en2x/covost2/full_judge44
MAN=evaluation/ast/manifests/covost2_en-de_full.jsonl
W=${W:-720}
# 키는 환경변수로 고른다. **기본은 하나다** — `OPENAI_API_KEY_2` 가 크레딧이 없어(2026-09-20
# insufficient_quota) 라운드로빈에 넣으면 호출 절반이 5회 재시도 끝에 죽는다. 크레딧을 채운 뒤
# `KEYS=OPENAI_API_KEY_2` 로 다시 넣으면 진행 줄과 최종 요약에 키별 지출이 이름으로 찍힌다.
KEYS=${KEYS:-}
# 본런 예산. v0 실측이 $70.05 였다 — 호출이 예측(2,725)의 1.8배인 4,900회 나왔다.
# `first_pass 0.842` 라 여섯 배치 중 하나는 형식이 깨져 쪼개 다시 부른다. 그 재시도가
# 비용의 40%다. best 프롬프트는 더 길어(11,109자 대 10,459자) 그보다 들 수 있어 넉넉히 둔다.
BUDGET_MAIN=${BUDGET_MAIN:-110}
# 어느 프롬프트를 돌릴지. v0 가 끝났으면 `PROMPTS=best` 로 이어서 돌린다.
PROMPTS=${PROMPTS:-v0 best}
mkdir -p $F/labels $F/logs $F/cache

label () {   # <이름> <프롬프트> <limit인자> <예산>
  $PY -u -m core.meaning_segmentator.tools.covost2_label.label_covost2 \
    --provider openai --model gpt-5-mini ${KEYS:+--extra-key-envs $KEYS} \
    --prompt "$2" --manifest $MAN \
    --out $F/labels/$1.jsonl --cache $F/cache/$1.cache.json \
    --min-gap 1 --t-floor 2 --batch-size 6 --workers $W $3 \
    --max-tokens 24000 --timeout 420 --budget $4 --cache-every 1 \
    > $F/logs/$1.log 2>&1
}

gate () {    # <이름> — 형식이 깨지면 본런을 시작하지 않는다. 기준은 12_full_label.sh 와 같다.
  $PY - "$F/logs/$1.log" <<'PYGATE'
import json, re, sys
t = open(sys.argv[1]).read()
m = re.search(r"\{[^{}]*\"format_pass\".*?\}", t, re.S)
if not m:
    print("  요약 JSON 없음"); sys.exit(1)
d = json.loads(m.group(0))
ok = d["format_pass"] >= 0.90 and d["text_preserved"] >= 0.95
print(f"  format_pass {d['format_pass']} / text_preserved {d['text_preserved']} / "
      f"first_pass {d['first_pass']} / {d['wall_sec']}초 → {'통과' if ok else '불통과'}")
sys.exit(0 if ok else 1)
PYGATE
}

for NAME in $PROMPTS; do
  case $NAME in
    v0)   PROMPT=$J/prompt_v0.txt ;;
    best) PROMPT=$J/best_prompt.txt ;;
    *)    echo "!! 모르는 프롬프트 이름: $NAME"; exit 1 ;;
  esac
  echo "===== $NAME 스모크 30문장 $(date '+%F %T') ====="
  label "${NAME}_smoke" "$PROMPT" "--limit 30" 2.0
  if ! gate "${NAME}_smoke"; then
    echo "!! $NAME 스모크 불통과 — 본런 중단"; exit 1
  fi
  echo "===== $NAME 본런 15,530문장 $(date '+%F %T') ====="
  label "$NAME" "$PROMPT" "" $BUDGET_MAIN
  rc=$?
  echo "== $(date '+%F %T') $NAME exit=$rc"
  grep -E "키별 최종|^\{" $F/logs/$NAME.log | tail -2
  [ $rc -eq 0 ] || exit 1
done
echo "== $(date '+%F %T') ALL DONE" >> $F/logs/chain.log
