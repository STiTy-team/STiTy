#!/usr/bin/env bash
# en→X autoseg 프롬프트 루프 — en-multi/run14.
#
# run13(experiment/README.md "공통 설정")과 다른 것만 적는다. 나머지는 그대로다.
#
#   --train 30 --train-pool 90 --dev 215 --test 100
#       매니페스트가 405문장이라 기본 풀(3 × --train)로는 안 들어간다 — split_data 가
#       문장 부족을 만나면 **홀드아웃을 조용히 포기**하므로 반드시 명시해서 맞춘다.
#       test 100 은 run13 과 같은 문장이고(같은 seed·같은 총량), dev 215 는 run13 dev 265
#       의 앞 215 개다. 배치 30 + 선별 홀드아웃 60.
#   --iterations 6 --patience 6
#       조기 종료 없이 6회 다 돈다 — 이 런의 목적은 거부 부검·삭제 후보·점수 감사가
#       실제 로그에서 어떻게 도는지 보는 것이라 이터를 잃으면 안 된다. 상한은 --budget 이 맡는다.
#   --v0-candidates 5, --revision-candidates 3 (free/add/remove)
#       기본값이 바뀐 것이라 인자로 안 준다. config.json 에 남는다.
#
# 예산: run13 $10.72 (5이터). v0 후보 +4개 ≈ $0.6, 홀드아웃 60 으로 선별 분절 ×2 ≈ +$0.6/이터,
#       이터 6회 → 약 $16~18. 상한 25.
# 시간: 이터당 ~35분 추정(GB10 기준 환산, 셔플 off·consistency 삭제·배치64/fp16 반영) → 4시간 안팎.
#       실측은 iter_00/timing.json 으로 확인할 것.
#
# GPU 가 비어야 시작한다 — 다른 ASR 실험이 24GB 를 잡고 있으면 OOM 으로 죽는다.
# 재개는 자동: history.json 이 있으면 --resume. --fresh 는 쓰지 않는다 (캐시가 날아간다).
set -u
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
cd "$REPO" || exit 1
set -a; . ./.env; set +a
export PYTHONPATH=.
PY=.venv-autoseg/bin/python

RUNS=core/meaning_segmentator/experiment/artifacts
PAIR=en2x/en-multi
RUN=run14
rundir=$RUNS/$PAIR/$RUN
log=$RUNS/en2x/logs/$RUN.log
llog=$RUNS/en2x/logs/$RUN.launch.log
mkdir -p "$RUNS/en2x/logs"

until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 4000 ]; do
  echo "[$(date '+%F %T')] GPU busy ($(nvidia-smi --query-gpu=memory.used --format=csv,noheader)) — waiting" >> "$llog"
  sleep 60
done

extra=()
if [ -f "$rundir/history.json" ]; then
  extra+=(--resume)
  if [ ! -f "$rundir/next_prompt.txt" ] \
     && [ ! -f "$rundir/iter_$(printf '%02d' "$(jq length "$rundir/history.json")")/prompt.txt" ] \
     && [ "${ALLOW_LOST_REVISION:-0}" != "1" ]; then
    echo "[$(date '+%F %T')] SKIP $PAIR/$RUN — next_prompt.txt 가 없다(개정본 유실). ALLOW_LOST_REVISION=1 로 강행" >> "$llog"
    exit 0
  fi
fi

echo "[$(date '+%F %T')] starting $PAIR/$RUN ${extra[*]:-fresh}" >> "$llog"
$PY -u -m core.meaning_segmentator.autoseg.loop \
    --dataset fleurs-en-multi --src-lang English --tgt-lang English \
    --pair-id $PAIR --run-id $RUN \
    --model gpt-5-mini \
    --agent-reasoning-effort none --seg-reasoning-effort none \
    --iterations 6 --patience 6 \
    --train 30 --train-pool 90 --dev 215 --test 100 \
    --budget 25 --workers 24 \
    --translate-backend local \
    --adequacy-backend cometkiwi --adopt-se-mult 0.5 \
    "${extra[@]}" >> "$log" 2>&1
rc=$?
echo "[$(date '+%F %T')] exit=$rc" >> "$llog"
