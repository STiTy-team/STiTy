#!/bin/bash
# 비교군의 T 격자를 우리 곡선 끝까지 늘린다 — 맞대결이 성립하는 구간을 넓히려고.
#
# 왜 — 4점 격자로 처음 재고 9점으로 넓혔더니, **우리 곡선이 비교군보다 훨씬 길다**는
# 것이 드러났다. zh 실측으로 `syntax` 는 LAAL 3.07~3.92 어절, `causal_align` 은
# 3.25~4.05, `mu_prefix` 는 4.64~5.14 뿐인데 점수 임계값은 2.72~5.33 을 훑는다.
# 그래서 4 어절 오른쪽의 "우리가 이긴다" 는 비교군 두 점 사이를 직선으로 이어 만든
# 값이라 근거가 없다 — ja 는 4.70 과 6.26 사이 1.6 어절이 통째로 비어 있다.
#
# 어디까지 늘리나 — `syntax` 자신의 실측 기울기로 잡는다. T 를 2→6 으로 늘리는 동안
# LAAL 이 3.15→3.92 로 0.77 어절 올랐으니 T 한 칸이 0.2 어절 남짓이다. 우리 곡선의
# 끝(zh 5.33 / ja 5.55)에 닿으려면 T 24 까지 필요하다. 8·12·16·24 를 더한다.
#
# 비용: **API $0.** 번역은 로컬 madlad 고, 기존 조각은 ../full 캐시에 걸린다.
# 늘어나는 것은 새 T 조각의 로컬 번역뿐이다 (비교군 4 × T 4 × 타깃 3).
#
# `punct` 는 뺀다 — T 에 반응하지 않아 네 점이 LAAL 7.5 어절대에 겹쳐 쌓이기만 하고
# 곡선에 보태는 것이 없다. 그 자리를 COMET 으로 재는 데만 10분이 든다. 기존 값은
# bleu_t6/ 에 남아 있으니 필요하면 거기서 읽는다.
#
# 선행: 진행 중인 9점 평가가 끝나야 GPU 가 빈다. 마커 파일로 기다린다.
#
#   tmux new-session -d -s tgrid -c <저장소> "bash core/meaning_segmentator/tools/covost2_chain/14c_baseline_tgrid.sh"
set -u
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)" || exit 1
set -a; [ -f ./.env ] && . ./.env; set +a
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True LANGSMITH_TRACING=0
PY=${PY:-.venv-autoseg/bin/python}
A=core/meaning_segmentator/experiment/artifacts
SRC=$A/en2x/covost2/full_judge44
WAIT=${WAIT:-$A/en2x/covost2/full_j44best/eval.done}
TGRID="${TGRID:-2 3 4 6 8 12 16 24}"
SGRID="${SGRID:-5 10 20 40 60 80 90 95 99}"
BASE="${BASE:-alignatt mu_prefix causal_align syntax}"
LOG=$SRC/logs/tgrid.log
ts () { date '+%F %T'; }

mkdir -p "$(dirname "$LOG")"
echo "== $(ts) T 격자 확장 대기: $WAIT" >> $LOG
until [ -f "$WAIT" ]; do sleep 30; done
echo "== $(ts) 선행 완료 확인 — 시작 (T: $TGRID)" >> $LOG

one () {   # <run 디렉토리 이름> <라벨 이름>
  local rid=en2x/covost2/$1 label=$2 D=$A/en2x/covost2/$1
  # 기존 9점 결과는 기록으로 옮긴다 — 같은 파일에 덮어쓰면 무엇으로 잰 값인지 잃는다.
  if [ -d $D/bleu ] && [ ! -d $D/bleu_t6 ]; then
    mkdir -p $D/bleu_t6 && mv $D/bleu/*.json $D/bleu/report.md $D/bleu_t6/ 2>/dev/null
  fi
  for t in zh de ja; do
    [ -s $D/bleu/$t.json ] && { echo "== $(ts) $1 $t skip (있음)" >> $LOG; continue; }
    echo "== $(ts) $1 bleu_eval $t" >> $LOG
    $PY -u -m core.meaning_segmentator.autoseg.scoring.bleu_eval \
      --run-id $rid --label $label --split test \
      --dataset covost2 --manifest-tag full --targets $t \
      --t-grid $TGRID --score-grid $SGRID --no-auto-t --src-spaced 1 \
      --translate-engine local --local-mt-model google/madlad400-3b-mt --mt-batch 48 \
      --workers 24 --baselines $BASE --bootstrap 0 --no-sentence-bleu --no-auto-greedy \
      > $D/logs/bleu_eval_$t.tgrid.log 2>&1
    echo "== $(ts) $1 bleu_eval $t exit=$?" >> $LOG
    # 이미 잰 조건의 COMET 을 새 파일로 옮겨 온다 — 조건 하나는 독립으로 계산되므로
    # T 격자가 넓어져도 `syntax_T2` 의 값은 같다. 이걸 안 옮기면 --only-missing 이
    # 56조건을 전부 다시 재서 타깃당 25분을 헛쓴다.
    $PY - "$D/bleu_t6/$t.json" "$D/bleu/$t.json" <<'PYEOF' >> $LOG 2>&1
import json, sys
from pathlib import Path
old, new = Path(sys.argv[1]), Path(sys.argv[2])
if old.exists() and new.exists():
    o = json.loads(old.read_text())["conditions"]
    b = json.loads(new.read_text()); n = 0
    for name, cell in b["conditions"].items():
        src = o.get(name)
        if src and src.get("comet") is not None and cell.get("comet") is None:
            cell["comet"], cell["comet_seg"] = src["comet"], src.get("comet_seg")
            n += 1
    new.write_text(json.dumps(b, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"   COMET 재사용 {n}조건 → {new.name}")
PYEOF
  done
  echo "== $(ts) $1 comet (새 T 조건만)" >> $LOG
  $PY -u -m core.meaning_segmentator.autoseg.baselines.comet_score \
    --run-id $rid --dataset covost2 --manifest-tag full --src en \
    --label $label --split test --targets zh de ja --only-missing \
    --model Unbabel/wmt22-comet-da --batch-size 256 > $D/logs/comet.tgrid.log 2>&1
  echo "== $(ts) $1 comet exit=$?" >> $LOG
}

one full_j44v0   auto_j44v0   || exit 1
one full_j44best auto_j44best || exit 1
$PY core/meaning_segmentator/tools/covost2_chain/plot_score_grid.py \
    --run full_j44v0 --run full_j44best >> $LOG 2>&1
echo "== $(ts) ALL DONE" >> $LOG
