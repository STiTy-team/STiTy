#!/bin/bash
# 비교군의 T 격자를 우리 곡선 끝까지 늘린다 — 맞대결이 성립하는 구간을 넓히려고.
#
# 왜 — 4점 격자로 처음 재고 9점으로 넓혔더니, **우리 곡선이 비교군보다 훨씬 길다**는
# 것이 드러났다. zh 실측으로 `syntax` 는 LAAL 3.07~3.92 어절, `causal_align` 은
# 3.25~4.05, `mu_prefix` 는 4.64~5.14 뿐인데 점수 임계값은 2.72~5.33 을 훑는다.
# 그래서 4 어절 오른쪽의 "우리가 이긴다" 는 비교군 두 점 사이를 직선으로 이어 만든
# 값이라 근거가 없다 — ja 는 4.70 과 6.26 사이 1.6 어절이 통째로 비어 있다.
#
# 어디까지 늘리나 — **코퍼스 길이 분포가 정한다.** CoVoST2 test 는 문장이 평균 9.1 어절
# (중앙 9, p10 5, p90 14)이라 `k = round(어절/T)` 가 T 10 부터 거의 전부 1 이 된다:
#
#     T      4     6     7     8    10    12    16    24
#     평균 k  2.39  1.59  1.36  1.27  1.05  1.01  1.00  1.00
#     무분절  17%   46%   65%   74%   96%   99%  100%  100%
#
# 12 위는 무분절 한 점 위에 조건이 겹쳐 쌓일 뿐이라 COMET 만 헛쓰고 그림에서도
# 오해를 부른다. 빈 구간(k 1.59 -> 1.05)을 7·10 으로 채우는 쪽이 맞다.
#
# **정수만으로는 성긴다.** T 2->3 에서 k 가 4.79->3.03 으로 건너뛰는데 우리 점수 격자의
# S60(k 3.91)·S80(2.93)이 바로 거기 있고, 1,379~1,826ms 구간에는 비교군 점이 아예 없어
# S90·S95 의 우위가 보간 위에 떠 있었다. 반올림 자리(2.5·3.5·4.5·5)를 넣어 k 간격을
# 0.2~0.4 로 고르게 만든다 — `--t-grid` 가 실수를 받게 고쳤다.
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
TGRID="${TGRID:-2 2.5 3 3.5 4 4.5 5 6 7 8 10}"
SGRID="${SGRID:-5 10 20 40 60 80 90 95 99}"
BASE="${BASE:-alignatt mu_prefix causal_align syntax}"
# 이전 격자를 옮겨 둘 곳. **COMET 을 물려받는 원본이므로 세대마다 다른 이름을 준다** —
# 같은 이름에 덮어쓰면 어느 코드로 잰 값인지 잃고, 낡은 값을 물려받을 위험이 생긴다.
ARCH="${ARCH:-bleu_t6}"
LOG=$SRC/logs/tgrid.log
ts () { date '+%F %T'; }

mkdir -p "$(dirname "$LOG")"
echo "== $(ts) T 격자 확장 대기: $WAIT" >> $LOG
until [ -f "$WAIT" ]; do sleep 30; done
echo "== $(ts) 선행 완료 확인 — 시작 (T: $TGRID)" >> $LOG

one () {   # <run 디렉토리 이름> <라벨 이름>
  local rid=en2x/covost2/$1 label=$2 D=$A/en2x/covost2/$1
  # 기존 9점 결과는 기록으로 옮긴다 — 같은 파일에 덮어쓰면 무엇으로 잰 값인지 잃는다.
  if [ -d $D/bleu ] && [ ! -d $D/$ARCH ]; then
    mkdir -p $D/$ARCH && mv $D/bleu/*.json $D/bleu/report.md $D/$ARCH/ 2>/dev/null
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
    rc=$?
    echo "== $(ts) $1 bleu_eval $t exit=$rc" >> $LOG
    # 이미 잰 조건의 COMET 을 새 파일로 옮겨 온다. **이름이 아니라 출력이 같을 때만
    # 옮긴다** — 예산 하한을 2 에서 1 로 내리면서 비교군 분절이 짧은 문장에서 달라졌으므로,
    # 이름만 보고 가져오면 `syntax_T2` 에 옛 분절의 점수가 박힌다. 우리 조건(auto_S*)과
    # 무분절·기계분절은 안 바뀌어 그대로 걸린다.
    $PY - "$D/$ARCH/$t.json" "$D/bleu/$t.json" <<'PYEOF' >> $LOG 2>&1
import json, sys
from pathlib import Path
old, new = Path(sys.argv[1]), Path(sys.argv[2])
if old.exists() and new.exists():
    o = json.loads(old.read_text())["conditions"]
    b = json.loads(new.read_text()); n = 0
    skipped = 0
    for name, cell in b["conditions"].items():
        src = o.get(name)
        if not (src and src.get("comet") is not None and cell.get("comet") is None):
            continue
        if src.get("hyps") != cell.get("hyps"):      # 분절이 바뀌었다 — 다시 재야 한다
            skipped += 1
            continue
        cell["comet"], cell["comet_seg"] = src["comet"], src.get("comet_seg")
        n += 1
    new.write_text(json.dumps(b, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"   COMET 재사용 {n}조건 / 분절이 바뀌어 재측정 {skipped}조건 → {new.name}")
PYEOF
  done
  echo "== $(ts) $1 comet (새 T 조건만)" >> $LOG
  $PY -u -m core.meaning_segmentator.autoseg.baselines.comet_score \
    --run-id $rid --dataset covost2 --manifest-tag full --src en \
    --label $label --split test --targets zh de ja --only-missing \
    --model Unbabel/wmt22-comet-da --batch-size 64 > $D/logs/comet.tgrid.log 2>&1
  # 종료코드는 명령 직후에 — echo 안의 $(ts) 가 $? 를 덮는다 (14b 주석 참조).
  rc=$?
  echo "== $(ts) $1 comet exit=$rc" >> $LOG
  [ $rc -eq 0 ] || return 1
}

one full_j44v0   auto_j44v0   || exit 1
one full_j44best auto_j44best || exit 1
$PY core/meaning_segmentator/tools/covost2_chain/plot_score_grid.py \
    --run full_j44v0 --run full_j44best --metric comet >> $LOG 2>&1
$PY core/meaning_segmentator/tools/covost2_chain/plot_score_grid.py \
    --run full_j44v0 --run full_j44best --metric bleu >> $LOG 2>&1
$PY core/meaning_segmentator/tools/covost2_chain/summarize_bleu.py \
    --run full_j44v0 --run full_j44best >> $LOG 2>&1
echo "== $(ts) ALL DONE" >> $LOG
