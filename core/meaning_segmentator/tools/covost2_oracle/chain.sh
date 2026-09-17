#!/bin/bash
# CoVoST2 en->{de,ja,zh} 전체 평가셋에서 그리디 오라클을 gold 참조로 잰다.
#
#   1. 경계별 라벨 (소스 NLI contra × 양쪽 CometKiwi) -> 상위 k 절단 -> prompt_eval
#   2. 타깃별 BLEU/chrF2/LAAL
#   3. COMET-DA
#
# API 비용 0 — madlad400-3b·CometKiwi·XLM-R NLI 전부 로컬이다. 드는 것은 GPU 시간뿐이라
# 비용 집계 대상이 아니다. 다만 단계마다 경과를 로그에 남겨 어디서 죽었는지 보이게 한다.
#
# 산출 런 디렉토리를 원 런과 **나눈다** — `bleu_eval` 은 조건 이름이 `auto_T*` 로 고정이라
# 같은 디렉토리에 쓰면 기존 제안 분절(auto_run13_mg1)의 값을 덮어쓴다.
set -u
REPO=/home/skkai/STiTy
cd "$REPO" || exit 1
export PYTHONPATH="$REPO"
PY="$REPO/.venv/bin/python"          # CometKiwi/COMET 는 이 venv 에만 있다

RUN=en2x/covost2/full_oracle_greedy
R=core/meaning_segmentator/experiment/artifacts/$RUN
LAB=oracle_greedy
GRID="2 3 4 6"                        # 기존 full 표와 같은 격자
mkdir -p "$R/logs" "$R/_status"
ts () { date '+%F %T'; }
mark () { echo "$(ts) $2" > "$R/_status/$1"; }

echo "===== [1/3] 라벨 + 절단 $(ts) ====="
$PY -u -m core.meaning_segmentator.tools.covost2_oracle.emit_oracle_greedy \
    --dataset covost2 --manifest-tag full --src en --targets de ja zh \
    --run-id $RUN --label $LAB --split test \
    --t-grid $GRID --min-gap 1 --min-duration 1.0 \
    --mt-model google/madlad400-3b-mt --mt-batch 48 --comet-batch 64 \
    --contra-source source --cache-flush-every 20000 \
    > "$R/logs/emit.log" 2>&1
rc=$?; echo "  emit exit=$rc $(ts)"; tail -6 "$R/logs/emit.log"
[ $rc -eq 0 ] || { mark emit.failed "exit=$rc"; exit 1; }
mark emit.done "ok"

echo "===== [2/3] BLEU $(ts) ====="
for t in zh de ja; do
  echo "----- $t $(ts)"
  $PY -u -m core.meaning_segmentator.autoseg.scoring.bleu_eval \
      --run-id $RUN --label $LAB --split test \
      --dataset covost2 --manifest-tag full --src en --targets $t \
      --t-grid $GRID --src-spaced 1 --wordtimes qwen \
      --translate-engine local --local-mt-model google/madlad400-3b-mt --mt-batch 48 \
      --workers 24 --bootstrap 0 --no-sentence-bleu --no-auto-greedy \
      > "$R/logs/bleu_$t.log" 2>&1
  echo "  $t exit=$? $(ts)"; tail -3 "$R/logs/bleu_$t.log"
done
missing=""
for t in de ja zh; do [ -f "$R/bleu/$t.json" ] || missing="$missing $t"; done
[ -z "$missing" ] || { mark bleu.failed "결손:$missing"; echo "!! bleu 결손:$missing"; exit 1; }
mark bleu.done "ok"

echo "===== [3/3] COMET $(ts) ====="
$PY -u -m core.meaning_segmentator.autoseg.baselines.comet_score \
    --run-id $RUN --dataset covost2 --manifest-tag full --src en \
    --label $LAB --split test --targets zh de ja --only-missing \
    --model Unbabel/wmt22-comet-da --batch-size 32 \
    > "$R/logs/comet.log" 2>&1
rc=$?; echo "  COMET exit=$rc $(ts)"; grep -E "조건 채점" "$R/logs/comet.log" | tail -5
[ $rc -eq 0 ] || { mark comet.failed "exit=$rc"; exit 1; }
mark comet.done "ok"

echo "===== 요약 $(ts) ====="
$PY - <<'PYSUM'
import json
from pathlib import Path
new = Path("core/meaning_segmentator/experiment/artifacts/en2x/covost2/full_oracle_greedy/bleu")
old = Path("core/meaning_segmentator/experiment/artifacts/en2x/covost2/full/bleu")
for t in ("de", "ja", "zh"):
    a = json.load(open(new / f"{t}.json"))
    b = json.load(open(old / f"{t}.json"))
    print(f"\n## en->{t}  (오라클 n={a['n']} / 기존 표 n={b['n']})")
    print("| 조건 | k | laal_ms | BLEU | chrF2 | COMET |")
    print("|---|---|---|---|---|---|")
    rows = [("unsegmented", a), *[(f"auto_T{T}", a) for T in (2, 3, 4, 6)]]
    for name, src in rows:
        c = src["conditions"].get(name)
        if not c:
            continue
        label = "그리디 오라클 " + name.replace("auto_", "") if name.startswith("auto") else name
        print(f"| {label} | {c['k']} | {c['laal_ms']} | {c['bleu']} | {c['chrf2']} | {c.get('comet')} |")
    for T in (2, 3, 4, 6):
        c = b["conditions"].get(f"auto_T{T}")
        if c:
            print(f"| run13 프롬프트 T{T} | {c['k']} | {c['laal_ms']} | {c['bleu']} | {c['chrf2']} | {c.get('comet')} |")
PYSUM
mark all.done "ok"
echo "===== 완료 $(ts) ====="
