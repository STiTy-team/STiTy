#!/bin/bash
# 오라클 그리디 — de·ja BLEU 재개 + COMET + 요약.
#
# **zh 는 다시 안 돈다.** 01:15:19 에 끝나 `bleu/zh.json` 에 들어갔다.
#
# 1차 체인을 중단하고 이걸로 갈아탄 이유: `bleu_eval` 이 번역 캐시를 기본값인 20건마다
# flush 했고, 그 flush 가 파일 전체를 다시 쓰는 구조라 캐시가 19MB 가 되자 디스크가
# 병목이 됐다. 실측 29.2건/초 — 같은 기계에서 라벨링이 낸 113건/초의 1/4 이다.
# `--cache-flush-every` 로 주기를 올린다. 캐시는 그대로 재사용되므로 잃는 진행은 없다.
set -u
REPO=/home/skkai/STiTy
cd "$REPO" || exit 1
export PYTHONPATH="$REPO"
PY="$REPO/.venv/bin/python"

RUN=en2x/covost2/full_oracle_greedy
R=core/meaning_segmentator/experiment/artifacts/$RUN
LAB=oracle_greedy
GRID="2 3 4 6"
mkdir -p "$R/logs" "$R/_status"
ts () { date '+%F %T'; }
mark () { echo "$(ts) $2" > "$R/_status/$1"; }

echo "===== BLEU ja (재개) $(ts) ====="
for t in ja; do
  echo "----- $t $(ts)"
  $PY -u -m core.meaning_segmentator.autoseg.scoring.bleu_eval \
      --run-id $RUN --label $LAB --split test \
      --dataset covost2 --manifest-tag full --src en --targets $t \
      --t-grid $GRID --src-spaced 1 --wordtimes qwen \
      --translate-engine local --local-mt-model google/madlad400-3b-mt --mt-batch 48 \
      --workers 24 --bootstrap 0 --no-sentence-bleu --no-auto-greedy \
      --cache-flush-every 20000 \
      > "$R/logs/bleu_$t.log" 2>&1
  echo "  $t exit=$? $(ts)"; tail -3 "$R/logs/bleu_$t.log"
done
missing=""
for t in de ja zh; do [ -f "$R/bleu/$t.json" ] || missing="$missing $t"; done
[ -z "$missing" ] || { mark bleu.failed "결손:$missing"; echo "!! bleu 결손:$missing"; exit 1; }
mark bleu.done "ok"

echo "===== COMET $(ts) ====="
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
    for name in ("unsegmented", "auto_T2", "auto_T3", "auto_T4", "auto_T6"):
        c = a["conditions"].get(name)
        if c:
            lbl = "오라클 " + name.replace("auto_", "") if name != "unsegmented" else name
            print(f"| {lbl} | {c['k']} | {c['laal_ms']} | {c['bleu']} | {c['chrf2']} | {c.get('comet')} |")
    for T in (2, 3, 4, 6):
        c = b["conditions"].get(f"auto_T{T}")
        if c:
            print(f"| run13 프롬프트 T{T} | {c['k']} | {c['laal_ms']} | {c['bleu']} | {c['chrf2']} | {c.get('comet')} |")
PYSUM
mark all.done "ok"
echo "===== 완료 $(ts) ====="
