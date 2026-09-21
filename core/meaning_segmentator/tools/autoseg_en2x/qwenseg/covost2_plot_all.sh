#!/bin/bash
# 오프라인 체인(covost2_offline_eval.sh)이 합친 결과에 judge13 채택본의 auto_T 를 `judge13_T*` 로 얹어
# 한 장에 다시 그린다. full_judge13_cmp 는 full 과 무분절·기계분절·구두점 값이 소수점까지 같다
# (같은 15,530문장·같은 번역기·같은 강제정렬 지연) — 그래서 조건만 옮겨 담아도 비교가 성립한다.
#   tmux new-session -d -s qwenplot -c <저장소> "bash core/meaning_segmentator/tools/autoseg_en2x/qwenseg/covost2_plot_all.sh"
set -u
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)" || exit 1
export PYTHONPATH=.
PY=.venv/bin/python
A=core/meaning_segmentator/experiment/artifacts/en2x/covost2
RUN=en2x/covost2/full_qwenseg_offline
F=$A/full_qwenseg_offline
while [ ! -f $F/covost2_offline_eval.done ]; do sleep 60; done
echo "== $(date '+%F %T') 오프라인 체인 완료 확인"

$PY - <<'PYEOF'
import json
A = "core/meaning_segmentator/experiment/artifacts/en2x/covost2"
for t in ("de", "ja", "zh"):
    live = f"{A}/full_qwenseg_offline/bleu/{t}.json"
    d = json.load(open(live, encoding="utf-8"))
    j = json.load(open(f"{A}/full_judge13_cmp/bleu/{t}.json", encoding="utf-8"))["conditions"]
    for T in (2, 3, 4, 6):
        d["conditions"][f"judge13_T{T}"] = j[f"auto_T{T}"]
    json.dump(d, open(live, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"[{t}] judge13_T2~T6 추가 -> 조건 {len(d['conditions'])}개")
PYEOF

for M in bleu comet; do
  S=$([ $M = comet ] && echo _comet || echo "")
  $PY -m core.meaning_segmentator.autoseg.scoring.plot_tradeoff \
    --run-id $RUN --targets de ja zh --metric $M --t-grid 2 3 4 6 \
    --drop auto alignatt mu_prefix alignatt_native --out tradeoff_all$S 2>&1 | tail -2
done
echo "== $(date '+%F %T') 완료 -> $F/bleu/tradeoff_all{,_comet}.png"
