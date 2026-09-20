#!/bin/bash
# 배치 크기가 어절당 비용을 얼마나 낮추나 — 60문장씩 재는 싼 실험.
#
# 왜 재나. 판정 프롬프트로 라벨링한 비용이 루프 실측의 **어절당 두 배**였다:
#   루프(judge44 iter2, 21어절 문장)  141 토큰/어절
#   covost2 라벨링(9.1어절 문장)      292 토큰/어절
# 문장당으로는 라벨링이 더 싼데 어절당으로는 두 배다. 출력의 90%가 사고 토큰이고 사고량은
# **호출당 기본량**이 있어 배치에 담긴 어절이 적으면 그 기본량을 적은 일에 나눠 쓴다.
#   루프   호출당 6문장 × 21어절 = 126어절
#   라벨링 호출당 6문장 × 9.1어절 =  55어절   ← 절반 이하
# 그러면 배치를 키워 호출당 어절을 맞추면 어절당 비용이 내려가야 한다. 그걸 확인한다.
#
# **캐시 키에 batch_size 가 들어가므로** 같은 문장을 배치만 바꿔 재면 전부 새로 부른다
# (`pipeline.cache_key`). 그래서 60문장으로 정직한 비교가 된다. 배치 6의 기준선은 본런
# 15,525문장 실측(292 토큰/어절, 문장당 $0.0054)을 쓴다 — 다시 재지 않는다.
#
#   bash core/meaning_segmentator/tools/covost2_chain/batch_probe.sh
set -u
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)" || exit 1
set -a; [ -f ./.env ] && . ./.env; set +a
export PYTHONPATH=. LANGSMITH_TRACING=0
PY=${PY:-.venv-autoseg/bin/python}
J=core/meaning_segmentator/experiment/artifacts/en2x/en-multi/judge44
F=core/meaning_segmentator/experiment/artifacts/en2x/covost2/full_judge44
N=${N:-60}
mkdir -p $F/probe

for B in ${BATCHES:-12 18}; do
  echo "===== batch $B / $N 문장 $(date '+%F %T') ====="
  $PY -u -m core.meaning_segmentator.tools.covost2_label.label_covost2 \
    --provider openai --model gpt-5-mini \
    --prompt $J/prompt_v0.txt \
    --manifest evaluation/ast/manifests/covost2_en-de_full.jsonl \
    --out $F/probe/b$B.jsonl --cache $F/probe/b$B.cache.json \
    --min-gap 1 --t-floor 2 --batch-size $B --workers 720 --limit $N \
    --max-tokens 32000 --timeout 600 --budget 3 --cache-every 1 \
    > $F/probe/b$B.log 2>&1
  echo "  exit=$?"
  $PY - "$F/probe/b$B.jsonl" "$F/probe/b$B.log" $B <<'PYSUM'
import json, re, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
t = open(sys.argv[2]).read()
d = json.loads(re.findall(r"\{[^{}]*\"format_pass\".*?\}", t, re.S)[-1])
words = sum(len(r["src_text"].split()) for r in rows)
cost = float(re.findall(r"누적 비용 \$([0-9.]+)", t)[-1]) if "누적 비용" in t else None
comp = int(re.findall(r"completion=(\d+)", t)[-1]) if "completion=" in t else None
calls = int(re.findall(r"calls=(\d+)", t)[-1]) if "calls=" in t else None
n = len(rows)
print(f"  batch {sys.argv[3]}: {n}문장 / {words}어절 / 호출 {calls} / {d['wall_sec']}초")
print(f"    format_pass {d['format_pass']} first_pass {d['first_pass']} "
      f"마커전부 {sum(1 for r in rows if r['n_boundaries']==r['required'])}/{n}")
if comp: print(f"    어절당 출력 {comp/words:.0f}토큰  (배치 6 기준선 292)")
if cost: print(f"    비용 ${cost:.4f} → 문장당 ${cost/n:.5f} / 어절당 ${cost/words:.6f}")
PYSUM
done
