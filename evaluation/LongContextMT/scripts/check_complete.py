"""모델 하나의 1.1 결과가 (N, idx) 전부를 한 번씩 갖는지 본다. 다 차면 종료 코드 0."""
import json
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parents[1]
cfg = yaml.safe_load((HERE / "configs" / "experiment.yml").read_text(encoding="utf-8"))
model = sys.argv[1]
path = HERE / "results" / cfg["run_id"] / model / "translations.jsonl"
talk = sum(1 for line in (HERE / cfg["talk"]).read_text(encoding="utf-8").splitlines() if line.strip())
have = set()
for line in path.read_text(encoding="utf-8").splitlines():
    try:
        r = json.loads(line)
    except json.JSONDecodeError:
        continue
    have.add((r["n"], r["idx"]))
missing = {(n, i) for n in cfg["n_grid"] for i in range(talk)} - have
print(f"{model}: {len(have)} rows, {len(missing)} missing")
sys.exit(1 if missing else 0)
