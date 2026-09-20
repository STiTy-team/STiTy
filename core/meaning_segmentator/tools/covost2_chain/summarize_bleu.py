"""`bleu/<타깃>.json` 에서 숫자만 뽑아 `bleu/summary.json` 으로 남긴다.

원본 한 개가 45~52MB 인데 그 부피는 전부 `hyps`(조각 번역 합본 15,530개)와
`comet_seg`(문장별 COMET)다. 표와 그림에 쓰는 것은 조건당 값 몇 개뿐이라, 그것만
남기면 수십 KB 가 된다. 재채점에는 원본이 필요하지만 그건 캐시에서 다시 만들어진다.

    python3 <이 파일> --run full_j44v0 [--run full_j44best]
"""
import argparse
import json
from pathlib import Path

A = Path(__file__).resolve().parents[2] / "experiment" / "artifacts" / "en2x" / "covost2"
KEEP = ("bleu", "chrf2", "comet", "k", "piece_units", "laal_words", "laal_ms",
        "bleu_signature")

p = argparse.ArgumentParser()
p.add_argument("--run", action="append", required=True)
p.add_argument("--targets", nargs="+", default=["zh", "de", "ja"])
a = p.parse_args()

for rid in a.run:
    out = {"run": rid, "targets": {}}
    for t in a.targets:
        path = A / rid / "bleu" / f"{t}.json"
        if not path.exists():
            continue
        blob = json.loads(path.read_text(encoding="utf-8"))
        out["targets"][t] = {
            "n": blob.get("n"), "tokenize": blob.get("tokenize"),
            "translator": blob.get("translator"), "comet_model": blob.get("comet_model"),
            "conditions": {n: {k: v[k] for k in KEEP if v.get(k) is not None}
                           for n, v in blob["conditions"].items()},
        }
    dst = A / rid / "bleu" / "summary.json"
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{dst}  ({dst.stat().st_size // 1024} KB, 타깃 {len(out['targets'])})")
