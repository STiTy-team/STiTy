#!/usr/bin/env python3
"""ACL 60/60 논문 표의 `SEG (ours)` 행만 뽑아 LaTeX 로 찍는다.

    python evaluation/ast/ours_row_acl6060.py --tag <태그> --split eval [--axis seg-c1]

읽는 것 (모두 `results/ACL6060/` 아래):
    streamlaal_{split}_{tag}.json   StreamLAAL(ms) · BLEU   ← score_acl6060.py
    comet_{split}_{tag}.json        COMET                   ← comet_acl6060.py
    hset_{split}_{tag}.json         H_set                   ← hset_acl6060.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROW = r" & SEG (ours)  & {laal:d} & {bleu:.2f} & {comet:.4f} & {hset:.4f} \\"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tag", required=True)
    p.add_argument("--split", default="eval")
    p.add_argument("--axis", default="seg-c1")
    p.add_argument("--langs", nargs="+", default=["zh", "de", "ja"])
    p.add_argument("--keep-hset", default=None,
                   help="H_set 을 새로 재지 않고 기존 표 값을 그대로 옮긴다. 예: zh:0.7131,de:0.6959,ja:0.7019")
    p.add_argument("--results-root", default=str(HERE / "results" / "ACL6060"))
    a = p.parse_args()
    root = Path(a.results_root)

    laal = {r["axis"] + "/" + r["lang"]: r for r in
            json.loads((root / f"streamlaal_{a.split}_{a.tag}.json").read_text())["results"]}
    comet_path = root / f"comet_{a.split}_{a.tag}.json"
    hset_path = root / f"hset_{a.split}_{a.tag}.json"
    comet = json.loads(comet_path.read_text())["system"] if comet_path.exists() else {}
    hset = json.loads(hset_path.read_text())["runs"] if hset_path.exists() else {}
    # `--keep-hset` 은 **기존 표의 값을 그대로 옮긴다.** 그 값들은 문장당 contra 를
    # 평균으로 집계한 구현에서 나왔고, hset_acl6060.py 는 최댓값을 쓴다 — 자가 다르므로
    # 한 표 안에서 섞으면 안 된다. 다시 재려면 모든 축을 같은 스크립트로 내야 한다.
    keep = dict(kv.split(":") for kv in a.keep_hset.split(",")) if a.keep_hset else {}

    for lang in a.langs:
        key = f"{a.axis}/{lang}"
        r = laal.get(key)
        if r is None:
            print(f"% EN-{lang.upper()}: StreamLAAL 산출 없음")
            continue
        c = comet.get(a.axis, {}).get(lang)
        h = float(keep[lang]) if lang in keep else hset.get(key, {}).get("h_set")
        print(f"% EN-{lang.upper()}")
        if c is None or h is None:
            print(f" & SEG (ours)  & {round(r['stream_laal_sec'] * 1000)} & {r['bleu']:.2f} & "
                  f"{'--' if c is None else format(c, '.4f')} & "
                  f"{'--' if h is None else format(h, '.4f')} \\\\")
        else:
            print(ROW.format(laal=round(r["stream_laal_sec"] * 1000), bleu=r["bleu"],
                             comet=c, hset=h))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
