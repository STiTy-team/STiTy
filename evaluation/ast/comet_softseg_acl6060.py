#!/usr/bin/env python3
"""ACL 60/60 장문 COMET — **SoftSegmenter(OmniSTEval) 재분절 문장 단위**로 낸다.

    .venv-autoseg/bin/python evaluation/ast/comet_softseg_acl6060.py \\
        --tag voxmix_t91c320_gate_arep_20260922_132907 --split eval

`comet_acl6060.py` 와 같은 일을 하되 재분절기가 다르다. 그쪽은 mwerSegmenter 가 남긴
`reseg_{split}_{tag}.json` 을 먹고, 이쪽은 `longyaal_acl6060.py` 가 돌린 OmniSTEval 의
`instances.resegmented.jsonl` 을 먹는다. LongYAAL 과 같은 재분절 위에 품질을 올려 표 한
장을 한 자로 통일하려는 것이다.

재분절기가 다르면 품질 점수도 다르다. 어느 쪽이든 조각을 참조 문장으로 되붙인 뒤 채점하므로
**커밋 경계가 문장을 자른 흔적은 둘 다 지워진다.** 조각화를 보려면 `contra_acl6060.py` 다.

채점 규칙은 `comet_acl6060.py` 와 같다:
  - src 는 gold 영어 전사(`srcs.txt`). ASR 출력을 쓰면 축마다 src 가 달라진다.
  - 빈 가설도 버리지 않는다. 빈 문자열로 채점되어 낮은 점수를 받는다.
  - 축 간 짝지은 부트스트랩은 문장 번호(`index`)로 짝짓는다. 참조 문장 집합은 축과 무관하다.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from comet_ast import paired_bootstrap  # noqa: E402

LANGS = ["de", "ja", "zh"]
ORDER = ["static-c3", "static-c4", "static-c5", "static-c6", "seg-c0.5", "seg-c1",
         "static-c10", "static-c12", "punct-c1", "punct"]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-root", default=str(HERE / "results" / "ACL6060"))
    p.add_argument("--tag", required=True)
    p.add_argument("--split", default="eval")
    p.add_argument("--langs", nargs="+", default=LANGS)
    p.add_argument("--model", default="Unbabel/wmt22-comet-da")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--gpus", type=int, default=1)
    p.add_argument("--out", default=None)
    a = p.parse_args()

    root = Path(a.results_root).expanduser().resolve()
    workroot = root / f"longyaal_{a.split}_{a.tag}"
    if not workroot.exists():
        raise SystemExit(f"!! 먼저 longyaal_acl6060.py 를 돌릴 것: {workroot} 없음")

    data: dict[tuple[str, str], list[dict]] = {}
    for d in sorted(workroot.iterdir()):
        f = d / "omnisteval" / "instances.resegmented.jsonl"
        if not f.exists():
            continue
        axis, lang = d.name.rsplit("-", 1)
        if lang not in a.langs:
            continue
        srcs = (d / "srcs.txt").read_text(encoding="utf-8").splitlines()
        rows = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
        if len(rows) != len(srcs):
            raise SystemExit(f"!! {d.name}: 재분절 {len(rows)}문장 vs src {len(srcs)}문장")
        data[(axis, lang)] = [
            {"index": r["index"], "src": srcs[r["index"]], "mt": r["prediction"],
             "ref": r["reference"]} for r in rows]
        print(f"  {axis:11s} {lang}  문장 {len(rows)}개")
    if not data:
        print("채점할 결과가 없습니다."); return 2

    from comet import download_model, load_from_checkpoint
    print(f"\nCOMET 모델 로드: {a.model}")
    model = load_from_checkpoint(download_model(a.model))

    seg_scores: dict[tuple[str, str], dict[int, float]] = {}
    system: dict[str, dict[str, float]] = {}
    n_empty: dict[str, dict[str, int]] = {}
    for (ax, lg), rows in sorted(data.items()):
        trip = [{"src": r["src"], "mt": r["mt"], "ref": r["ref"]} for r in rows]
        out = model.predict(trip, batch_size=a.batch_size, gpus=a.gpus, progress_bar=False)
        seg_scores[(ax, lg)] = {r["index"]: float(s) for r, s in zip(rows, out.scores)}
        system.setdefault(ax, {})[lg] = round(float(out.system_score), 4)
        n_empty.setdefault(ax, {})[lg] = sum(1 for r in rows if not r["mt"].strip())
        print(f"[{ax}/{lg}] COMET {out.system_score:.4f}  (빈 가설 {n_empty[ax][lg]})", flush=True)

    axes = [x for x in ORDER if any(k[0] == x for k in data)]
    axes += [x for x in sorted({k[0] for k in data}) if x not in axes]
    pairs_out: dict[str, dict] = {}
    for lg in a.langs:
        for x, y in itertools.combinations(axes, 2):
            if (x, lg) not in seg_scores or (y, lg) not in seg_scores:
                continue
            common = sorted(set(seg_scores[(x, lg)]) & set(seg_scores[(y, lg)]))
            r = paired_bootstrap([seg_scores[(x, lg)][k] for k in common],
                                 [seg_scores[(y, lg)][k] for k in common])
            r["n"] = len(common)
            pairs_out.setdefault(lg, {})[f"{x}-{y}"] = r

    # mwerSegmenter 기준 COMET 병기 — 같은 tag 의 comet_acl6060.py 산출물
    mwer_path = root / f"comet_{a.split}_{a.tag}.json"
    mwer = json.loads(mwer_path.read_text(encoding="utf-8"))["system"] if mwer_path.exists() else {}

    out_path = Path(a.out) if a.out else root / f"comet_softseg_{a.split}_{a.tag}.json"
    out_path.write_text(json.dumps({
        "comet_model": a.model, "split": a.split, "tag": a.tag,
        "resegmenter": "SoftSegmenter (OmniSTEval 0.1.10)",
        "system": system, "n_empty": n_empty, "paired_bootstrap": pairs_out,
        "system_mwerseg": mwer,
        "segment_scores": {f"{ax}/{lg}": {str(k): round(v, 5) for k, v in d.items()}
                           for (ax, lg), d in seg_scores.items()},
    }, ensure_ascii=False, indent=1), encoding="utf-8")

    print("\n=== COMET (system)  SoftSegmenter / mwerSegmenter ===")
    print(f"{'axis':11s} " + "  ".join(f"{lg:>15s}" for lg in a.langs))
    for ax in axes:
        cells = []
        for lg in a.langs:
            s = system.get(ax, {}).get(lg)
            m = mwer.get(ax, {}).get(lg)
            cells.append(f"{s if s is None else f'{s:.4f}'} / {m if m is None else f'{m:.4f}'}")
        print(f"{ax:11s} " + "  ".join(f"{c:>15s}" for c in cells))
    print(f"\n저장: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
