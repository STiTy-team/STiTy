#!/usr/bin/env python3
"""ACL 60/60 장문 `H_set` — 재분절 문장 단위로 낸다.

    .venv-autoseg/bin/python evaluation/ast/hset_acl6060.py \\
        --tag voxmix_t91c320_gate_20260921_173311 --split eval --langs de ja zh

    H_set(문장) = CometKiwi(gold 영어 문장, 재분절된 가설) × (1 − max contra(문장))

CoVoST2 쪽 정의(`runtime/hset.py`)와 같은 꼴이다. 다른 점은 조각 번역을 직접 이어
붙이는 대신 **mwerSegmenter 가 참조 문장 경계로 되붙인 가설**(`reseg_{split}_{tag}.json`)
을 쓴다는 것뿐이다 — 장문에서는 발화 하나가 12분짜리 발표라 문장 대응이 재분절로만
생긴다. `--reseg softseg` 를 주면 그 자리에 OmniSTEval 의 SoftSegmenter 재분절
(`longyaal_acl6060.py` 산출)을 쓴다. LongYAAL 과 같은 재분절 위에 올리려는 용도다.
두 항의 출처:

    QE     `reseg_{split}_{tag}.json` 의 `(src, hyp)`. src 는 gold 영어 전사다
           (ASR 출력을 쓰면 축마다 src 가 달라져 축 간 비교가 깨진다).
    contra `contra_{split}_{tag}_{lang}.json` 의 문장별 집계. `--contra-agg` 로 고른다.
           `mean`(기본) 은 문장 안 절단들의 평균, `max` 는 최댓값이다. 최댓값 쪽이
           목적함수로는 맞지만(한 번의 위험한 절단이 문장을 망친다), **같은 표 안의
           다른 행과 같은 집계를 써야 한다** — 섞으면 max 를 쓴 행만 구조적으로 낮다.
           잡음 바닥(c0)은 contra 쪽에서 이미 빼고 저장한다.

절단이 하나도 없는 문장(문장 안을 안 끊은 경우)은 contra 0 으로 둔다 — 위험을 안 만든
것이지 점수가 없는 게 아니다.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))


def load_softseg(workroot: Path, manifest_dir: Path, split: str) -> dict:
    """OmniSTEval 재분절 → `reseg_*.json` 과 같은 꼴 `{axis/lang: [{utt_id, seg_id, src, hyp}]}`.

    `instances.resegmented.jsonl` 의 `index` 는 longyaal_acl6060.py 가 segments.yaml 을 쓴
    순서, 곧 manifest 의 발표 순서 × 문장 순서다. 그 순서로 `(utt_id, seg_id)` 를 되찾는다.
    src 는 gold 영어 전사(manifest 의 `src`)다.
    """
    out: dict = {}
    if not workroot.exists():
        return out
    keys_by_lang: dict = {}
    for d in sorted(workroot.iterdir()):
        f = d / "omnisteval" / "instances.resegmented.jsonl"
        if not f.exists():
            continue
        axis, lang = d.name.rsplit("-", 1)
        if lang not in keys_by_lang:
            keys = []
            with (manifest_dir / f"acl6060_{split}_en-{lang}.jsonl").open(encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        e = json.loads(line)
                        keys += [(e["utt_id"], s["seg_id"], s["src"]) for s in e["sentences"]]
            keys_by_lang[lang] = keys
        keys = keys_by_lang[lang]
        rows = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
        if len(rows) != len(keys):
            raise SystemExit(f"!! {d.name}: 재분절 {len(rows)}문장 vs manifest {len(keys)}문장")
        out[f"{axis}/{lang}"] = [
            {"utt_id": keys[r["index"]][0], "seg_id": keys[r["index"]][1],
             "src": keys[r["index"]][2], "hyp": r["prediction"], "ref": r["reference"]}
            for r in rows]
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tag", required=True)
    p.add_argument("--split", default="eval")
    p.add_argument("--langs", nargs="+", default=["de", "ja", "zh"])
    p.add_argument("--axes", nargs="+", default=None,
                   help="미지정 시 reseg 파일에 있는 축 전부")
    p.add_argument("--contra-agg", default="mean", choices=["mean", "max"],
                   help="문장 안 절단들의 contra 집계. 표의 다른 행과 같은 값을 쓸 것")
    p.add_argument("--reseg", default="mwer", choices=["mwer", "softseg"],
                   help="QE 항에 쓸 재분절 가설. mwer = score_acl6060.py 의 reseg_*.json, "
                        "softseg = longyaal_acl6060.py 가 돌린 OmniSTEval 의 "
                        "instances.resegmented.jsonl. contra 항은 소스 쪽 값이라 둘 다 같다")
    p.add_argument("--manifest-dir", default=str(HERE / "manifests"))
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--results-root", default=str(HERE / "results" / "ACL6060"))
    p.add_argument("--out", default=None)
    a = p.parse_args()

    root = Path(a.results_root)
    if a.reseg == "mwer":
        reseg_path = root / f"reseg_{a.split}_{a.tag}.json"
        if not reseg_path.exists():
            print(f"재분절 산출이 없습니다: {reseg_path}\n"
                  f"  먼저: score_acl6060.py --tag {a.tag} --split {a.split}", file=sys.stderr)
            return 2
        reseg = json.loads(reseg_path.read_text(encoding="utf-8"))
    else:
        reseg = load_softseg(root / f"longyaal_{a.split}_{a.tag}", Path(a.manifest_dir), a.split)
        if not reseg:
            print(f"SoftSegmenter 산출이 없습니다: {root / f'longyaal_{a.split}_{a.tag}'}\n"
                  f"  먼저: longyaal_acl6060.py --tag {a.tag} --split {a.split}", file=sys.stderr)
            return 2

    from core.meaning_segmentator.autoseg.runtime import metrics
    qe = metrics.make_adequacy_backend("cometkiwi", batch_size=a.batch_size)

    out = {"tag": a.tag, "split": a.split, "contra_agg": a.contra_agg, "reseg": a.reseg,
           "definition": f"H_set = CometKiwi(gold en 문장, 재분절 가설) × (1 − {a.contra_agg} contra(문장))",
           "runs": {}}
    for key, rows in reseg.items():
        axis, lang = key.split("/")
        if lang not in a.langs or (a.axes and axis not in a.axes):
            continue
        cpath = root / f"contra_{a.split}_{a.tag}_{lang}.json"
        if not cpath.exists():
            print(f"[{key}] contra 산출 없음 — 건너뜀 ({cpath.name})", flush=True)
            continue
        per_sent = json.loads(cpath.read_text(encoding="utf-8"))["axes"][axis]["per_sentence"]

        scores = qe.score([r["src"] for r in rows], [r["hyp"] for r in rows])
        vals, n_nocut, n_missing = [], 0, 0
        for r, s in zip(rows, scores):
            k = f"{r['utt_id']}#{r['seg_id']}"        # contra 쪽 키 형식과 같다
            if k not in per_sent:
                n_missing += 1
                rec = None
            else:
                rec = per_sent[k]                        # None = 문장 안을 안 끊음
            if rec is None:
                n_nocut += 1
                c = 0.0
            else:
                c = rec["contra_max"] if a.contra_agg == "max" else rec["contra"]
            vals.append(s * (1.0 - c))
        out["runs"][key] = {"h_set": round(st.mean(vals), 4),
                            "qe_mean": round(st.mean(scores), 4),
                            "n_sentences": len(vals),
                            "n_no_cut": n_nocut, "n_unmatched": n_missing}
        print(f"[{key}] H_set {out['runs'][key]['h_set']:.4f}  "
              f"(QE {out['runs'][key]['qe_mean']:.4f}, 문장 {len(vals)}, "
              f"무분절 {n_nocut}, 미매칭 {n_missing})", flush=True)

    suffix = "" if a.contra_agg == "mean" else f"_{a.contra_agg}"
    stem = "hset" if a.reseg == "mwer" else "hset_softseg"
    dst = Path(a.out) if a.out else root / f"{stem}_{a.split}_{a.tag}{suffix}.json"
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"저장: {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
