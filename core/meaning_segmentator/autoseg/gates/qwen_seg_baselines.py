"""CoVoST2 비교군 형식으로 qwen `<SEG>` 스트리밍 분절을 낸다 — `bleu_eval --baselines` 가 읽는 파일.

임계값 θ 마다 `baselines/qwenseg_th<θ>_all_<split>.json` 하나. 분절은 `batch_stream_cuts`
(= `StreamSegmenter(feed_seg=True)` 를 배치로 돌린 것, 문장마다 문맥을 새로 짠다). 문장 목록과
순서는 기존 `punct_all_<split>.json` 에서 그대로 가져온다 — 비교군과 ID 집합이 같아야 한다.

    PYTHONPATH=. .venv/bin/python -m core.meaning_segmentator.autoseg.gates.qwen_seg_baselines \
        --src-run en2x/covost2/full --run-id en2x/covost2/full_qwenseg --thresholds 0.05 0.1 0.2 0.3 0.5
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import time
from pathlib import Path

from ..paths import RUNS_DIR
from .qwen_seg_prob import DEFAULT_MODEL, load_thinker, log
from .qwen_seg_stream import batch_stream_cuts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src-run", required=True, help="punct_all_<split>.json 이 있는 런 (문장 목록 출처)")
    ap.add_argument("--run-id", required=True, help="산출을 쓸 런")
    ap.add_argument("--split", default="test")
    ap.add_argument("--thresholds", type=float, nargs="+", required=True)
    ap.add_argument("--min-words", type=int, default=2)
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--batch-size", type=int, default=64)
    a = ap.parse_args()

    src = json.loads((RUNS_DIR / a.src_run / "baselines" / f"punct_all_{a.split}.json").read_text(encoding="utf-8"))
    rows = [{"id": r["id"], "text": r["text"]} for r in src["rows"]]
    out_dir = RUNS_DIR / a.run_id / "baselines"
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"[data] {len(rows)}문장 (출처 {a.src_run}/baselines/punct_all_{a.split}.json)")
    tok, thinker = load_thinker(Path(a.model))
    for th in a.thresholds:
        path = out_dir / f"qwenseg_th{th}_all_{a.split}.json"
        if path.exists():
            log(f"[θ={th}] 이미 있음 — 건너뜀")
            continue
        t0 = time.time()
        cuts = batch_stream_cuts(tok, thinker, [r["text"] for r in rows], "English", th, a.min_words,
                                 a.batch_size, log=log)
        out = []
        for r, c in zip(rows, cuts):
            u = r["text"].split()
            b = [0, *c, len(u)]
            out.append({**r, "pieces": [" ".join(u[x:y]) for x, y in zip(b, b[1:])]})
        path.write_text(json.dumps({"policy": f"qwenseg_th{th}", "tgt": "all", "split": a.split,
                                    "threshold": th, "min_words": a.min_words, "feed_seg": True,
                                    "model": Path(a.model).name, "rows": out}, ensure_ascii=False),
                        encoding="utf-8")
        k = [len(x["pieces"]) for x in out]
        log(f"[θ={th}] 조각 {st.mean(k):.2f}/문장 / {time.time() - t0:.0f}s -> {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
