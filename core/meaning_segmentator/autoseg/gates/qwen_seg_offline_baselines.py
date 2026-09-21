"""qwen `<SEG>` 확률을 **오프라인 점수**로 써서 auto_T 와 같은 규칙으로 자른 CoVoST2 비교군.

스트리밍판(`qwen_seg_baselines.py`)은 임계값 θ 로 그 자리에서 자른다. 이쪽은 문장 전체의
경계 점수를 먼저 매기고, 판단형 프롬프트 라벨(auto_judge13)과 **같은 절단기**
(`pipeline.truncate`, 목표 조각 길이 T, 점수 상위, 동점 앞쪽, min_gap 1)로 자른다 — 차이는
점수의 출처 하나뿐이다. 문장 끝을 봐야 하므로 스트리밍 정책이 아니다.

점수 두 벌 (`qwen_seg_prob.score_sentences`):
    p0     log P(<SEG> | 앞 어절들). 값 자체는 앞만 보지만 순위를 매기려면 문장 전체가 필요하다
    lrall  p0 + [뒤 어절 전부의 log P(<SEG> 넣음) − log P(안 넣음)] — 문장 끝까지 본다

산출: `baselines/qwen{p0,lrall}_T<T>_all_<split>.json` — `bleu_eval --baselines-native` 용.

    PYTHONPATH=. .venv/bin/python -m core.meaning_segmentator.autoseg.gates.qwen_seg_offline_baselines \
        --src-run en2x/covost2/full --run-id en2x/covost2/full_qwenseg_offline --t-grid 2 3 4 6
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

from ..paths import RUNS_DIR
from ..runtime import pipeline as P
from .qwen_seg_prob import DEFAULT_MODEL, load_thinker, log, score_sentences

KINDS = ("p0", "lrall")


def tagged(units: list[str], scores: list[float]) -> str:
    """경계 점수를 정수 태그로. truncate 는 순서만 보므로 단조 변환이면 된다 (소수 3자리 보존)."""
    out = units[0]
    for j in range(1, len(units)):
        out += f" <SEG:{int(round((scores[j - 1] + 100) * 1000))}> {units[j]}"
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src-run", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--t-grid", type=int, nargs="+", default=[2, 3, 4, 6])
    ap.add_argument("--min-gap", type=int, default=1, help="auto_judge13 CoVoST2 변환과 같은 값")
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--batch-size", type=int, default=32)
    a = ap.parse_args()

    src = json.loads((RUNS_DIR / a.src_run / "baselines" / f"punct_all_{a.split}.json").read_text(encoding="utf-8"))
    rows = [{"id": r["id"], "text": r["text"]} for r in src["rows"]]
    run_dir = RUNS_DIR / a.run_id
    (run_dir / "baselines").mkdir(parents=True, exist_ok=True)
    score_path = run_dir / f"qwen_scores_{a.split}_{Path(a.model).name}.json"
    if score_path.exists():
        qs = json.loads(score_path.read_text(encoding="utf-8"))
        log(f"[qwen] 캐시 {score_path.name}")
    else:
        tok, thinker = load_thinker(Path(a.model))
        qs = score_sentences(tok, thinker, [r["text"] for r in rows], "English", a.batch_size)
        score_path.write_text(json.dumps(qs), encoding="utf-8")
    assert len(qs) == len(rows)

    for kind in KINDS:
        for T in a.t_grid:
            name = f"qwen{kind}_T{T}"
            out = []
            for r, q in zip(rows, qs):
                u = r["text"].split()
                if len(u) < 2:
                    out.append({**r, "pieces": [r["text"]]})
                    continue
                cut, _miss = P.truncate(tagged(u, q[kind]), T, True, a.min_gap, higher_first=True)
                out.append({**r, "pieces": P.split_segments(cut) or [r["text"]]})
            (run_dir / "baselines" / f"{name}_all_{a.split}.json").write_text(
                json.dumps({"policy": name, "tgt": "all", "split": a.split, "score": kind, "T": T,
                            "min_gap": a.min_gap, "model": Path(a.model).name, "rows": out},
                           ensure_ascii=False), encoding="utf-8")
            log(f"[{name}] 조각 {st.mean(len(x['pieces']) for x in out):.2f}/문장")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
