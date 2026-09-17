#!/usr/bin/env python3
"""AlignAtt(FBK) SimulEval 산출을 BLEU·chrF2·COMET 로 채점한다.

    .venv/bin/python evaluation/ast/comet_alignatt.py \
        --runs /tmp/.../alignatt_acl_f2 /tmp/.../alignatt_acl_f4 ... \
        --meta ~/datasets/acl6060_clips/eval_en-de/meta.jsonl --tgt de

**재분절이 없다.** 입력 클립을 참조 문장 경계로 잘라 넣었으므로 SimulEval 인스턴스
하나가 참조 문장 하나다 — `comet_acl6060.py` 가 mwerSegmenter 를 거치는 이유(행 하나가
12분짜리 발표)가 여기엔 없다. 대신 **AlignAtt 에게 gold 분절을 준 조건**이라는 점을
표에 같이 적어야 한다.

`comet_acl6060.py` 와 같은 두 규칙을 지킨다.
  - src 는 **정답 영어 전사**(`meta.jsonl` 의 `src`)다. 시스템마다 다른 ASR 출력을 src 로
    쓰면 시스템 간 비교가 깨진다.
  - 빈 가설도 **안 버린다.** 어려운 문장을 빠뜨려 점수를 올리는 길을 막는다.

SimulEval 이 예측 끝에 붙이는 `</s>` 는 떼고 잰다 (SimulEval 자체 BLEU 도 뗀 값이다).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

EOS = re.compile(r"\s*</s>\s*$")
TOKENIZE = {"de": "13a", "es": "13a", "en": "13a", "ja": "ja-mecab", "zh": "zh", "ko": "ko-mecab"}


def load_instances(run_dir: Path) -> list[dict]:
    p = run_dir / "instances.log"
    return [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, help="SimulEval --output 디렉토리들")
    ap.add_argument("--meta", required=True, help="클립 meta.jsonl (src 전사와 wav 이름)")
    ap.add_argument("--tgt", default="de")
    ap.add_argument("--model", default="Unbabel/wmt22-comet-da")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    import sacrebleu
    from core.meaning_segmentator.autoseg.runtime import metrics
    # BLEU 는 `bleu_eval` 이 쓰는 함수를 그대로 쓴다 — signature 와 ja/zh 토크나이저
    # 폴백이 같아야 오라클 표와 같은 자가 된다.
    from evaluation.ast import metrics_ast as M

    meta = {}
    for line in Path(a.meta).open(encoding="utf-8"):
        d = json.loads(line)
        meta[d["wav"]] = d["src"]

    backend = metrics.CometBackend(model_name=a.model, batch_size=a.batch_size)
    tok = TOKENIZE.get(a.tgt, "13a")
    out: dict[str, dict] = {}

    for run in a.runs:
        run_dir = Path(run)
        rows = load_instances(run_dir)
        srcs, hyps, refs = [], [], []
        for r in rows:
            src_path = r["source"][0] if isinstance(r["source"], list) else r["source"]
            name = Path(src_path).name
            if name not in meta:
                print(f"  ! {run_dir.name}: {name} 가 meta 에 없다 — 중단", file=sys.stderr)
                return 2
            srcs.append(meta[name])
            hyps.append(EOS.sub("", r["prediction"]).strip())
            refs.append(r["reference"])

        scores = json.loads((run_dir / "scores").read_text(encoding="utf-8")) \
            if (run_dir / "scores").exists() else {}
        bleu, sig = M.corpus_bleu_score(hyps, refs, tok)
        chrf = sacrebleu.corpus_chrf(hyps, [refs], word_order=2)
        seg = backend.score(srcs, hyps, refs)
        cell = {
            "n": len(hyps),
            "bleu": round(bleu, 3) if bleu is not None else None,
            "bleu_signature": sig,
            "chrf2": round(chrf.score, 3),
            "comet": round(sum(seg) / len(seg), 4),
            "comet_seg": [round(float(x), 5) for x in seg],
            "empty_hyps": sum(1 for h in hyps if not h),
            "latency": scores.get("Latency", {}),
        }
        out[run_dir.name] = cell
        lat = cell["latency"]
        print(f"  {run_dir.name:20s} n={cell['n']:4d}  BLEU {cell['bleu']:6.2f}  "
              f"chrF2 {cell['chrf2']:6.2f}  COMET {cell['comet']:.4f}  "
              f"AL {lat.get('AL', float('nan')):.0f}ms  AL_CA {lat.get('AL_CA', float('nan')):.0f}ms  "
              f"빈가설 {cell['empty_hyps']}", flush=True)

    if a.out:
        Path(a.out).write_text(json.dumps(
            {"model": a.model, "tgt": a.tgt, "tokenize": tok, "meta": a.meta,
             "runs": out}, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"→ {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
