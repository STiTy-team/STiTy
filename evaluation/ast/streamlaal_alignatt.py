#!/usr/bin/env python3
"""AlignAtt(SimulEval) 산출을 **StreamLAAL** 로 잰다 — 우리 AST 트랙과 같은 자.

    .venv-streamlaal/bin/python evaluation/ast/streamlaal_alignatt.py \
        --runs .../alignatt_acl_f2 ... --meta .../meta.jsonl --lang de

왜 SimulEval 의 `AL` 을 그대로 쓰면 안 되나
------------------------------------------
SimulEval 1.0.2 가 내는 것은 Average Lagging(Ma et al. 2019)이고, `gamma` 를 **참조
길이**로만 정한다. 가설이 참조보다 길면 지연이 과소평가된다. StreamLAAL 이 쓰는 LAAL 은
`gamma = max(가설길이, 참조길이) / |X|` 라 그 편향이 없다. 우리 ACL6060 표가 StreamLAAL
이므로, 옆에 놓으려면 같은 식으로 다시 재야 한다.

재분절은 **안 한다.** 우리 파이프라인은 커밋 경계가 참조 문장 경계와 안 맞아
mwerSegmenter 를 거치지만, 여기 입력은 참조 문장 경계로 자른 클립이라 인스턴스 하나가
참조 문장 하나다. 대신 **AlignAtt 가 gold 분절을 받은 조건**이라는 점은 표에 적어야 한다.

SimulEval `instances.log` 규약 (실측 확인, 416/416)
  - `prediction.split()` 의 토큰 수 == `delays` 수, 마지막 토큰은 `</s>`
  - `delays`(비계산인지)·`elapsed`(계산인지)·`source_length` 는 전부 **밀리초**
EOS 토큰과 그 지연은 떼고 잰다.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from simulstream.metrics.readers import (  # noqa: E402
    OutputWithDelays, ReferenceSentenceDefinition, text_items)
from simulstream.metrics.scorers.latency.mwersegmenter import (  # noqa: E402
    ResegmentedLatencyScoringSample)
from simulstream.metrics.scorers.latency.stream_laal import StreamLaal  # noqa: E402

LANG_UNIT = {"de": "word", "ja": "char", "zh": "char"}
EOS = "</s>"


def build_sample(row: dict, ref_text: str, duration_sec: float, unit: str):
    """인스턴스 하나 → (`ResegmentedLatencyScoringSample`, 단위 수). 빈 가설이면 (sample, 0)."""
    toks = row["prediction"].split()
    delays = list(row["delays"])
    elapsed = list(row["elapsed"])
    if len(toks) != len(delays):
        raise AssertionError(f"{row.get('index')}: 토큰 {len(toks)} != delays {len(delays)}")
    if toks and toks[-1] == EOS:
        toks, delays, elapsed = toks[:-1], delays[:-1], elapsed[:-1]

    text = " ".join(toks)
    n_units = len(text_items(text, unit))
    if n_units != len(delays):
        # `char` 단위에서는 공백도 한 단위라 토큰 수와 안 맞는다. 단위 수에 맞춰 펼친다.
        if n_units and delays:
            per = []
            per_ca = []
            for i, t in enumerate(toks):
                k = len(text_items(t, unit))
                if i:                       # 토큰 사이 공백 — 앞 토큰의 지연을 물려받는다
                    k += len(text_items(" ", unit))
                per.extend([delays[i]] * k)
                per_ca.extend([elapsed[i]] * k)
            delays, elapsed = per[:n_units], per_ca[:n_units]
        else:
            delays, elapsed = [], []

    owd = OutputWithDelays(text, [d / 1000.0 for d in delays], [e / 1000.0 for e in elapsed])
    ref = ReferenceSentenceDefinition(content=ref_text, start_time=0.0, duration=duration_sec)
    return ResegmentedLatencyScoringSample(str(row.get("index")), [owd], [ref]), len(delays)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--lang", default="de")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    unit = LANG_UNIT[a.lang]
    scorer = StreamLaal(argparse.Namespace(latency_unit=unit))
    meta = {}
    for line in Path(a.meta).open(encoding="utf-8"):
        d = json.loads(line)
        meta[d["wav"]] = d

    out: dict[str, dict] = {}
    for run in a.runs:
        run_dir = Path(run)
        rows = [json.loads(l) for l in (run_dir / "instances.log").open(encoding="utf-8") if l.strip()]
        samples, per_sent, n_empty = [], [], 0
        for r in rows:
            src_path = r["source"][0] if isinstance(r["source"], list) else r["source"]
            m = meta[Path(src_path).name]
            # 길이는 gold 타이밍(`meta.jsonl`)에서 온다. SimulEval 의 `source_length` 와
            # 어긋나면 클립이 그 길이로 안 잘렸다는 뜻이므로 먼저 막는다.
            if abs(m["duration"] * 1000 - r["source_length"]) > 50:
                raise AssertionError(
                    f"{Path(src_path).name}: gold {m['duration']*1000:.0f}ms vs "
                    f"SimulEval {r['source_length']:.0f}ms")
            s, n = build_sample(r, r["reference"], float(m["duration"]), unit)
            if n == 0:
                n_empty += 1
                continue
            samples.append(s)
            h, ref = s.hypothesis[0], s.reference[0]
            tl = len(text_items(ref.content, unit))
            per_sent.append({
                "laal_sec": StreamLaal._sentence_level_laal(
                    [x - ref.start_time for x in h.ideal_delays], ref.duration, tl),
                "laal_ca_sec": StreamLaal._sentence_level_laal(
                    [x - ref.start_time for x in h.computational_aware_delays], ref.duration, tl),
            })

        scores = scorer._do_score(samples)
        mean_sent = statistics.mean(x["laal_sec"] for x in per_sent)
        # 검산 — 문장별 LAAL 의 평균은 공식 집계와 같아야 한다. 어긋나면 짝짓기가 틀린 것이다.
        if abs(mean_sent - scores.ideal_latency) > 1e-3:
            raise AssertionError(f"{run_dir.name}: 문장평균 {mean_sent:.4f} != "
                                 f"공식집계 {scores.ideal_latency:.4f}")
        cell = {
            "n_scored": len(samples), "n_empty_hyp": n_empty,
            "stream_laal_sec": round(scores.ideal_latency, 4),
            "stream_laal_ca_sec": round(scores.computational_aware_latency, 4),
            "laal_sec_per_sentence": [round(x["laal_sec"], 4) for x in per_sent],
        }
        out[run_dir.name] = cell
        print(f"  {run_dir.name:20s} n={cell['n_scored']:4d} 빈가설={n_empty}  "
              f"StreamLAAL {cell['stream_laal_sec']*1000:7.0f}ms  "
              f"CA {cell['stream_laal_ca_sec']*1000:7.0f}ms", flush=True)

    if a.out:
        Path(a.out).write_text(json.dumps(
            {"lang": a.lang, "latency_unit": unit, "resegmented": False, "runs": out},
            ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"→ {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
