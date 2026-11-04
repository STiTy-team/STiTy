"""1.1·1.2 집계. results/<run>/summary.json 과 표(markdown)를 쓴다.

품질(1.1): 모델 × N, 채점 대상은 idx >= score_from_idx 문장(모든 N 에서 문맥이 꽉 찬 문장).
  문장 평균 COMET·XCOMET·CometKiwi·MetricX(낮을수록 좋음)·문장 chrF++, 코퍼스 chrF++·spBLEU,
  그리고 N=0 대비 쌍체 차이(같은 문장끼리)와 부트스트랩 95% 구간.
지연(1.1 로그): 모델 × N 의 입력 토큰·지연·TTFT 중앙값과 p95.
지연(1.2 스윕): 모델 × N 의 같은 지표. 1.1 과 달리 입력이 문맥 길이만 다르다.

로컬 모델 행 중 다른 프로세스가 GPU 를 같이 쓰던 시각(gpu_samples.csv)의 것은 지연 집계에서 뺀다.
"""
import argparse
import hashlib
import json
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean, median

HERE = Path(__file__).resolve().parents[1]
OWN = ("run_translate", "latency_sweep", "rerun_sdk")
KST = timezone(timedelta(hours=9))


def read_jsonl(path: Path):
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out


def key(src, hyp, ref):
    return hashlib.sha1(json.dumps([src, hyp, ref], ensure_ascii=False).encode()).hexdigest()


def pct(xs, q):
    xs = sorted(xs)
    if not xs:
        return None
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]


def foreign_gpu_times() -> list[datetime]:
    path = HERE / "logs" / "gpu_samples.csv"
    if not path.exists():
        return []
    times = []
    for line in path.read_text(encoding="utf-8").splitlines()[1:]:
        parts = line.split(",", 4)
        if len(parts) == 5 and parts[0] != "time" and not any(o in parts[4] for o in OWN):
            times.append(datetime.fromisoformat(parts[0]).replace(tzinfo=KST))
    return sorted(times)


def contaminated(at: str, foreign: list[datetime]) -> bool:
    t = datetime.fromisoformat(at)
    return any(abs((t - f).total_seconds()) <= 6 for f in foreign)


def latency_block(rows):
    lat = [r["latency_ms"] for r in rows]
    ttft = [r["ttft_ms"] for r in rows if r.get("ttft_ms") is not None]
    toks = [r["input_tokens"] for r in rows if r.get("input_tokens") is not None]
    return {"calls": len(rows), "input_tokens_median": median(toks) if toks else None,
            "latency_ms_median": median(lat) if lat else None, "latency_ms_p95": pct(lat, .95),
            "ttft_ms_median": median(ttft) if ttft else None, "ttft_ms_p95": pct(ttft, .95)}


def bootstrap_ci(deltas, n_boot=2000, seed=7):
    rng = random.Random(seed)
    k = len(deltas)
    means = sorted(mean(rng.choices(deltas, k=k)) for _ in range(n_boot))
    return [means[int(.025 * n_boot)], means[int(.975 * n_boot)]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--score-from", type=int, default=None, help="score_from_idx 덮어쓰기 (스모크)")
    args = ap.parse_args()
    import yaml

    cfg = yaml.safe_load((HERE / "configs" / "experiment.yml").read_text(encoding="utf-8"))
    run_dir = Path(args.run_dir) if args.run_dir else HERE / "results" / cfg["run_id"]
    start = cfg["score_from_idx"] if args.score_from is None else args.score_from
    cache = {c["key"]: c for c in read_jsonl(run_dir / "scores_cache.jsonl")}
    foreign = foreign_gpu_times()

    from sacrebleu.metrics import BLEU, CHRF

    chrf, bleu = CHRF(word_order=2), BLEU(tokenize="flores200")
    summary = {"run_dir": str(run_dir), "score_from_idx": start, "models": {}}
    for model_dir in sorted(p for p in run_dir.iterdir() if (p / "translations.jsonl").exists()):
        model = model_dir.name
        rows = read_jsonl(model_dir / "translations.jsonl")
        local = cfg["models"].get(model, {}).get("kind") == "local"
        by_n = {}
        for r in rows:
            by_n.setdefault(r["n"], []).append(r)
        base = {r["idx"]: r for r in by_n.get(0, []) if r["idx"] >= start}
        quality, latency = {}, {}
        for n in sorted(by_n):
            scored = sorted((r for r in by_n[n] if r["idx"] >= start), key=lambda r: r["idx"])
            q = {"sentences": len(scored),
                 "multi_line": sum(r["multi_line"] for r in scored),
                 "empty": sum(not r["hyp"] for r in scored)}
            if scored:
                q["corpus_chrfpp"] = chrf.corpus_score([r["hyp"] for r in scored],
                                                       [[r["ref"] for r in scored]]).score
                q["corpus_spbleu"] = bleu.corpus_score([r["hyp"] for r in scored],
                                                       [[r["ref"] for r in scored]]).score
            for m in ("comet", "xcomet", "kiwi", "metricx", "chrfpp"):
                vals = {r["idx"]: cache.get(key(r["src"], r["hyp"], r["ref"]), {}).get(m)
                        for r in scored}
                vals = {i: v for i, v in vals.items() if v is not None}
                if not scored:
                    continue
                if len(vals) != len(scored):
                    q[f"{m}_missing"] = len(scored) - len(vals)
                    continue
                q[m] = mean(vals.values())
                if n and base:
                    b = {i: cache.get(key(x["src"], x["hyp"], x["ref"]), {}).get(m)
                         for i, x in base.items()}
                    d = [vals[i] - b[i] for i in vals if b.get(i) is not None]
                    if d:
                        q[f"{m}_delta_vs_n0"] = mean(d)
                        q[f"{m}_delta_ci95"] = bootstrap_ci(d)
            quality[n] = q
            clean = [r for r in by_n[n] if not (local and contaminated(r["at"], foreign))]
            latency[n] = {"all": latency_block(clean),
                          "scored": latency_block([r for r in clean if r["idx"] >= start]),
                          "excluded_contaminated": len(by_n[n]) - len(clean)}
        sweep_rows = read_jsonl(model_dir / "latency_sweep.jsonl")
        sweep = {}
        for r in sweep_rows:
            if local and contaminated(r["at"], foreign):
                continue
            sweep.setdefault(r["n"], []).append(r)
        summary["models"][model] = {
            "quality": quality, "latency_1_1": latency,
            "latency_sweep": {n: latency_block(v) for n, v in sorted(sweep.items())}}

    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    lines = []
    for model, s in summary["models"].items():
        lines += [f"## {model}", "", "### 1.1 품질 (idx >= %d)" % start, "",
                  "| N | 문장 | COMET | ΔCOMET (95%) | XCOMET | ΔXCOMET (95%) | Kiwi | MetricX↓ | ΔMetricX (95%) | chrF++ | spBLEU | 여러 줄 |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        f = lambda v, d=4: "-" if v is None else f"{v:.{d}f}"  # noqa: E731
        ci = lambda q, m: ("-" if f"{m}_delta_vs_n0" not in q else  # noqa: E731
                           f"{q[m + '_delta_vs_n0']:+.4f} [{q[m + '_delta_ci95'][0]:+.4f}, {q[m + '_delta_ci95'][1]:+.4f}]")
        for n, q in s["quality"].items():
            lines.append(f"| {n} | {q['sentences']} | {f(q.get('comet'))} | {ci(q, 'comet')} | "
                         f"{f(q.get('xcomet'))} | {ci(q, 'xcomet')} | {f(q.get('kiwi'))} | "
                         f"{f(q.get('metricx'))} | {ci(q, 'metricx')} | "
                         f"{f(q.get('corpus_chrfpp'), 2)} | {f(q.get('corpus_spbleu'), 2)} | {q['multi_line']} |")
        for title, block in (("1.1 로그 지연 (idx >= %d)" % start,
                              {n: v["scored"] for n, v in s["latency_1_1"].items()}),
                             ("1.2 지연 스윕", s["latency_sweep"])):
            if not block:
                continue
            lines += ["", f"### {title}", "",
                      "| N | 호출 | 입력 토큰 | 지연 p50 | 지연 p95 | TTFT p50 | TTFT p95 |",
                      "|---|---|---|---|---|---|---|"]
            for n, b in block.items():
                lines.append(f"| {n} | {b['calls']} | {f(b['input_tokens_median'], 0)} | "
                             f"{f(b['latency_ms_median'], 0)} | {f(b['latency_ms_p95'], 0)} | "
                             f"{f(b['ttft_ms_median'], 0)} | {f(b['ttft_ms_p95'], 0)} |")
        lines.append("")
    (run_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
