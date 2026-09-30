"""맥락 카드 집계 → results/<run_id>/summary.{json,md}

조건: K1~K4(이번 실행)와 비교군 N0·N8·N16(1.1 의 체인, 앞 번역 N개를 그대로 넣은 것). 모두 같은 번역 문면이다.
구간 둘: 첫 카드가 쓰이는 문장부터(score_from_idx) 끝까지, 그리고 1.1 과 같은 256번째부터.

문장 평균 COMET·MetricX(낮을수록 좋음)·Doc-COMET-w2(card_doc_comet.py). XCOMET-XL 은 이 머신의 HF 토큰으로 받을 수 없어(403) 뺐다, 코퍼스 chrF++, N0 대비 같은 문장끼리 차이와 부트스트랩 95% 구간.
말투: 문장 끝으로 합니다체·해요체·그 밖을 가르고, 정답과 같은 말투인 문장의 비율(말투 일치)과 합니다체 비율을 낸다.
비용: 번역 입력 토큰 평균과 조건별 추출 비용.
"""
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

import yaml

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE.parent / "LongContextMT" / "scripts"))

from score import key, read_jsonl  # noqa: E402
from card_doc_comet import doc_key  # noqa: E402

METRICS = ("comet", "metricx", "doc2")
HAMNIDA = re.compile(r"(니다|니까|십시오|시다)\s*[.!?…\"'”’)]*\s*$")
HAEYO = re.compile(r"요\s*[.!?…\"'”’)]*\s*$")


def register(s: str) -> str:
    s = s.strip()
    if HAMNIDA.search(s):
        return "합니다"
    if HAEYO.search(s):
        return "해요"
    return "기타"


def ci(d, n_boot=2000, seed=7):
    rng = random.Random(seed)
    ms = sorted(mean(rng.choices(d, k=len(d))) for _ in range(n_boot))
    return [ms[int(.025 * n_boot)], ms[int(.975 * n_boot)]]


def main():
    from sacrebleu.metrics import CHRF

    cfg = yaml.safe_load((HERE / "configs" / "card.yml").read_text(encoding="utf-8"))
    run_id = sys.argv[1] if len(sys.argv) > 1 else cfg["run_id"]
    run_dir = HERE / "results" / run_id
    mdir = run_dir / cfg["translator"]
    rows = read_jsonl(mdir / "translations.jsonl") + read_jsonl(mdir / "baseline_translations.jsonl")
    cache = {c["key"]: c for c in read_jsonl(run_dir / "scores_cache.jsonl")}
    doc2 = {c["key"]: c["doc2"] for c in read_jsonl(run_dir / "doc2_cache.jsonl")}

    def score_of(r, m):
        if m == "doc2":
            return doc2.get(doc_key(r["idx"], r["hyp"]))
        return cache.get(key(r["src"], r["hyp"], r["ref"]), {}).get(m)
    by = defaultdict(dict)
    for r in rows:
        by[r["cond"]][r["idx"]] = r
    conds = [c for c in ["N0", "N8", "N16"] + cfg["conditions"] if c in by]
    extract_cost = defaultdict(float)
    for u in read_jsonl(run_dir / "api_usage.jsonl"):
        if u.get("kind") == "extract":
            extract_cost[u["cond"]] += u["cost"]
    cards = read_jsonl(mdir / "cards.jsonl")
    chrf = CHRF(word_order=2)

    out, md = {}, []
    for lo, label in ((cfg["score_from_idx"], f"idx >= {cfg['score_from_idx']}"), (256, "idx >= 256 (1.1 과 같은 구간)")):
        res = {}
        base = {i: r for i, r in by["N0"].items() if i >= lo}
        for c in conds:
            v = {i: r for i, r in by[c].items() if i >= lo and i in base}
            if not v:
                continue
            e = {"n": len(v),
                 "chrfpp": chrf.corpus_score([r["hyp"] for r in v.values()], [[r["ref"] for r in v.values()]]).score,
                 "register_match": mean(register(r["hyp"]) == register(r["ref"]) for r in v.values()),
                 "hamnida": mean(register(r["hyp"]) == "합니다" for r in v.values()),
                 "input_tokens": mean(r["input_tokens"] for r in v.values()),
                 "latency_p50": median(r["latency_ms"] for r in v.values() if r.get("latency_ms") is not None)}
            for m in METRICS:
                s = {i: score_of(r, m) for i, r in v.items()}
                if any(x is None for x in s.values()):
                    continue
                e[m] = mean(s.values())
                if c != "N0":
                    b = {i: score_of(r, m) for i, r in base.items()}
                    d = [s[i] - b[i] for i in s]
                    e[f"{m}_delta"], e[f"{m}_ci95"] = mean(d), ci(d)
            if c != "N0":
                dr = [(register(r["hyp"]) == register(r["ref"])) - (register(base[i]["hyp"]) == register(base[i]["ref"]))
                      for i, r in v.items()]
                e["register_match_delta"], e["register_match_ci95"] = mean(dr), ci(dr)
            res[c] = e
        ref_h = mean(register(r["ref"]) == "합니다" for r in base.values())
        out[label] = {"conditions": res, "ref_hamnida": ref_h}

        def f(e, k, p=3):
            return "-" if e.get(k) is None else f"{e[k]:.{p}f}"

        def d(e, m, p=3):
            if f"{m}_delta" not in e:
                return "-"
            lo_, hi = e[f"{m}_ci95"]
            return f"{e[m + '_delta']:+.{p}f} [{lo_:+.{p}f}, {hi:+.{p}f}]" + ("*" if lo_ > 0 or hi < 0 else "")

        md += [f"### {label}", "",
               "| 조건 | 문장 | COMET | ΔCOMET (95%) | MetricX↓ | ΔMetricX (95%) | Doc-COMET-w2 | Δ (95%) | chrF++ | "
               "말투 일치 | Δ (95%) | 합니다체 | 입력 토큰 | 지연 p50 |",
               "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for c, e in res.items():
            md.append(f"| {c} | {e['n']} | {f(e, 'comet')} | {d(e, 'comet')} | "
                      f"{f(e, 'metricx', 2)} | {d(e, 'metricx', 2)} | {f(e, 'doc2')} | {d(e, 'doc2')} | {f(e, 'chrfpp', 1)} | {f(e, 'register_match')} | "
                      f"{d(e, 'register_match')} | {f(e, 'hamnida')} | {e['input_tokens']:.0f} | {e['latency_p50']:.0f} |")
        md += ["", f"정답의 합니다체 비율: {ref_h:.3f}", ""]
    md += ["### 추출", "", "| 조건 | 카드 판 | JSON 실패 | 추출 비용 |", "|---|---|---|---|"]
    for c in cfg["conditions"]:
        cs = [x for x in cards if x["cond"] == c]
        md.append(f"| {c} | {len(cs)} | {sum(not x['parsed'] for x in cs)} | ${extract_cost[c]:.4f} |")
        out.setdefault("extract", {})[c] = {"cards": len(cs), "parse_fail": sum(not x["parsed"] for x in cs),
                                            "cost": extract_cost[c]}
    (run_dir / "summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    (run_dir / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
