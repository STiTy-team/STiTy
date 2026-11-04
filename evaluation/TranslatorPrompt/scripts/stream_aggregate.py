"""2.1 집계 → results/<run_id>/stream_summary.{json,md}

대상은 두 조각 이상으로 잘린 문장이다 (t20: 321문장). 문장 하나가 표본 하나이고, P0 대비 차이는 같은 문장끼리
짝지어 부트스트랩 95% 구간을 낸다.

모순률      문장 안 경계마다 contradiction 확률, 문장 값은 그중 최댓값 (경계가 없으면 0 — 먼저 보여 준 것이 없다).
            경계 단위 평균과 0.5 를 넘는 경계 비율도 낸다.
충실도      내보낸 조각의 CometKiwi(조각 원문, 조각 번역). 마지막이 아닌 조각만 평균한 값이 주 지표다.
조기 완성률  마지막이 아닌 조각 중, 번역 글자 수 / 원문 글자 수가 그 문장 전체 번역의 같은 비율보다 1.5배 넘게
            긴 것의 비율 (공백 빼고 센다). 2배 기준도 같이 낸다.
닫힌 어미율  마지막이 아닌 조각 번역이 '…다.', '…요', '…까?' 처럼 문장을 끝내 버린 비율.
미룸        P3 에서 빈 출력으로 다음 조각에 넘긴 비율, 마지막 조각까지 비어 P1 로 다시 부른 수(forced).
형식 위반    내보낸 조각 번역을 2.4 규칙으로 판정 (앞 조각 번역을 그대로 되풀이한 것도 여기서 잡힌다).
            H 조건은 태그를 벗긴 <out> 으로 판정하고, 태그를 못 읽은 비율은 따로 낸다.
최종 품질   내보낸 조각 번역을 이어 붙인 것의 COMET·MetricX·Doc-COMET-w2. 옆에 같은 모델이 문장을 통째로 번역한
            것(2.3 S0)의 같은 값을 둔다.
보류        H 조건에서 원문을 남긴 경계 비율, 남긴 어절이 원문 전체 어절에서 차지하는 비율(한 조각씩 늦게 나간 양),
            원문에 없는 말을 남기려 해 버린 수(invalid), 마지막 조각에서도 남겨 번역되지 않은 수(final_hold).
"""
import json
import random
import re
import sys
from pathlib import Path
from statistics import mean

import yaml

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from stream_score import boundaries, full_translations, joined, key, read_jsonl  # noqa: E402
from tp.violations import check  # noqa: E402

CLOSED = re.compile(r"(다|요|죠|까|니다|습니다|세요)\s*[.!?…]*[\"'”’)]*\s*$|[.!?]\s*[\"'”’)]*\s*$")


def ci(d, n_boot=2000, seed=7):
    rng = random.Random(seed)
    ms = sorted(mean(rng.choices(d, k=len(d))) for _ in range(n_boot))
    return [ms[int(.025 * n_boot)], ms[int(.975 * n_boot)]]


def nchar(s: str) -> int:
    return len(re.sub(r"\s", "", s))


def sentence_stats(r: dict, full: str, scores: dict) -> dict:
    em = [(s, t) for s, t in r["emitted"]]
    nonfinal = em[:-1]
    bs = boundaries(r, full)
    contra = [scores[key("nli", p, h)]["contra"] for p, h in bs]
    kiwi = {i: scores[key("kiwi", s, t)]["kiwi"] for i, (s, t) in enumerate(em) if t}
    ratio_full = nchar(full) / max(nchar(r["en"]), 1)
    ratios = [nchar(t) / max(nchar(s), 1) / ratio_full for s, t in nonfinal if t]
    steps = r["steps"]
    prev, viol = [], 0
    for st in steps:
        if st.get("emitted") and st["hyp"]:
            raw = st["hyp"] if "hold" in st else st["raw_output"]
            viol += bool(check(raw, st["hyp"], st["src"], prev, st["truncated"], None))
            prev.append(st["hyp"])
    held = [st.get("hold", "") for st in steps if not st["final"]]
    nf_kiwi = [kiwi[i] for i in range(len(nonfinal)) if i in kiwi]
    return {
        "contra_max": max(contra, default=0.0),
        "contra_b": contra,
        "kiwi_nonfinal": mean(nf_kiwi) if nf_kiwi else None,
        "kiwi_all": mean(kiwi.values()) if kiwi else None,
        "early15": [x > 1.5 for x in ratios],
        "early20": [x > 2.0 for x in ratios],
        "closed": [bool(CLOSED.search(t)) for _, t in nonfinal if t],
        "n_emit": len(em),
        "deferred": sum(st["empty"] and not st["forced"] for st in steps),
        "forced": sum(st["forced"] for st in steps),
        "n_steps": len(steps),
        "viol": viol,
        "comet": scores[key("comet", r["en"], [joined(r), r["ref"]])]["comet"],
        "comet_full": scores[key("comet", r["en"], [full, r["ref"]])]["comet"],
        "metricx": scores[key("metricx", r["en"], [joined(r), r["ref"]])]["metricx"],
        "metricx_full": scores[key("metricx", r["en"], [full, r["ref"]])]["metricx"],
        "doc2": scores[key("doc2", r["idx"], joined(r))]["doc2"],
        "doc2_full": scores[key("doc2", r["idx"], full)]["doc2"],
        "hold_b": [bool(h) for h in held],
        "held_words": sum(len(h.split()) for h in held),
        "src_words": len(r["en"].split()),
        "hold_invalid": sum(st.get("hold_invalid", False) for st in steps),
        "parse_fail": sum(st.get("parse_fail", False) for st in steps),
        "final_hold": sum(bool(st.get("final_hold")) for st in steps),
    }


def frac(xs):
    return sum(xs) / len(xs) if xs else None


def main():
    cfg = yaml.safe_load((HERE / "configs" / "stream.yml").read_text(encoding="utf-8"))
    run_dir = HERE / "results" / (sys.argv[1] if len(sys.argv) > 1 else cfg["run_id"])
    scores = {c["key"]: c for c in read_jsonl(run_dir / "stream_scores.jsonl")}
    out, md = {}, []
    for mdir in sorted(p for p in run_dir.iterdir() if (p / "stream.jsonl").exists()):
        full = full_translations(cfg, mdir.name)
        rows = [r for r in read_jsonl(mdir / "stream.jsonl") if len(r["pieces"]) >= 2]
        conds = sorted({r["cond"] for r in rows})
        per = {c: {r["idx"]: sentence_stats(r, full[r["idx"]], scores) for r in rows if r["cond"] == c}
               for c in conds}
        # 문장 단위 값 (짝 비교용). 조각 비율은 문장 안에서 평균한 뒤 문장끼리 평균한다.
        sent = {
            "comet": lambda s: s["comet"],
            "metricx": lambda s: s["metricx"],
            "doc2": lambda s: s["doc2"],
            "contra_max": lambda s: s["contra_max"],
            "kiwi_nonfinal": lambda s: s["kiwi_nonfinal"],
            "early15": lambda s: frac(s["early15"]),
            "closed": lambda s: frac(s["closed"]),
        }
        res = {}
        for c in conds:
            v = per[c]
            b = [x for s in v.values() for x in s["contra_b"]]
            e = {"n_sent": len(v),
                 "comet": mean(s["comet"] for s in v.values()),
                 "comet_full": mean(s["comet_full"] for s in v.values()),
                 "metricx": mean(s["metricx"] for s in v.values()),
                 "metricx_full": mean(s["metricx_full"] for s in v.values()),
                 "doc2": mean(s["doc2"] for s in v.values()),
                 "doc2_full": mean(s["doc2_full"] for s in v.values()),
                 "hold_rate": frac([x for s in v.values() for x in s["hold_b"]]),
                 "held_share": sum(s["held_words"] for s in v.values()) / sum(s["src_words"] for s in v.values()),
                 "hold_invalid": sum(s["hold_invalid"] for s in v.values()),
                 "parse_fail": sum(s["parse_fail"] for s in v.values()),
                 "final_hold": sum(s["final_hold"] for s in v.values()),
                 "contra_max": mean(s["contra_max"] for s in v.values()),
                 "contra_boundary_mean": mean(b) if b else None,
                 "contra_boundary_gt05": frac([x > 0.5 for x in b]),
                 "n_boundaries": len(b),
                 "kiwi_nonfinal": mean(s["kiwi_nonfinal"] for s in v.values() if s["kiwi_nonfinal"] is not None),
                 "kiwi_all": mean(s["kiwi_all"] for s in v.values() if s["kiwi_all"] is not None),
                 "early15": frac([x for s in v.values() for x in s["early15"]]),
                 "early20": frac([x for s in v.values() for x in s["early20"]]),
                 "closed": frac([x for s in v.values() for x in s["closed"]]),
                 "defer_rate": sum(s["deferred"] for s in v.values()) / sum(s["n_steps"] for s in v.values()),
                 "forced": sum(s["forced"] for s in v.values()),
                 "emit_per_sent": mean(s["n_emit"] for s in v.values()),
                 "viol_rate": sum(s["viol"] for s in v.values()) / sum(s["n_emit"] for s in v.values())}
            if c != "P0" and "P0" in per:
                for m, f in sent.items():
                    d = [f(v[i]) - f(per["P0"][i]) for i in v
                         if i in per["P0"] and f(v[i]) is not None and f(per["P0"][i]) is not None]
                    if d:
                        e[f"{m}_delta"], e[f"{m}_ci95"] = mean(d), ci(d)
            res[c] = e
        out[mdir.name] = res

        def f(e, k, p=3):
            return "-" if e.get(k) is None else f"{e[k]:.{p}f}"

        def d(e, m, p=3):
            if f"{m}_delta" not in e:
                return "-"
            lo, hi = e[f"{m}_ci95"]
            star = "*" if lo > 0 or hi < 0 else ""
            return f"{e[m + '_delta']:+.{p}f} [{lo:+.{p}f}, {hi:+.{p}f}]{star}"

        md += [f"## {mdir.name}", "",
               "| 조건 | 최종 COMET | Δ (95%) | MetricX↓ | Δ (95%) | Doc-COMET-w2 | Δ (95%) | 통째 번역 COMET / MetricX / Doc | "
               "보류 경계 | 보류 어절 비율 | 태그 실패 | 보류 무효 | 마지막 보류 |",
               "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for c, e in res.items():
            md.append(f"| {c} | {f(e, 'comet')} | {d(e, 'comet')} | {f(e, 'metricx', 2)} | {d(e, 'metricx', 2)} | "
                      f"{f(e, 'doc2')} | {d(e, 'doc2')} | "
                      f"{f(e, 'comet_full')} / {f(e, 'metricx_full', 2)} / {f(e, 'doc2_full')} | {f(e, 'hold_rate')} | "
                      f"{f(e, 'held_share')} | {e['parse_fail']} | {e['hold_invalid']} | {e['final_hold']} |")
        md += ["",
               "| 조건 | 문장 | 모순률(문장 최댓값)↓ | Δ (95%) | 경계 평균↓ | 경계>0.5↓ | 충실도 Kiwi | Δ (95%) | "
               "조기 완성 1.5×↓ | Δ (95%) | 2×↓ | 닫힌 어미↓ | Δ (95%) | 미룸 | 방출/문장 | 위반 |",
               "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for c, e in res.items():
            md.append(f"| {c} | {e['n_sent']} | {f(e, 'contra_max')} | {d(e, 'contra_max')} | "
                      f"{f(e, 'contra_boundary_mean')} | {f(e, 'contra_boundary_gt05')} | {f(e, 'kiwi_nonfinal')} | "
                      f"{d(e, 'kiwi_nonfinal')} | {f(e, 'early15')} | {d(e, 'early15')} | {f(e, 'early20')} | "
                      f"{f(e, 'closed')} | {d(e, 'closed')} | {f(e, 'defer_rate')} | {f(e, 'emit_per_sent', 2)} | "
                      f"{f(e, 'viol_rate')} |")
        md.append("")
    (run_dir / "stream_summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    (run_dir / "stream_summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
