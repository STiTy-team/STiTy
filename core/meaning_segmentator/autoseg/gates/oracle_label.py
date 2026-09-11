"""오라클 라벨 실험 — 위치별 **실측** 점수가 절단 점수로 일하는가.

질문: 프롬프트가 매기는 점수 대신, 자리마다 실제로 잰 값(prefix 번역이 full 번역에
반박당하는 확률, 조각 QE)을 점수로 쓰면 같은 목적함수에서 얼마나 오르는가.

라벨은 **분할과 무관하게** 만든다. 위치 j 의 hypothesis 는 `words[:j]` 를 한 덩어리로
번역한 것이다 — 앞에 경계가 하나도 없는 셈이라 앞 조각의 오역이 끼어들 통로가 없고,
어떤 분할에서든 j 의 값은 하나다 (루프의 `pieces_contra` 는 누적 방출분이라 앞 조각
오역을 물려받는다). Zhang 2020 MU 의 접두사 판정을 함의 확률로 바꾼 것과 같다.

정책(점수)마다 같은 후보 집합을 `truncate` 로 자르고 **루프와 같은 `score_split`** 로
채점한다. 라벨을 목적함수 그 자체로 쓰지 않는 이유는 그게 자기 채점이기 때문이다 —
여기서도 절단 뒤의 `effective` 는 조각별 독립 번역·누적 NLI 그대로다.

LLM 호출 0. 로컬 MT + NLI + CometKiwi 만 돈다. 번역 캐시는 런의 것을 그대로 쓴다.

    PYTHONPATH=. python -m core.meaning_segmentator.autoseg.gates.oracle_label \
        --run-id en2x/en-multi/run14 --split dev --llm-iter 3
"""
from __future__ import annotations

import argparse
import json
import random
import statistics as st
import sys
import time
from pathlib import Path

from ..runtime import metrics
from ..runtime.pipeline import (JsonCache, LocalTranslator, TAG_RE, to_lang_code,
                                truncate, unit_count)
from ..loop import (DEFAULT_TARGET_POOL, load_contra_floor, resolve_targets, score_split,
                    target_is_spaced)
from . import noise_floor
from ..paths import RUNS_DIR

SCALE = 10000          # truncate 는 정수 점수만 읽는다. 0..1 라벨을 이 배수로 올린다


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _units(text: str, spaced: bool) -> list[str]:
    return text.split() if spaced else list(text.replace(" ", ""))


def _join(units: list[str], spaced: bool) -> str:
    return (" " if spaced else "").join(units)


def build_seg(units: list[str], scores: dict[int, float], spaced: bool) -> str:
    """위치 j(1..n−1) 에 점수 s∈[0,1] 를 단 태그를 넣는다. 점수가 없는 자리는 후보 아님."""
    out = units[0]
    for j in range(1, len(units)):
        if j in scores:
            s = max(0, min(SCALE, round(scores[j] * SCALE)))
            out = f"{out} <SEG:{s}> {units[j]}"
        else:
            out = out + (" " if spaced else "") + units[j]
    return out


def llm_positions(seg_text: str, spaced: bool) -> dict[int, int]:
    """LLM seg_text -> {위치 j: 점수}. 위치는 태그 앞까지의 단위 수."""
    parts = TAG_RE.split(seg_text)
    out: dict[int, int] = {}
    n = 0
    for i in range(0, len(parts), 2):
        n += unit_count(parts[i], spaced)
        if i + 1 < len(parts):
            out[n] = int(parts[i + 1]) if parts[i + 1] else 0
    return out


def spearman(a: list[float], b: list[float]) -> float | None:
    return metrics._spearman(a, b)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-id", required=True, help="예: en2x/en-multi/run14")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--llm-iter", type=int, default=None,
                    help="LLM 비교군으로 쓸 이터레이션 (기본: history 의 채택본)")
    ap.add_argument("--t-grid", type=int, nargs="*", default=None)
    ap.add_argument("--min-gap", type=int, default=None)
    ap.add_argument("--targets", nargs="*", default=None)
    ap.add_argument("--n", type=int, default=0, help="문장 수 상한 (스모크)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-floor", action="store_true")
    args = ap.parse_args()

    run_dir = RUNS_DIR / args.run_id
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    measured = json.loads((run_dir / "measured_profile.json").read_text(encoding="utf-8"))
    spaced = bool(measured.get("uses_spaces_between_words", True))
    min_gap = cfg["min_gap"] if args.min_gap is None else args.min_gap
    t_grid = args.t_grid or cfg.get("final_t_grid") or cfg["t_grid"]
    targets = args.targets or resolve_targets(cfg.get("tgt_langs") or DEFAULT_TARGET_POOL,
                                              cfg["src_lang"])
    sents = json.loads((run_dir / "data" / f"{args.split}.json").read_text(encoding="utf-8"))
    if args.n:
        sents = sents[: args.n]

    if args.llm_iter is None:
        hist = json.loads((run_dir / "history.json").read_text(encoding="utf-8"))
        adopted = [h["version"] for h in hist if h.get("adopted")]
        args.llm_iter = adopted[-1] if adopted else 0
    llm_rows = {r["id"]: r for r in json.loads(
        (run_dir / f"iter_{args.llm_iter:02d}" / f"{args.split}_rows.json")
        .read_text(encoding="utf-8"))}
    # 비교는 LLM 행이 유효한 문장에서만 — 정책 전부가 같은 문장 집합을 본다.
    sents = [s for s in sents if s["id"] in llm_rows and llm_rows[s["id"]].get("valid")]
    ids = [s["id"] for s in sents]
    texts = [s["text"] for s in sents]
    units = [_units(t, spaced) for t in texts]
    log(f"{args.run_id} {args.split}: 문장 {len(sents)} / 타깃 {targets} / T {t_grid} / "
        f"min_gap {min_gap} / spaced {spaced} / LLM iter {args.llm_iter}")

    adequacy = metrics.make_adequacy_backend(cfg.get("adequacy_backend", "cometkiwi"),
                                             batch_size=cfg.get("comet_batch_size", 64))
    contradiction = metrics.make_contradiction_backend()

    # ── 1. 라벨 ─────────────────────────────────────────────────────────
    translators: dict[str, LocalTranslator] = {}
    fulls: dict[str, list[str]] = {}
    labels: dict[str, list[dict]] = {}          # tgt -> 문장별 {키: [j=1..n-1]}
    label_path = run_dir / f"oracle_labels_{args.split}.json"
    cached_labels = (json.loads(label_path.read_text(encoding="utf-8"))
                     if label_path.exists() else {})
    for tgt in targets:
        code = to_lang_code(tgt)
        tsp = target_is_spaced(tgt)
        tr = LocalTranslator(tgt_code=code,
                             cache=JsonCache(run_dir / "cache" / f"translate_{code}.json"),
                             model_id=cfg.get("local_mt_model", "google/madlad400-3b-mt"))
        translators[tgt] = tr
        t0 = time.time()
        full = tr.full(texts)
        fulls[tgt] = full
        if tgt in cached_labels and [x["id"] for x in cached_labels[tgt]] == ids:
            labels[tgt] = cached_labels[tgt]
            log(f"[{tgt}] 라벨 캐시 사용 ({label_path.name})")
            continue
        pre_src, suf_src, owner = [], [], []
        for i, u in enumerate(units):
            for j in range(1, len(u)):
                pre_src.append(_join(u[:j], spaced))
                suf_src.append(_join(u[j:], spaced))
                owner.append(i)
        log(f"[{tgt}] full {len(texts)} / prefix {len(pre_src)} 번역 중")
        pre_tr = tr.full(pre_src)
        suf_tr = tr.full(suf_src)
        log(f"[{tgt}] 번역 {time.time() - t0:.0f}s, NLI + QE")
        prem = [full[i] for i in owner]
        contra, one_minus_ent = contradiction.score_dual(prem, pre_tr)
        adq_l = adequacy.score(pre_src, pre_tr)
        adq_r = adequacy.score(suf_src, suf_tr)
        floor_fn = None
        if not args.no_floor:
            floor_fn = load_contra_floor(
                run_dir, [{"full_trans": f} for f in full], contradiction,
                filename=f"contra_floor_{code}.json", tgt_spaced=tsp)
        per = [{"id": ids[i], "contra": [], "ent": [], "contra_floor": [],
                "adq_l": [], "adq_r": [], "hyp_units": []} for i in range(len(texts))]
        for k, i in enumerate(owner):
            hl = len(noise_floor.split_units(pre_tr[k], tsp))
            c0 = floor_fn(hl) if floor_fn else 0.0
            d = per[i]
            d["contra"].append(round(contra[k], 4))
            d["ent"].append(round(one_minus_ent[k], 4))
            d["contra_floor"].append(round(max(0.0, contra[k] - c0), 4))
            d["adq_l"].append(round(adq_l[k], 4))
            d["adq_r"].append(round(adq_r[k], 4))
            d["hyp_units"].append(hl)
        labels[tgt] = per
        cached_labels[tgt] = per
        label_path.write_text(json.dumps(cached_labels, ensure_ascii=False), encoding="utf-8")
        log(f"[{tgt}] 라벨 완료 {time.time() - t0:.0f}s -> {label_path.name}")

    # ── 2. 점수 정책 ────────────────────────────────────────────────────
    def mean_over_targets(key: str, i: int, j: int) -> float:
        return st.mean(labels[t][i][key][j - 1] for t in targets)

    rng = random.Random(args.seed)
    policies: dict[str, list[str]] = {}

    def all_pos(fn) -> list[str]:
        out = []
        for i, u in enumerate(units):
            sc = {j: fn(i, j, len(u)) for j in range(1, len(u))}
            out.append(build_seg(u, sc, spaced))
        return out

    policies["contra"] = all_pos(lambda i, j, n: 1 - mean_over_targets("contra", i, j))
    policies["contra_floor"] = all_pos(lambda i, j, n: 1 - mean_over_targets("contra_floor", i, j))
    policies["ent"] = all_pos(lambda i, j, n: 1 - mean_over_targets("ent", i, j))
    policies["contra_adqL"] = all_pos(
        lambda i, j, n: (1 - mean_over_targets("contra", i, j)) * mean_over_targets("adq_l", i, j))
    policies["contra_adqLR"] = all_pos(
        lambda i, j, n: (1 - mean_over_targets("contra", i, j))
        * (mean_over_targets("adq_l", i, j) + mean_over_targets("adq_r", i, j)) / 2)
    # NLI 세 확률은 합이 1 이라 contra 와 entail 을 같이 쓰는 것은 **neutral 을 얼마나
    # 벌하느냐**의 선택이다. `ent` 키는 1 − entail 로 저장돼 있다.
    def _adq_lr(i, j):
        return (mean_over_targets("adq_l", i, j) + mean_over_targets("adq_r", i, j)) / 2

    policies["ent_adqLR"] = all_pos(
        lambda i, j, n: (1 - mean_over_targets("ent", i, j)) * _adq_lr(i, j))
    policies["contra_ent_adqLR"] = all_pos(
        lambda i, j, n: (1 - mean_over_targets("contra", i, j))
        * (1 - mean_over_targets("ent", i, j)) * _adq_lr(i, j))
    policies["margin_adqLR"] = all_pos(
        lambda i, j, n: (2 - mean_over_targets("contra", i, j)
                         - mean_over_targets("ent", i, j)) / 2 * _adq_lr(i, j))
    policies["first"] = all_pos(lambda i, j, n: 1 - j / n)
    policies["random"] = all_pos(lambda i, j, n: rng.random())
    # LLM 비교군 — 같은 후보 위치에 (a) LLM 점수 (b) 오라클 라벨 (c) 앞쪽 우선
    llm_seg, llm_or, llm_first = [], [], []
    for i, u in enumerate(units):
        pos = llm_positions(llm_rows[ids[i]]["seg_text"], spaced)
        pos = {j: s for j, s in pos.items() if 1 <= j < len(u)}
        llm_seg.append(build_seg(u, {j: s / 100 for j, s in pos.items()}, spaced))
        llm_or.append(build_seg(u, {j: 1 - mean_over_targets("contra_floor", i, j)
                                    for j in pos}, spaced))
        llm_first.append(build_seg(u, {j: 1 - j / len(u) for j in pos}, spaced))
    policies["llm"] = llm_seg
    policies["llm_pos_oracle"] = llm_or
    policies["llm_pos_first"] = llm_first

    # ── 3. 절단 + 채점 ─────────────────────────────────────────────────
    results: dict[str, dict] = {}
    for name, segs in policies.items():
        results[name] = {}
        for T in t_grid:
            cuts = [truncate(s, T, spaced, min_gap)[0] for s in segs]
            per_t_eff: dict[str, list] = {}
            laal = None
            k = None
            for tgt in targets:
                sp = score_split(cuts, texts, fulls[tgt], translators[tgt], adequacy,
                                 spaced, target_is_spaced(tgt), contradiction)
                per_t_eff[tgt] = sp.effective
                laal, k = sp.laal_words, sp.k
            eff = []
            for i in range(len(texts)):
                v = [per_t_eff[t][i] for t in targets if per_t_eff[t][i] is not None]
                eff.append(st.mean(v) if v else None)
            ok = [x for x in eff if x is not None]
            results[name][str(T)] = {
                "effective": eff,
                "mean": round(st.mean(ok), 4) if ok else None,
                "p10": round(metrics.percentile10(ok), 4) if ok else None,
                "n": len(ok),
                "laal": round(st.mean(laal), 3),
                "k": round(st.mean(k), 3),
                "by_tgt": {t: round(st.mean([x for x in per_t_eff[t] if x is not None]), 4)
                           for t in targets},
            }
            log(f"{name:15s} T={T:<3d} eff={results[name][str(T)]['mean']} "
                f"p10={results[name][str(T)]['p10']} laal={results[name][str(T)]['laal']} "
                f"k={results[name][str(T)]['k']}")

    # ── 4. 쌍체 Δ (대 llm) ─────────────────────────────────────────────
    def paired(a: list, b: list) -> tuple[float, float, int]:
        d = [x - y for x, y in zip(a, b) if x is not None and y is not None]
        if len(d) < 2:
            return 0.0, 0.0, len(d)
        return st.mean(d), st.stdev(d) / len(d) ** 0.5, len(d)

    table = ["| 정책 | " + " | ".join(f"T={T}" for T in t_grid) + " | 격자 평균 |",
             "|---|" + "---|" * (len(t_grid) + 1)]
    summary: dict[str, dict] = {}
    for name in policies:
        cells, means = [], []
        for T in t_grid:
            r = results[name][str(T)]
            m, se, n = paired(r["effective"], results["llm"][str(T)]["effective"])
            cells.append(f"{r['mean']:.4f} ({m:+.4f}±{se:.4f})" if name != "llm"
                         else f"{r['mean']:.4f}")
            means.append(r["mean"])
            summary.setdefault(name, {})[str(T)] = {
                "mean": r["mean"], "p10": r["p10"], "laal": r["laal"], "k": r["k"],
                "delta_vs_llm": round(m, 4), "se": round(se, 4), "n": n,
                "by_tgt": r["by_tgt"]}
        table.append(f"| {name} | " + " | ".join(cells) + f" | {st.mean(means):.4f} |")

    # ── 5. 라벨 대 LLM 점수 정렬도 (LLM 후보 위치에서) ────────────────────
    pooled_s, pooled_l, within = [], [], []
    for i, u in enumerate(units):
        pos = llm_positions(llm_rows[ids[i]]["seg_text"], spaced)
        pos = {j: s for j, s in pos.items() if 1 <= j < len(u)}
        s = [pos[j] for j in sorted(pos)]
        l = [1 - mean_over_targets("contra_floor", i, j) for j in sorted(pos)]
        pooled_s += s
        pooled_l += l
        if len(s) >= 3:
            w = spearman(s, l)
            if w is not None:
                within.append(w)
    align = {"pooled_spearman": spearman(pooled_s, pooled_l),
             "within_sentence_mean": round(st.mean(within), 4) if within else None,
             "n_boundaries": len(pooled_s), "n_sentences_within": len(within)}

    out = {"run_id": args.run_id, "split": args.split, "llm_iter": args.llm_iter,
           "targets": targets, "t_grid": t_grid, "min_gap": min_gap, "n": len(texts),
           "summary": summary, "llm_score_vs_label": align,
           "note": ("effective = 정책별 절단 뒤 score_split (조각 독립 번역, 누적 NLI, "
                    "타깃 원값 평균). 괄호는 llm 대비 쌍체 Δ±se.")}
    out_path = run_dir / f"oracle_{args.split}.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    md = "\n".join(table)
    (run_dir / f"oracle_{args.split}.md").write_text(
        f"# 오라클 라벨 실험 — {args.run_id} {args.split} (n={len(texts)}, LLM iter {args.llm_iter})\n\n"
        f"{md}\n\n괄호: llm 대비 쌍체 Δ±se. LLM 점수 대 라벨(contra_floor) 정렬도: "
        f"{json.dumps(align)}\n", encoding="utf-8")
    print("\n" + md)
    print("\nLLM 점수 대 라벨 정렬도:", json.dumps(align))
    log(f"저장 {out_path.name}, oracle_{args.split}.md")
    for tr in translators.values():
        tr.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
