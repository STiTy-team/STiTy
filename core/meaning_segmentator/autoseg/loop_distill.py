"""증류 루프 — 실측 라벨을 점수 전용 프롬프트로 옮긴다.

`loop.py` 와 다른 점 셋.

1. **후보 위치는 코드가 정한다.** 모든 단위 경계(양끝 `min_gap` 안쪽 제외)에 `<SEG:?>`
   를 박아 주고 모델은 점수만 채운다. 커버리지·간격 규칙이 프롬프트에서 사라진다.
2. **정답이 있다.** 위치마다 실측 라벨(`runtime/labels.py`, 런당 분할별 1회, LLM 0콜)이
   있으므로 이터레이션 평가에 번역·NLI·QE 가 필요 없다 — 분절 호출만 든다. 지표는 전부
   결정론이고 경계 단위다: 절단기가 모델 점수로 고른 집합의 라벨 질량을 라벨로 고른
   집합 대비 얼마나 달성했나(`achv`), 집합 겹침, 문장 내 순위 상관, 점수 구간별 보정표.
3. **판정자(A7)가 없다.** 라벨 분해(contra / 왼쪽 adq / 오른쪽 adq)가 이유를 말한다.
   Critic 은 모델이 과신·불신한 경계와 그 분해값을 받아 표면형 조건을 찾는다.

최종 test 만 실제 목적함수(`score_split` 의 effective)로 다시 재고, gold 참조 평가는
`prompt_eval/best_test.json` 을 `bleu_eval`/`comet_eval` 에 넣어 한다.

    PYTHONPATH=. python -m core.meaning_segmentator.autoseg.loop_distill \\
        --dataset fleurs-en-multi --src-lang English --pair-id en2x/en-multi --run-id run15 \\
        --labels-from en2x/en-multi/run14 --iterations 5 --budget 25
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import shutil
import statistics as st
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .infra.gateway import BudgetExceeded, Gateway, add_provider_args
from .runtime import agents, agents_distill as ad, data, labels as L, metrics
from .runtime.pipeline import (JsonCache, LocalTranslator, TAG_RE, boundaries,
                               segment_batch, split_segments, tag_positions, to_lang_code,
                               truncate)
from .loop import (DEFAULT_TARGET_POOL, MIN_GAP_MS, derive_min_gap, derive_t_grids,
                   resolve_targets, score_split, target_is_spaced)
from .paths import RUNS_DIR

SCALE = 10000
PROMPT_MAX_TOKENS = 16000
AGENT_MAX_TOKENS = 12000


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _seg_with(units: list[str], scores: dict[int, int], spaced: bool) -> str:
    out = units[0]
    for j in range(1, len(units)):
        out = (f"{out} <SEG:{scores[j]}> {units[j]}" if j in scores
               else out + (" " if spaced else "") + units[j])
    return out


def kept_positions(seg: str, T: int, spaced: bool, min_gap: int) -> list[int]:
    cut, _ = truncate(seg, T, spaced, min_gap)
    pos, _ = tag_positions(cut, spaced)
    return pos


# ── 평가 (결정론, 라벨 기반) ─────────────────────────────────────────────

def evaluate(gw: Gateway, prompt: str, sents: list[data.Sentence], labels: dict,
             spaced: bool, min_gap: int, t_grid: list[int], seg_cache: JsonCache,
             workers: int, batch_size: int, reasoning_effort: str | None,
             ) -> tuple[list[dict], dict]:
    texts = [s.text for s in sents]
    units = [L.units_of(t, spaced) for t in texts]
    cand = [ad.candidate_positions(len(u), min_gap) for u in units]
    idx = [i for i, c in enumerate(cand) if c]
    marked = [ad.mark_candidates(units[i], cand[i]) for i in idx]
    first_pass: list[dict] = []
    outs, ok1 = segment_batch(
        gw, prompt, marked, cache=seg_cache, workers=workers,
        validate_fn=lambda t, o: ad.validate_scored("", t, o, spaced),
        normalize_fn=ad.normalize_scored, reasoning_effort=reasoning_effort,
        batch_size=batch_size, first_pass_sink=first_pass)

    rows: list[dict] = []
    pairs_s: list[float] = []
    pairs_l: list[float] = []
    within: list[float] = []
    per_T_over: dict[int, list[float]] = {T: [] for T in t_grid}
    for k, i in enumerate(idx):
        u, c = units[i], cand[i]
        out = outs[k]
        viol = ad.validate_scored(sents[i].id, marked[k], out, spaced)
        row = {"id": sents[i].id, "text": texts[i], "marked": marked[k], "out": out,
               "valid": not viol, "first_pass": ok1[k],
               "violations": [v.rule for v in viol], "positions": c}
        if viol:
            rows.append(row)
            continue
        sc = ad.scores_of(out)
        lab = [L.label_value(labels, i, j) for j in c]
        row["scores"] = sc
        row["labels"] = [round(x, 4) for x in lab]
        pairs_s += sc
        pairs_l += lab
        if len(sc) >= 3:
            w = metrics._spearman([float(x) for x in sc], lab)
            if w is not None:
                within.append(w)
        seg_model = _seg_with(u, dict(zip(c, sc)), spaced)
        seg_label = _seg_with(u, {j: round(y * SCALE) for j, y in zip(c, lab)}, spaced)
        lab_at = dict(zip(c, lab))
        by_T = {}
        u_vals, o_vals = [], []
        for T in t_grid:
            if boundaries(texts[i], T, spaced) <= 0:
                continue
            km = kept_positions(seg_model, T, spaced, min_gap)
            ko = kept_positions(seg_label, T, spaced, min_gap)
            if not ko:
                continue
            um = st.mean(lab_at[j] for j in km) if km else 0.0
            uo = st.mean(lab_at[j] for j in ko)
            by_T[str(T)] = {"kept_model": km, "kept_label": ko, "util_model": round(um, 4),
                            "util_label": round(uo, 4),
                            "overlap": round(len(set(km) & set(ko)) / len(ko), 3)}
            per_T_over[T].append(by_T[str(T)]["overlap"])
            u_vals.append(um)
            o_vals.append(uo)
        row["by_T"] = by_T
        # 문장 안 동점 — `truncate` 는 동점을 앞쪽 우선으로 깨므로 절단이 내용이 아니라
        # 위치로 정해진다. 실측(run15 test): 경계의 71% 가 동점, 절단 경계선의 21% 가 동점.
        cnt = collections.Counter(sc)
        row["n_tied"] = sum(v for v in cnt.values() if v > 1)
        row["tie_at_cut"] = sum(
            1 for T in t_grid if str(T) in by_T
            for k_ in [len(by_T[str(T)]["kept_label"])]
            if 0 < k_ < len(sc) and sorted(sc, reverse=True)[k_ - 1] == sorted(sc, reverse=True)[k_])
        if u_vals:
            row["util"] = round(st.mean(u_vals), 4)
            row["util_oracle"] = round(st.mean(o_vals), 4)
            row["achv"] = round(st.mean(u_vals) / max(1e-9, st.mean(o_vals)), 4)
            row["overlap"] = round(st.mean(by_T[str(T_)]["overlap"] for T_ in t_grid
                                           if str(T_) in by_T), 4)
        rows.append(row)
    # 후보가 없는 짧은 문장은 행으로 남기되 점수 없음
    for i, c in enumerate(cand):
        if not c:
            rows.append({"id": sents[i].id, "text": texts[i], "valid": True,
                         "first_pass": True, "positions": [], "note": "no candidate"})
    scored = [r for r in rows if r.get("achv") is not None]
    bands: dict[str, list[float]] = {}
    for s_, y in zip(pairs_s, pairs_l):
        b = f"{min(int(s_) // 10 * 10, 90):02d}-{min(int(s_) // 10 * 10, 90) + 9:02d}"
        bands.setdefault(b, []).append(y)
    m = {
        "n": len(sents), "n_scored": len(scored),
        "format_pass_rate": round(sum(1 for r in rows if r.get("valid")) / max(1, len(rows)), 4),
        "format_pass_rate_no_retry": round(sum(1 for r in rows if r.get("first_pass")) / max(1, len(rows)), 4),
        "achv": round(st.mean(r["achv"] for r in scored), 4) if scored else None,
        "util": round(st.mean(r["util"] for r in scored), 4) if scored else None,
        "util_oracle": round(st.mean(r["util_oracle"] for r in scored), 4) if scored else None,
        "overlap": (round(st.mean(r["overlap"] for r in scored), 4) if scored else None),
        "overlap_by_T": {str(T): round(st.mean(v), 3) if v else None for T, v in per_T_over.items()},
        # 동점률 — 점수가 순위로 일할 수 있는 상태인가. 높으면 절단이 위치로 정해진다.
        "tie_rate": (round(sum(r.get("n_tied", 0) for r in scored)
                           / max(1, sum(len(r["scores"]) for r in scored)), 4) if scored else None),
        "tie_at_cut_rate": (round(sum(r.get("tie_at_cut", 0) for r in scored)
                                  / max(1, sum(len(r.get("by_T", {})) for r in scored)), 4)
                            if scored else None),
        "spearman_pooled": (round(metrics._spearman([float(x) for x in pairs_s], pairs_l), 4)
                            if len(pairs_s) > 2 else None),
        "spearman_within": round(st.mean(within), 4) if within else None,
        "calibration": {b: {"n": len(v), "label_mean": round(st.mean(v) * 100, 1)}
                        for b, v in sorted(bands.items())},
        "n_boundaries": len(pairs_s),
        # 1차 시도 위반 — 재시도 전 상태. 무엇이 깨지는지 안 남기면 재시도 비용의 원인을 못 찾는다.
        "first_pass_violations": {
            "counts": dict(sorted(collections.Counter(v["rule"] for v in first_pass).items())),
            "samples": [{"rule": v["rule"], "detail": v["detail"], "seg_text": v["seg_text"][:300]}
                        for v in first_pass[:3]],
        },
    }
    return rows, m


# ── 규칙 일반화 관문 ────────────────────────────────────────────────────

def _norm_tok(t: str) -> str:
    return "".join(ch for ch in t.lower() if ch.isalnum() or ch in "%'-")


def _tok_matches(tok: str, wanted: list[str]) -> bool:
    n = _norm_tok(tok)
    for w in wanted:
        if w == "NUM":
            if any(ch.isdigit() for ch in tok):
                return True
        elif w == "PUNCTEND":
            if tok[-1:] in ".?!…。？！":
                return True
        elif n == _norm_tok(w):
            return True
    return False


def validate_rules(critique: dict, rows: list[dict], units_by_id: dict,
                   min_matches: int, min_t: float) -> tuple[dict, list[dict]]:
    """제안 규칙을 **라벨 전량에 대고** 재서 일반화 못 하는 것을 걸러낸다.

    한 문장에서 뽑은 규칙이 검증 없이 그대로 프롬프트에 들어가고 있었다. run18 채택
    개정의 규칙들이 `en_us_66`, `en_us_278` 처럼 문장 id 를 호명했고, 그 개정본은 dev
    에서는 이겼지만 **test 쌍체 -0.0603±0.0228 (2.6 se) 로 일반화에 실패했다.**
    홀드아웃도 dev 도 선택에 쓴 집합이라 이 실패를 못 잡는다.

    라벨은 train 전 문장의 **모든 경계**를 덮고 있고 LLM 호출이 0 이다. 규칙이 실제로
    몇 자리에 걸리는지, 그 자리들의 라벨이 주장한 방향인지를 공짜로 잴 수 있다.
    run15 커밋 본문이 손으로 하던 계산(관계사 +0.081±0.022, 전치사 -0.015±0.014 를 재서
    "넷 중 셋이 반대" 판정)을 관문으로 만든 것이다.

    통과 조건 둘: 걸리는 경계가 `min_matches` 이상이고, 걸린 자리의 라벨 평균이 나머지와
    `min_t` 표준오차 이상 벌어지며 **부호가 `direction` 과 맞을 것**. 떨어진 규칙은
    프롬프트에 안 들어가지만 판정 내역은 산출물에 남는다.
    """
    pool: list[tuple[str, str, float]] = []      # (앞 토큰, 뒤 토큰, 라벨)
    for r in rows:
        u = units_by_id.get(r["id"])
        if not u or not r.get("labels"):
            continue
        for j, y in zip(r["positions"], r["labels"]):
            if 0 < j < len(u):
                pool.append((u[j - 1], u[j], y))
    verdicts: list[dict] = []
    kept: list[dict] = []
    for c in critique.get("cases") or []:
        chk = c.get("check") or {}
        left, right = chk.get("left_last") or [], chk.get("right_first") or []
        v = {"id": c.get("id"), "direction": c.get("direction"),
             "surface_condition": (c.get("surface_condition") or "")[:120]}
        if not left and not right:
            v.update(verdict="drop", reason="check 없음")
            verdicts.append(v)
            continue
        hit = [y for (lt, rt, y) in pool
               if (not left or _tok_matches(lt, left)) and (not right or _tok_matches(rt, right))]
        miss = [y for (lt, rt, y) in pool
                if not ((not left or _tok_matches(lt, left))
                        and (not right or _tok_matches(rt, right)))]
        v["n_matched"] = len(hit)
        if len(hit) < min_matches or len(miss) < 2:
            v.update(verdict="drop", reason=f"걸린 경계 {len(hit)} < {min_matches}")
            verdicts.append(v)
            continue
        mh, mm = st.mean(hit), st.mean(miss)
        se = ((st.pvariance(hit) / len(hit)) + (st.pvariance(miss) / len(miss))) ** 0.5
        t = (mh - mm) / se if se > 0 else 0.0
        # `lower` 는 "이 자리 점수를 내려라" 이므로 라벨이 **낮아야** 맞다.
        want_neg = (c.get("direction") or "lower") == "lower"
        ok = (abs(t) >= min_t) and ((t < 0) if want_neg else (t > 0))
        v.update(label_matched=round(mh, 4), label_rest=round(mm, 4), t=round(t, 2),
                 verdict="keep" if ok else "drop")
        if not ok:
            v["reason"] = ("방향 반대" if abs(t) >= min_t else f"|t| {abs(t):.2f} < {min_t}")
        else:
            kept.append(c)
        verdicts.append(v)
    out = dict(critique)
    out["cases"] = kept
    return out, verdicts


def moved_sentences(rows_a: list[dict], rows_b: list[dict], t_grid: list[int],
                    key: str = "overlap", top_n: int = 5, gains: bool = False) -> list[dict]:
    """`rows_a` 가 `rows_b` 대비 가장 크게 **떨어진**(또는 `gains` 면 오른) 문장들.

    **`metrics.regressions` 를 쓸 수 없다.** 그쪽은 `by_T[T]["seg_text"]` 로 분절이 실제로
    바뀌었는지를 보는데, 증류 루프의 `by_T` 에는 그 키가 없다 (`kept_model` /
    `kept_label` / `util_*` / `overlap` 뿐). `a.get("seg_text") != c.get("seg_text")` 가
    `None != None` 이라 **항상 거짓**이 되어 결과가 언제나 빈 리스트다 — 이식했던 거부
    부검이 조용히 죽어 있던 이유다.

    여기서는 분절이 바뀌었는지를 `kept_model`(절단기가 실제로 남긴 경계)로 판정한다.
    같은 경계 집합인데 값이 다르면 그건 개정이 한 일이 아니라 분절 비결정성이므로
    귀책할 대상이 없다 — `metrics.regressions` 의 판단과 같고 근거만 이 루프의 것이다.

    싣는 예산 T 는 **변화가 가장 큰 하나**뿐이다. 격자 전체를 실으면 사례 하나가 T
    개수만큼 부풀어 에이전트 프롬프트를 삼킨다.
    """
    b = {r["id"]: r for r in rows_b}
    out: list[dict] = []
    for r in rows_a:
        c = b.get(r["id"])
        if not c:
            continue
        total, seen, pick = 0.0, 0, None
        for T in t_grid:
            x = (r.get("by_T") or {}).get(str(T))
            y = (c.get("by_T") or {}).get(str(T))
            if not x or not y or x.get(key) is None or y.get(key) is None:
                continue
            gap = x[key] - y[key]
            total += gap
            seen += 1
            if x.get("kept_model") != y.get("kept_model"):
                if pick is None or (gap > pick[1] if gains else gap < pick[1]):
                    pick = (T, gap, x, y)
        if not seen or pick is None:
            continue
        mean_d = total / seen
        if (mean_d <= 0) if gains else (mean_d >= 0):
            continue
        T, gap, x, y = pick
        out.append({
            "id": r["id"], "text": r.get("text"),
            "delta": round(mean_d, 5), "budget_T": T, "delta_at_T": round(gap, 5),
            "kept_now": x.get("kept_model"), "kept_before": y.get("kept_model"),
            "kept_by_label": x.get("kept_label"),
            "overlap_now": x.get(key), "overlap_before": y.get(key),
        })
    return sorted(out, key=lambda d: -d["delta"] if gains else d["delta"])[:top_n]


def paired(rows_a: list[dict], rows_b: list[dict], key: str = "overlap") -> dict:
    b = {r["id"]: r.get(key) for r in rows_b}
    d = [r[key] - b[r["id"]] for r in rows_a
         if r.get(key) is not None and b.get(r["id"]) is not None]
    if len(d) < 2:
        return {"mean_delta": 0.0, "se_delta": 0.0, "n_pairs": len(d)}
    return {"mean_delta": round(st.mean(d), 4),
            "se_delta": round(st.stdev(d) / len(d) ** 0.5, 4), "n_pairs": len(d)}


# ── Critic 사례 ─────────────────────────────────────────────────────────

def build_cases(rows: list[dict], labels: dict, sent_index: dict[str, int], units_by_id: dict,
                main_T: int, spaced: bool, n_cases: int = 8,
                t_grid: list[int] | None = None) -> list[dict]:
    """Critic 이 볼 실패 사례 — 손실 상위 `n_cases` 문장.

    **작동점을 `main_T` 하나로 두지 않는다.** 종전에는 주 작동점에서만 골라, 거기서는
    멀쩡한데 다른 예산에서 무너지는 문장이 Critic 에게 아예 안 보였다. 문장마다 격자
    전체에서 가장 크게 진 T 를 골라 그 자리를 싣는다 — `metrics.regressions` 가 거부
    부검에 쓰는 것과 같은 방식이고, 사례 하나가 T 개수만큼 부풀지 않는다.

    격자를 안 주면 종전대로 `main_T` 만 본다.
    """
    grid = t_grid or [main_T]

    def worst(r: dict) -> tuple[int, float] | None:
        """(가장 크게 진 T, 그 손실). 채점된 예산이 없으면 None."""
        best_at = None
        for T in grid:
            bt = (r.get("by_T") or {}).get(str(T))
            if not bt:
                continue
            loss = bt["util_label"] - bt["util_model"]
            if best_at is None or loss > best_at[1]:
                best_at = (T, loss)
        return best_at

    def failure_kind(r: dict, T_at: int) -> str:
        """이 문장이 **왜** 졌나 — 잘못 남긴 경계들의 지배적 성분.

        라벨은 세 성분으로 분해된다: `contra`(조기 방출 — 뒤가 앞을 뒤집는다),
        `adq_left`/`adq_right`(조각이 파편이라 그 자체로 번역이 안 된다). 경계마다 가장
        나쁜 성분을 고르고 문장 단위로 다수결한다.
        """
        bt = r["by_T"][str(T_at)]
        bad = set(bt["kept_model"]) - set(bt["kept_label"])
        if not bad:
            return "other"
        i = sent_index[r["id"]]
        votes = []
        for j in bad:
            p = L.label_parts(labels, i, j)
            votes.append(max({"contra": p["contra"],
                              "frag_left": 1 - p["adq_left"],
                              "frag_right": 1 - p["adq_right"]}.items(),
                             key=lambda kv: kv[1])[0])
        return collections.Counter(votes).most_common(1)[0][0]

    picked = [(w, r) for r in rows if (w := worst(r))]
    picked.sort(key=lambda x: -x[0][1])

    # **손실 크기로만 뽑으면 시끄러운 실패 하나만 계속 본다.** run18 train 실측:
    # 손실이 있는 93문장이 frag_right 42 / contra 28 / frag_left 23 으로 갈리는데,
    # 상위 8개는 contra 6 / frag_right 2 / **frag_left 0** 이었다. 조기 방출은 건당
    # 손실이 크고 파편은 작아서, 줄을 세우면 위쪽을 contra 가 독차지한다. 그래서
    # 전체의 4분의 1인 실패 종류를 Critic 이 **한 번도 못 본다.**
    #
    # 종류별로 통을 만들고 돌아가며 뽑는다. 통 안에서는 손실 순이므로 "각 종류에서 가장
    # 크게 진 것"이 먼저 온다. 한 종류가 모자라면 남은 자리는 다른 통이 가져간다.
    # **진 문장만 통에 넣는다.** 손실 0 은 절단기가 라벨과 같은 자리를 고른 문장이라
    # 고칠 것이 없다. 종전에는 손실 순 상위만 잘라 자연히 걸러졌는데, 돌아가며 뽑으면
    # 바닥에서 올라와 자리를 먹는다 (실측: n_cases 12 중 3칸이 loss 0.000 이었다).
    buckets: dict[str, list] = {}
    for (T_at, loss), r in picked:
        if loss <= 0:
            continue
        buckets.setdefault(failure_kind(r, T_at), []).append(((T_at, loss), r))
    order = sorted(buckets, key=lambda k: -len(buckets[k]))
    chosen: list = []
    while len(chosen) < n_cases and any(buckets[k] for k in order):
        for k in order:
            if buckets[k] and len(chosen) < n_cases:
                chosen.append((buckets[k].pop(0), k))

    cases = []
    for ((T_at, loss), r), kind_ in chosen:
        key = str(T_at)
        i = sent_index[r["id"]]
        u = units_by_id[r["id"]]
        bt = r["by_T"][key]
        km, ko = set(bt["kept_model"]), set(bt["kept_label"])
        cands = []
        for j, s_, y in zip(r["positions"], r["scores"], r["labels"]):
            parts = L.label_parts(labels, i, j)
            cands.append({
                "pos": j,
                "left": L.join_units(u[max(0, j - 3):j], spaced),
                "right": L.join_units(u[j:j + 3], spaced),
                "score": s_, "label": round(y * 100),
                "contra": parts["contra"], "adq_left": parts["adq_left"],
                "adq_right": parts["adq_right"],
                "kept_by_model": j in km, "kept_by_label": j in ko,
            })
        cases.append({"id": r["id"], "text": r["text"], "T": T_at,
                      "loss": round(loss, 4),
                      # 어느 통에서 왔는지 — Critic 이 종류별로 규칙을 나눠 쓸 수 있게.
                      "failure_kind": kind_,
                      "candidates": cands})
    return cases


def run_critic(gw: Gateway, prompt: str, m: dict, cases: list[dict],
               rejected: list[dict], effort: str | None,
               accepted: list[dict] | None = None) -> dict:
    user = (f"=== METRICS (train batch) ===\n"
            + json.dumps({"overlap": m["overlap"], "overlap_by_T": m["overlap_by_T"],
                           "achv": m["achv"], "spearman_within": m["spearman_within"],
                           "tie_rate": m["tie_rate"], "tie_at_cut_rate": m["tie_at_cut_rate"]},
                          ensure_ascii=False, indent=1)
            + ("\n\n=== CASES (each shown at the budget T where that sentence fell hardest, "
               "so T differs between cases). `failure_kind` says which component of the "
               "label the wrongly-kept boundaries broke: `contra` = the rest of the sentence "
               "overturns what was emitted, `frag_left` / `frag_right` = that side is a "
               "fragment that cannot be translated on its own. **Cases are drawn round-robin "
               "across those kinds, not by loss alone** — ranking by loss alone buries the "
               "fragment failures under the louder contra ones, so treat the kinds as equally "
               "important and produce rules for each kind you see. ===\n")
            + json.dumps(cases, ensure_ascii=False, indent=1))
    # 실패만 보여주면 비평이 "무엇이 이미 일하고 있는가"를 모른 채 그것까지 뜯자고 한다.
    if accepted:
        user += ("\n\n=== ACCEPTED DIRECTIONS (measured as better) ===\n"
                 + json.dumps(accepted, ensure_ascii=False, indent=1)
                 + "\nDo not propose undoing these — they are why the prompt scores what "
                   "it does. Find what they did NOT fix.")
    if rejected:
        user += ("\n\n=== REJECTED DIRECTIONS ===\n"
                 + json.dumps(rejected, ensure_ascii=False, indent=1))
    user += f"\n\n<prompt_under_review>\n{prompt}\n</prompt_under_review>"
    return gw.chat_json(ad.critic_system(), user, max_tokens=AGENT_MAX_TOKENS,
                        reasoning_effort=effort, purpose="critic")


def run_engineer(gw: Gateway, prompt: str, critique: dict, history: list[dict],
                 rejected: list[dict], size_budget: int, mode: str,
                 effort: str | None, output_rules: str, src_lang: str,
                 targets: list[str], accepted: list[dict] | None = None) -> dict | None:
    sys_p, user = ad.engineer_messages(prompt, critique, history, rejected, size_budget, mode,
                                       accepted=accepted)

    def limit_of(pr: str) -> str | None:
        """분량 제약 위반 문구. 안 어겼으면 None."""
        if mode == "shrink" and len(pr) >= len(prompt):
            return f"{len(pr)} >= {len(prompt)}"
        if mode == "neutral" and len(pr) > len(prompt):
            return f"{len(pr)} > {len(prompt)}"
        if len(pr) > size_budget:
            return f"예산 초과 {len(pr)} > {size_budget}"
        return None

    # **분량 위반은 한 번 되묻는다.** 종전에는 그냥 버렸는데, 그러면 `shrink`/`neutral`
    # 이 통째로 사라지고 탐색이 제한 없는 `grow` 하나로 줄어든다 — run17 iter 0 에서
    # 두 라운드 모두 그렇게 됐다 (shrink 13320/13520, neutral 13643/13104, 모두 기준
    # 12032 초과). 그 결과 r0·r1 이 사실상 같은 모드의 재추출이 되고 프롬프트만 +30%
    # 부푼다. 분절 호출에는 같은 성격의 복구 재시도가 이미 있다.
    #
    # 되묻는 값은 **잰 글자 수**다. 지시문을 다시 쓰지 않는다 — `engineer_messages` 가
    # 이미 제약을 담고 있고, 여기서 다시 쓰면 두 문구가 어긋날 때 고칠 수 없는 것을
    # 시키게 된다.
    for attempt in range(2):
        try:
            rv = gw.chat_json(sys_p, user, max_tokens=PROMPT_MAX_TOKENS,
                              reasoning_effort=effort, purpose="prompt_engineer")
        except Exception as e:  # noqa: BLE001
            log(f"[pe/{mode}] 실패: {e}")
            return None
        pr = (rv.get("prompt") or "").strip()
        if not pr:
            return None
        pr = ad.replace_section(pr, "[Output Rules]", output_rules)
        errs = ad.check_skeleton(pr) + agents.check_target_agnostic(pr, src_lang, targets)
        if errs:
            log(f"[pe/{mode}] 관문 실패: {errs}")
            return None
        bad = limit_of(pr)
        if not bad:
            rv["prompt"] = pr
            rv["candidate_kind"] = mode
            return rv
        if attempt == 0:
            log(f"[pe/{mode}] 분량 제약 위반 {bad} — 재시도")
            user += (f"\n\n=== LENGTH VIOLATION ===\nYour previous answer was {len(pr)} "
                     f"characters, which breaks the size constraint for mode "
                     f"'{mode}' ({bad}). Cut text until it fits. Remove whole rules that "
                     f"the measurements do not support rather than trimming wording "
                     f"everywhere — a shorter prompt with fewer, better rules is the point "
                     f"of this mode.")
    log(f"[pe/{mode}] 분량 제약 위반 {bad} — 재시도 후에도 초과, 버린다")
    return None


# ── main ────────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="fleurs-en-multi")
    p.add_argument("--src-lang", default="English")
    p.add_argument("--pair-id", required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--split-from", default=None,
                   help="이 런의 test/dev 를 그대로 물려받는다 (예: en2x/en-multi/run16). "
                        "매니페스트를 늘려 train 을 키우면서 평가 분할은 고정해 이전 "
                        "런과 비교를 유지할 때 쓴다. 새 문장은 전부 train 으로 간다")
    p.add_argument("--labels-from", default=None,
                   help="같은 분할의 oracle_labels_*.json 을 가진 런 (예: en2x/en-multi/run14)")
    p.add_argument("--model", default="gpt-5-mini")
    add_provider_args(p)
    p.add_argument("--seg-reasoning-effort", default="medium")
    p.add_argument("--agent-reasoning-effort", default="medium")
    p.add_argument("--batch-size", type=int, default=6)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--tgt-langs", nargs="+", default=None)
    p.add_argument("--local-mt-model", default="google/madlad400-3b-mt")
    p.add_argument("--adequacy-backend", default="cometkiwi")
    p.add_argument("--comet-batch-size", type=int, default=64)
    p.add_argument("--min-gap", type=int, default=None)
    p.add_argument("--units-per-sec", type=float, default=None)
    p.add_argument("--t-grid", type=int, nargs="+", default=None)
    p.add_argument("--final-t-grid", type=int, nargs="+", default=None)
    p.add_argument("--main-t", type=int, default=None)
    p.add_argument("--train", type=int, default=30)
    p.add_argument("--train-pool", type=int, default=None)
    p.add_argument("--dev", type=int, default=215)
    p.add_argument("--test", type=int, default=100)
    p.add_argument("--seed", type=int, default=data.DEFAULT_SEED)
    p.add_argument("--iterations", type=int, default=5)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--v0-candidates", type=int, default=3)
    p.add_argument("--revision-candidates", type=int, default=3)
    p.add_argument("--rule-min-matches", type=int, default=8,
                   help="제안 규칙이 train 전 경계 중 최소 몇 자리에 걸려야 하나. **판정은 "
                        "t 가 한다** — 표본이 작으면 se 가 커져 t 가 안 나오므로, 이 값은 "
                        "n=1~3 짜리 퇴화 사례만 막는 바닥이다. 20 으로 잡으면 실측으로 "
                        "유효한 규칙까지 죽는다 (숫자|단위 n=9 t=-2.85, 문말부호 뒤 "
                        "n=15 t=+3.65). LLM 호출 0. 0 = 관문 끔")
    p.add_argument("--rule-min-t", type=float, default=2.0,
                   help="걸린 경계의 라벨 평균이 나머지와 벌어져야 하는 표준오차 배수. "
                        "부호도 `direction` 과 맞아야 한다 — lower 면 라벨이 낮아야 한다")
    p.add_argument("--pool-size", type=int, default=2,
                   help="다음 이터의 출발점으로 남길 기각 후보 수 (홀드아웃 성적 상위). "
                        "첫 후보는 항상 현재 best 에서 출발하므로 실제로 쓰이는 것은 "
                        "min(pool-size, revision-candidates-1) 개다. 0 = 끔(항상 best)")
    p.add_argument("--revision-rounds", type=int, default=2,
                   help="한 이터레이션에서 개정 후보를 다시 뽑을 최대 라운드 수. 홀드아웃 "
                        "기준선(Δ>0)을 넘는 후보가 나오면 즉시 멈춘다. 끝까지 없으면 dev "
                        "평가를 건너뛰고 무개선으로 센다. 1 = 재추출 없음(관문만)")
    p.add_argument("--adopt-se-mult", type=float, default=1.0)
    p.add_argument("--objective", default="overlap", choices=["overlap", "achv"],
                   help="채택·후보 선별 기준. 기본 overlap — 라벨 최적 절단과 몇 %% 같은 자리를 "
                        "골랐나. achv 는 같은 k 로 아무렇게나 골라도 0.836 이 나와(run15 test 실측) "
                        "실사용 폭이 0.164 뿐이고, 같은 프롬프트 차이를 재도 t 가 절반이다")
    p.add_argument("--max-prompt-growth", type=float, default=1.6)
    p.add_argument("--n-cases", type=int, default=12,
                   help="Critic 에게 줄 실패 사례 수. **실패 종류별로 통을 만들어 돌아가며 "
                        "뽑는다** (contra / frag_left / frag_right) — 손실 순으로만 뽑으면 "
                        "조기 방출이 위를 독차지해 파편 실패를 못 본다. 통이 셋이므로 3의 "
                        "배수가 고르게 나뉜다. 분절 호출은 안 늘고 Critic 입력만 길어진다")
    p.add_argument("--budget", type=float, default=25.0)
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--skip-final-effective", action="store_true")
    args = p.parse_args()

    run_dir = RUNS_DIR / args.pair_id / args.run_id
    if args.fresh and run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    labels_from = (RUNS_DIR / args.labels_from) if args.labels_from else None

    # ── 데이터·프로파일·노브 (loop.py 와 같은 절차) ────────────────────
    sentences = data.load(args.dataset)
    args.train_pool = args.train_pool or 3 * args.train
    pool_n = max(args.train, args.train_pool)
    # **평가 분할을 기존 런에서 물려받는다** (`--split-from`). 층화가 층마다 셔플을
    # 하므로 매니페스트에 문장을 붙이면 test/dev 가 통째로 뒤바뀐다 (실측: +70 에
    # test 22/100 잔존). 근거 문장을 늘리려면 이 고정이 먼저다 — 붙인 문장은 전부
    # train 으로만 들어가고 test/dev 는 순서까지 그대로다.
    pinned = None
    if args.split_from:
        src = RUNS_DIR / args.split_from / "data"
        pinned = {name: [r["id"] for r in json.loads((src / f"{name}.json").read_text("utf-8"))]
                  for name in ("test", "dev")}
        log(f"[data] 평가 분할 고정: {args.split_from} 에서 "
            f"test {len(pinned['test'])} / dev {len(pinned['dev'])}")
    splits = data.split_data(sentences, pool_n, args.dev, args.test, seed=args.seed,
                             pinned=pinned)
    data.write_splits(splits, run_dir / "data")
    fit = splits["train"][:args.train] + splits["dev"]
    measured = data.measure_profile([x.text for x in fit])
    spaced, _punct = data.profile_settings(measured)
    (run_dir / "measured_profile.json").write_text(
        json.dumps(measured, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.min_gap is None:
        rate, src = data.units_per_sec(args.dataset, fit, spaced)
        if rate is None and args.units_per_sec:
            rate, src = args.units_per_sec, "cli"
        if rate is None:
            log("[stop] 발화 속도를 잴 수 없다 — --min-gap 또는 --units-per-sec 필요")
            return 2
        args.min_gap = derive_min_gap(rate)
        log(f"[min_gap] {rate:.2f}{measured['unit']}/초 × {MIN_GAP_MS}ms → {args.min_gap} ({src})")
    grid_loop, grid_final = derive_t_grids(args.min_gap)
    t_grid = args.t_grid or grid_loop
    final_grid = args.final_t_grid or grid_final
    main_T = args.main_t or sorted(t_grid)[len(t_grid) // 2]
    targets = resolve_targets(args.tgt_langs or DEFAULT_TARGET_POOL, args.src_lang)
    cfg = {**vars(args), "t_grid": t_grid, "final_t_grid": final_grid, "main_t": main_T,
           "targets": targets, "spaced": spaced, "mode": "distill",
           "translator_id": f"local:{args.local_mt_model}:{to_lang_code(targets[0])}:ctx=False",
           "adequacy_backend": args.adequacy_backend, "min_gap": args.min_gap,
           "pair_id": args.pair_id}
    (run_dir / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2),
                                         encoding="utf-8")
    log(f"[data] train {len(splits['train'])} (배치 {args.train} + 홀드아웃 "
        f"{len(splits['train']) - args.train}) / dev {len(splits['dev'])} / test "
        f"{len(splits['test'])} / T {t_grid} main {main_T} / min_gap {args.min_gap} / 타깃 {targets}")

    # ── 라벨 (LLM 0콜) ─────────────────────────────────────────────────
    adequacy = metrics.make_adequacy_backend(args.adequacy_backend,
                                             batch_size=args.comet_batch_size)
    contradiction = metrics.make_contradiction_backend()
    translators: dict[str, LocalTranslator] = {}
    lab: dict[str, dict] = {}
    for split in ("train", "dev", "test"):
        ss = splits[split]
        lab[split] = L.compute_labels(
            run_dir, split, [s.id for s in ss], [s.text for s in ss], targets, spaced,
            adequacy, contradiction, args.local_mt_model, target_is_spaced, log=log,
            translators=translators, reuse_from=labels_from)
    batch = splits["train"][:args.train]
    holdout = splits["train"][args.train:] or batch
    lab_batch = {t: per[:args.train] for t, per in lab["train"].items()}
    lab_hold = ({t: per[args.train:] for t, per in lab["train"].items()}
                if splits["train"][args.train:] else lab_batch)

    gw = Gateway.from_args(args, model=args.model, budget=args.budget,
                           reasoning_effort=args.agent_reasoning_effort)
    seg_cache = JsonCache(run_dir / "cache" / "segment.json")
    out_rules = ad.output_rules(spaced)
    seg_effort = None if args.seg_reasoning_effort == "none" else args.seg_reasoning_effort
    ag_effort = None if args.agent_reasoning_effort == "none" else args.agent_reasoning_effort

    def ev(prompt, sents, labels_):
        return evaluate(gw, prompt, sents, labels_, spaced, args.min_gap, t_grid, seg_cache,
                        args.workers, args.batch_size, seg_effort)

    history: list[dict] = []
    best: dict = {}
    try:
        # ── 프로파일 + v0 ─────────────────────────────────────────────
        prof_path = run_dir / "language_profile.json"
        if prof_path.exists():
            profile = json.loads(prof_path.read_text(encoding="utf-8"))
        else:
            profile = agents.Profiler(gw).profile([s.text for s in batch[:20]])
            prof_path.write_text(json.dumps(profile, ensure_ascii=False, indent=2),
                                 encoding="utf-8")
        facts = agents.measured_facts(measured)
        v0_path = run_dir / "prompt_v0.txt"
        if v0_path.exists():
            prompt_v0 = v0_path.read_text(encoding="utf-8")
        else:
            def write_one(k: int):
                user = (f"Language profile:\n{json.dumps(profile, ensure_ascii=False, indent=2)}\n\n"
                        + (facts + "\n\n" if facts else "")
                        + "The prompt must be TARGET-LANGUAGE-AGNOSTIC.\n\n"
                        + f"Copy this [Output Rules] section verbatim into the prompt:\n\n{out_rules}")
                pr = gw.chat(ad.writer_system(spaced, args.min_gap), user,
                             max_tokens=PROMPT_MAX_TOKENS, reasoning_effort=ag_effort,
                             purpose="prompt_v0").strip()
                pr = ad.replace_section(pr, "[Output Rules]", out_rules)
                errs = ad.check_skeleton(pr) + agents.check_target_agnostic(pr, args.src_lang, targets)
                (run_dir / f"prompt_v0_cand{k}.txt").write_text(pr, encoding="utf-8")
                return pr, errs
            with ThreadPoolExecutor(max_workers=args.v0_candidates) as ex:
                cands = list(ex.map(write_one, range(args.v0_candidates)))
            scored_c = []
            for k, (pr, errs) in enumerate(cands):
                if errs:
                    log(f"[v0] 후보 {k} 관문 실패 {errs}")
                    continue
                _, m = ev(pr, holdout, lab_hold)
                log(f"[v0] 후보 {k}: {len(pr)}자 holdout overlap={m['overlap']} achv={m['achv']} "
                    f"overlap={m['overlap_by_T']} fmt={m['format_pass_rate']} "
                    f"(1st {m['format_pass_rate_no_retry']}, 위반 {m['first_pass_violations']['counts']})")
                if m["first_pass_violations"]["samples"]:
                    log(f"[v0] 1차 위반 예: {m['first_pass_violations']['samples'][0]}")
                scored_c.append((m[args.objective] or 0.0, k, pr))
            if not scored_c:
                log("[stop] v0 후보가 전부 관문에 걸렸다")
                return 2
            scored_c.sort(key=lambda x: -x[0])
            prompt_v0 = scored_c[0][2]
            log(f"[v0] 후보 {scored_c[0][1]} 채택 ({args.objective}={scored_c[0][0]})")
            v0_path.write_text(prompt_v0, encoding="utf-8")
        size_budget = int(len(prompt_v0) * args.max_prompt_growth)

        # ── 이터레이션 ────────────────────────────────────────────────
        prompt = prompt_v0
        rejected: list[dict] = []
        # 실측으로 이긴 개정들 (A). `rejected` 와 짝이고 PE·Critic 이 둘 다 받는다.
        accepted: list[dict] = []
        # 기각됐지만 홀드아웃 성적이 좋았던 후보 (D). 기준선이 못 오를 때 여기서 다시
        # 출발한다 — 다음 이터가 항상 `best` 에서만 시작하면 골짜기를 못 넘는다.
        pool: list[dict] = []
        no_improve = 0
        for it in range(args.iterations):
            it_dir = run_dir / f"iter_{it:02d}"
            it_dir.mkdir(exist_ok=True)
            (it_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
            t0 = time.time()
            # **바뀌지 않은 프롬프트를 다시 재지 않는다.** 개정 관문이 후보를 하나도
            # 통과시키지 못하면 `prompt` 가 `best` 그대로 남는데, 그걸 다음 이터에서
            # 또 채점하면 관문이 아끼려던 dev 평가가 그대로 나간다 (게다가 분절이
            # 비결정론적이라 같은 프롬프트가 잡음만큼 다른 Δ 를 받아 채택·기각이
            # 동전던지기가 된다). 이미 있는 행을 그대로 쓰고 개정 단계로 넘어간다.
            reuse = bool(best) and prompt == best["prompt"]
            if reuse:
                log(f"[iter {it}] 프롬프트 불변 — 재채점 생략, 개정만 다시 시도")
                tr_rows, tr_m = best["train_rows"], best["train_m"]
                dv_rows, dv_m = best["dev_rows"], best["dev_m"]
            else:
                tr_rows, tr_m = ev(prompt, batch, lab_batch)
                dv_rows, dv_m = ev(prompt, splits["dev"], lab["dev"])
            (it_dir / "train_rows.json").write_text(json.dumps(tr_rows, ensure_ascii=False),
                                                    encoding="utf-8")
            (it_dir / "dev_rows.json").write_text(json.dumps(dv_rows, ensure_ascii=False),
                                                  encoding="utf-8")
            pd = None if reuse else (paired(dv_rows, best["dev_rows"], key=args.objective)
                                     if best else None)
            adopted = (not reuse) and ((not best)
                                       or pd["mean_delta"] > args.adopt_se_mult * pd["se_delta"])
            entry = {"version": it, "adopted": adopted, "train": tr_m, "dev": dv_m,
                     "score_train": tr_m[args.objective], "score_dev": dv_m[args.objective],
                     "paired_dev": pd, "changelog": best.get("next_changelog"),
                     "prompt_len": len(prompt), "usage": gw.usage.snapshot()}
            log(f"[iter {it}] train overlap={tr_m['overlap']} achv={tr_m['achv']} "
                f"tie={tr_m['tie_rate']}/{tr_m['tie_at_cut_rate']} sp={tr_m['spearman_within']} "
                f"fmt={tr_m['format_pass_rate']}(1st {tr_m['format_pass_rate_no_retry']}, "
                f"위반 {tr_m['first_pass_violations']['counts']}) | "
                f"dev overlap={dv_m['overlap']} {dv_m['overlap_by_T']} achv={dv_m['achv']} "
                f"fmt={dv_m['format_pass_rate']}(1st {dv_m['format_pass_rate_no_retry']}, "
                f"위반 {dv_m['first_pass_violations']['counts']}) "
                # 재사용 이터는 잰 것이 없으므로 '거부' 로 찍으면 안 된다 — 나중에
                # 로그만 보면 기각이 실제보다 많아 보인다.
                f"Δ={pd} | {'재사용' if reuse else ('채택' if adopted else '거부')} "
                f"| 비용 {gw.usage.snapshot()['cost']:.3f}")
            (it_dir / "metrics.json").write_text(json.dumps(entry, ensure_ascii=False, indent=1),
                                                 encoding="utf-8")
            if adopted:
                # ── 채택 귀속 ──────────────────────────────────────────
                # **무엇이 이겼는지 남긴다.** 거부 부검의 반대 방향이다. 종전에는 이긴
                # 개정도 changelog(자기 보고)와 Δ 숫자만 남아, 다음 이터의 PE 가
                # "무엇을 이어서 밀지"를 알 수 없었다. `regressions` 는 인자를 뒤집으면
                # 그대로 개선분을 준다 — 새 행이 옛 행을 이긴 문장을 낙폭 순으로
                # 돌려주므로 여기서는 **상승 순**이 된다. 새로 재는 것도 LLM 호출도 없다.
                if best.get("dev_rows"):
                    gains = moved_sentences(dv_rows, best["dev_rows"], t_grid, "overlap",
                                            gains=True)
                    accepted.append({
                        "version": it, "changelog": best.get("next_changelog"),
                        "sections_changed": best.get("next_sections"),
                        "dev_delta": pd,
                        "prompt_len": {"before": len(best["prompt"]), "after": len(prompt)},
                        # 이긴 개정이 실제로 살린 문장 — 어느 경계를 새로 잡았고 라벨이
                        # 원하던 자리가 어디였는지까지. PE 가 "이어서 밀" 근거다.
                        "gained_on": [{"id": g["id"], "text": g["text"],
                                       "budget_T": g["budget_T"], "gain": g["delta"],
                                       "kept_now": g["kept_now"],
                                       "kept_before": g["kept_before"],
                                       "kept_by_label": g["kept_by_label"]}
                                      for g in gains[:4]],
                    })
                    (it_dir / "accepted.json").write_text(
                        json.dumps(accepted[-1], ensure_ascii=False, indent=2), encoding="utf-8")
                    top = ", ".join(f"{g['id']} +{g['delta']:.3f}" for g in gains[:3])
                    log(f"[iter {it}] 채택 귀속 — 살린 문장 {len(gains)}개 ({top})")
                best = {"prompt": prompt, "version": it, "dev_rows": dv_rows, "dev_m": dv_m,
                        "train_rows": tr_rows, "train_m": tr_m}
                (run_dir / "best_prompt.txt").write_text(prompt, encoding="utf-8")
                no_improve = 0
            elif not reuse:
                rejected.append({"version": it, "changelog": best.get("next_changelog"),
                                 "dev_delta": pd})
                no_improve += 1
            # `reuse` 는 기각이 아니다 — 잰 것이 없다. 관문 단계에서 이미 무개선으로
            # 셌으므로 여기서 또 세면 patience 가 두 배로 닳는다.
            history.append(entry)
            (run_dir / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=1),
                                                  encoding="utf-8")
            if no_improve >= args.patience or it == args.iterations - 1:
                if no_improve >= args.patience:
                    log(f"[stop] dev 무개선 {no_improve}회")
                break

            # ── 거부 부검 ──────────────────────────────────────────────
            # **거부된 개정의 실패를 지금까지 아무도 안 봤다.** 아래 Critic 은 항상
            # `best` 의 train 실패만 본다 — 방금 진 개정본의 행은 `best` 에 들어간 적이
            # 없으므로 그 경로에서는 원리적으로 안 보인다. 그래서 거부 이력에 남는 것이
            # changelog 와 Δ 숫자뿐이었다: "무엇을 했다가 졌다"는 있고 **"어디서 어떻게
            # 졌다"가 없다.** run16 은 그렇게 9번 지고도 축적된 정보가 0이었다.
            #
            # `paired` 가 그 답을 이미 계산한다 — 문장별 Δ 를 다 구해 놓고 평균과 오차만
            # 남기고 버린다. `metrics.regressions` 가 같은 계산에서 낙폭 상위 문장을
            # 되돌려 준다. **새로 재는 것은 없다** (LLM 호출 0, 부검 1콜만 든다).
            #
            # `loop.py` 에 있던 것을 옮겼다. 증류 루프를 새로 쓰면서 빠져 있었다.
            #
            # 판정 직후가 아니라 여기서 부르는 이유: 조기 종료·마지막 이터는 위쪽
            # break 로 빠지므로 아무도 안 읽을 부검에 돈을 안 쓴다.
            if (not adopted and rejected and rejected[-1].get("version") == it
                    and best.get("dev_rows")):
                # `by_T` 에 들어 있는 축은 overlap 하나다 (achv 는 행 단위라 없다).
                regs = moved_sentences(dv_rows, best["dev_rows"], t_grid, "overlap")
                if regs:
                    try:
                        pm = agents.Critic(gw).diagnose_regression(
                            best["prompt"], prompt, regs,
                            changelog=rejected[-1].get("changelog"),
                            sections_changed=best.get("next_sections"),
                            delta=pd)
                    except BudgetExceeded:
                        raise
                    except Exception as e:  # noqa: BLE001
                        # **부검 실패로 루프를 끝내지 않는다.** 없어도 이터레이션은
                        # 그대로 돌아간다 — 다음 개정의 정보가 줄 뿐이다.
                        pm = None
                        log(f"[iter {it}] 거부 부검 실패 — 건너뛴다: {e}")
                    if pm:
                        # 거부 이력에 실어 둔다 — Critic 과 PE 둘 다 이 목록을 통째로
                        # 받으므로 별도 배선이 필요 없다.
                        rejected[-1].update({
                            k: pm[k] for k in
                            ("why_failed", "blamed_lines", "mechanism", "lesson")
                            if pm.get(k)})
                        (it_dir / "regression.json").write_text(
                            json.dumps({"regressions": regs, "post_mortem": pm},
                                       ensure_ascii=False, indent=2), encoding="utf-8")
                        log(f"[iter {it}] 거부 부검 {pm.get('mechanism')} "
                            f"| {(pm.get('why_failed') or '')[:90]}")

            # Critic — 항상 best 의 train 실패를 본다
            sent_index = {s.id: k for k, s in enumerate(batch)}
            units_by_id = {s.id: L.units_of(s.text, spaced) for s in batch}
            cases = build_cases(best["train_rows"], lab_batch, sent_index, units_by_id,
                                main_T, spaced, args.n_cases, t_grid)
            critique = run_critic(gw, best["prompt"], best["train_m"], cases, rejected,
                                  ag_effort, accepted=accepted)
            critique["metrics"] = {"overlap": best["train_m"]["overlap"],
                                   "overlap_by_T": best["train_m"]["overlap_by_T"],
                                   "achv": best["train_m"]["achv"],
                                   "tie_rate": best["train_m"]["tie_rate"],
                                   "spearman_within": best["train_m"]["spearman_within"]}
            (it_dir / "critique.json").write_text(json.dumps(critique, ensure_ascii=False, indent=1),
                                                  encoding="utf-8")
            (it_dir / "cases.json").write_text(json.dumps(cases, ensure_ascii=False, indent=1),
                                               encoding="utf-8")
            # 일반화 못 하는 규칙은 프롬프트에 못 들어간다 (LLM 호출 0). 판정 내역은
            # 남긴다 — 무엇이 왜 떨어졌는지 안 남기면 관문을 조정할 근거가 없다.
            if args.rule_min_matches > 0:
                n_before = len(critique.get("cases") or [])
                critique, verdicts = validate_rules(
                    critique, best["train_rows"], units_by_id,
                    args.rule_min_matches, args.rule_min_t)
                (it_dir / "rule_gate.json").write_text(
                    json.dumps(verdicts, ensure_ascii=False, indent=1), encoding="utf-8")
                kept = len(critique.get("cases") or [])
                drops = collections.Counter(v.get("reason") for v in verdicts
                                            if v["verdict"] == "drop")
                log(f"[iter {it}] 규칙 관문 {kept}/{n_before} 통과"
                    + (f" — 탈락 {dict(drops)}" if drops else ""))
            # **기준선을 못 넘은 후보에 dev 평가를 쓰지 않는다.** 종전에는 홀드아웃
            # **절대값** 최대만 골라 무조건 승격했다. 홀드아웃은 후보 여럿 중 최댓값을
            # 뽑는 자리라 위로 편향돼 있으므로(run16 실측: 승격 후보의 홀드아웃 Δ 가 dev
            # 보다 +0.063, +0.080 후했다), **거기서도 지는 후보는 dev 에서 확실히 진다.**
            # run16 은 3이터 중 2이터를 그런 후보에 썼고, 그 두 번이 patience 를 채워
            # 런을 끝냈다 (iter 0 후보 Δ -0.017/-0.062/-0.043 → dev -0.0800 기각,
            # iter 2 -0.051/-0.064/-0.047 → dev -0.1045 기각).
            #
            # 관문은 채택 시험이 아니라 **확정 패배 거르기**라 문턱이 Δ>0 이다. 못 넘으면
            # 후보를 다시 뽑는다 — 재추출 1라운드가 확정 기각될 dev 평가보다 싸다
            # (run16 단가: 후보 생성 ~$0.013/개, 홀드아웃 라운드 ~$0.5, dev 평가 ~$0.6
            # 에 patience 1칸). 라운드 상한이 없으면 비용이 안 막히므로 둘로 끊고,
            # 그래도 없으면 dev 를 건너뛰고 no_improve 만 올린다.
            base_rows, _bm = ev(best["prompt"], holdout, lab_hold)
            picks: list[tuple[float, float, dict]] = []   # (Δ, holdout objective, 후보)
            attempts = 0
            # **재추출은 앞 라운드가 진 것을 보고 뽑아야 한다.** 안 넘기면 r1 이 r0 와
            # 똑같은 입력으로 같은 분포에서 다시 뽑는 것이라 복권을 한 장 더 사는 것과
            # 같다 — 라운드를 늘린 값어치가 없다. PE 는 이미 `rejected` 를
            # "REJECTED DIRECTIONS (already measured as no better)" 로 받으므로
            # (`agents_distill.engineer_messages`), 이 라운드에서 잰 실패를 거기 얹는다.
            round_fails: list[dict] = []
            for rnd in range(args.revision_rounds):
                modes = [("grow", "neutral", "shrink")[k % 3]
                         for k in range(args.revision_candidates)]
                seen = rejected + round_fails
                # **출발점을 `best` 하나로 고정하지 않는다** (D). 이터마다 같은 자리에서
                # 다시 뽑으면 후보 분포가 정지 상태가 된다 — run17 실측으로 이터별
                # 최댓값이 0.393 -> 0.474 -> 0.439 -> 0.387 로 올랐다 내려왔고, 위로
                # 움직인 자리는 채택으로 기준선이 오른 한 번뿐이었다. 기각됐어도 홀드아웃
                # 성적이 좋았던 후보에서 다시 출발하면 골짜기를 넘어갈 길이 생긴다.
                #
                # 첫 후보는 항상 `best` 에서 출발한다 — 풀이 전부 나쁜 쪽으로 흘러가도
                # 현재 최선 주변 탐색이 끊기지 않게 하는 닻이다. 채점 기준선(`base_rows`)은
                # 출발점과 무관하게 언제나 `best` 다.
                bases = [best["prompt"]]
                for p_ in pool[:max(0, len(modes) - 1)]:
                    bases.append(p_["prompt"])
                while len(bases) < len(modes):
                    bases.append(best["prompt"])
                with ThreadPoolExecutor(max_workers=len(modes)) as ex:
                    rvs = list(ex.map(lambda t_: run_engineer(
                        gw, t_[1], critique, history, seen, size_budget, t_[0],
                        ag_effort, out_rules, args.src_lang, targets,
                        accepted=accepted), list(zip(modes, bases))))
                cands = [rv for rv in rvs if rv]
                attempts += len(cands)
                for rv in cands:
                    c_rows, hm = ev(rv["prompt"], holdout, lab_hold)
                    d = paired(c_rows, base_rows, key=args.objective)
                    delta = d["mean_delta"] if d["mean_delta"] is not None else 0.0
                    log(f"[iter {it} 개정 r{rnd}] {rv['candidate_kind']:8s} "
                        f"{len(rv['prompt'])}자 holdout overlap={hm['overlap']} "
                        f"achv={hm['achv']} Δ={d['mean_delta']}±{d['se_delta']}")
                    rv["holdout_delta"] = d
                    picks.append((delta, hm[args.objective] or 0.0, rv))
                    if delta <= 0:
                        # 다음 라운드가 같은 방향을 다시 시도하지 않게 근거와 함께 남긴다.
                        round_fails.append({
                            "version": f"{it}.r{rnd}", "candidate_kind": rv["candidate_kind"],
                            "changelog": rv.get("changelog"),
                            "sections_changed": rv.get("sections_changed"),
                            "gate": "holdout", "holdout_delta": d,
                        })
                if any(p[0] > 0 for p in picks):
                    break
                if rnd + 1 < args.revision_rounds:
                    log(f"[iter {it}] 기준선 넘은 후보 없음 — 재추출 (r{rnd + 1})")
            if not picks:
                log(f"[iter {it}] 개정 후보 전멸 — 중단")
                break
            # 승격 후보는 Δ 최대. 동점이면 홀드아웃 절대값으로 가른다.
            picks.sort(key=lambda x: (-x[0], -x[1]))
            passed = [p for p in picks if p[0] > 0]
            # 풀 갱신 (D) — 승격 못 한 후보 중 홀드아웃 성적 상위를 다음 이터의 출발점
            # 후보로 남긴다. 기준선(`best`)보다 나쁜 것만 들어오지만, 다음 이터의 출발점
            # 으로는 쓸모가 있다 — 다른 골짜기에 있을 수 있다. 프롬프트가 커지면 메모리도
            # 커지므로 `--pool-size` 로 끊는다.
            # 승격될 후보(`passed[0]`)는 넣지 않는다 — 채택되면 그게 `best` 가 되므로
            # 풀 한 칸이 best 복제로 낭비된다. 여기서 걸러야 한다: `best` 는 아직 옛
            # 것이라 `rv["prompt"] != best["prompt"]` 로는 안 잡힌다.
            promoted = passed[0][2]["prompt"] if passed else None
            for _d, obj, rv in picks:
                if rv["prompt"] not in (best["prompt"], promoted):
                    pool.append({"prompt": rv["prompt"], "kind": rv["candidate_kind"],
                                 "holdout_obj": obj, "from_iter": it})
            # 이미 `best` 가 된 옛 항목도 걷어낸다 (다음 이터에서 중복 출발점이 된다).
            pool[:] = [p_ for p_ in pool if p_["prompt"] != best["prompt"]]
            pool.sort(key=lambda p_: -p_["holdout_obj"])
            del pool[args.pool_size:]
            (it_dir / "changelog.json").write_text(json.dumps({
                "selected_kind": picks[0][2]["candidate_kind"] if passed else None,
                "gate": {"threshold": 0.0, "rounds": args.revision_rounds,
                         "candidates_tried": attempts, "n_passed": len(passed)},
                "changelog": picks[0][2].get("changelog") if passed else None,
                "sections_changed": picks[0][2].get("sections_changed") if passed else None,
                "candidates": [{"kind": rv["candidate_kind"], "len": len(rv["prompt"]),
                                "holdout_delta": rv["holdout_delta"], "holdout_obj": a}
                               for _d, a, rv in picks]},
                ensure_ascii=False, indent=1), encoding="utf-8")
            if not passed:
                # dev 평가를 건너뛴다. 기준선을 못 넘은 것이 이미 근거이므로 patience 는 쓴다.
                no_improve += 1
                best_try = picks[0]
                log(f"[iter {it}] 후보 {attempts}개 전부 기준선 이하 (최선 Δ={best_try[0]:+.4f}) "
                    f"— dev 평가 생략, 무개선 {no_improve}회")
                rejected.append({"version": it, "changelog": best_try[2].get("changelog"),
                                 "gate": "holdout", "holdout_delta": best_try[2]["holdout_delta"]})
                if no_improve >= args.patience:
                    log(f"[stop] dev 무개선 {no_improve}회")
                    break
                # `best` 로 되돌린다 — 다음 이터의 재사용 경로가 이걸 보고 재채점을
                # 건너뛴다. 안 되돌리면 진 개정본을 dev 에서 또 재게 된다.
                prompt = best["prompt"]
                continue
            chosen = picks[0][2]
            best["next_changelog"] = chosen.get("changelog")
            # 부검이 "자기 보고"가 아니라 diff 로 잰 변경 구역을 함께 보게 한다.
            best["next_sections"] = chosen.get("sections_changed")
            prompt = chosen["prompt"]
            (run_dir / "next_prompt.txt").write_text(prompt, encoding="utf-8")
            log(f"[iter {it}] 소요 {time.time() - t0:.0f}s")
    except BudgetExceeded as e:
        log(f"[stop] 예산 초과: {e}")
    finally:
        seg_cache.flush()

    if not best:
        log("[stop] 채택된 프롬프트가 없다")
        gw.close()
        return 1

    # ── 최종 test ─────────────────────────────────────────────────────
    test = splits["test"]
    te_rows, te_m = evaluate(gw, best["prompt"], test, lab["test"], spaced, args.min_gap,
                             final_grid, seg_cache, args.workers, args.batch_size, seg_effort)
    seg_cache.flush()
    log(f"[test] achv={te_m['achv']} overlap={te_m['overlap_by_T']} "
        f"sp={te_m['spearman_within']} fmt={te_m['format_pass_rate']}")
    units_t = [L.units_of(s.text, spaced) for s in test]
    by_id = {r["id"]: r for r in te_rows}
    pe_rows = []
    label_segs = []
    for i, s in enumerate(test):
        r = by_id[s.id]
        u = units_t[i]
        if r.get("scores"):
            seg = _seg_with(u, dict(zip(r["positions"], r["scores"])), spaced)
        else:
            seg = s.text
        c = r.get("positions") or []
        lseg = (_seg_with(u, {j: round(L.label_value(lab["test"], i, j) * SCALE) for j in c}, spaced)
                if c else s.text)
        label_segs.append(lseg)
        by_T = {}
        for T in final_grid:
            cut, miss = truncate(seg, T, spaced, args.min_gap)
            pieces = split_segments(cut) or [s.text]
            by_T[str(T)] = {"seg_text": cut, "k": len(pieces), "missing_boundaries": miss,
                            "pieces_src": pieces}
        pe_rows.append({"id": s.id, "text": s.text, "seg_text": seg, "valid": bool(r.get("valid")),
                        "by_T": by_T})
    (run_dir / "prompt_eval").mkdir(exist_ok=True)
    (run_dir / "prompt_eval" / "best_test.json").write_text(json.dumps({
        "prompt_file": "best_prompt.txt", "split": "test", "t_grid": final_grid,
        "min_gap": args.min_gap, "src_spaced": spaced, "tag_convention": "score",
        "rows": pe_rows}, ensure_ascii=False, indent=1), encoding="utf-8")
    (run_dir / "test_rows.json").write_text(json.dumps(te_rows, ensure_ascii=False),
                                            encoding="utf-8")

    report = {"test_label_metrics": te_m, "effective": {}}
    if not args.skip_final_effective:
        texts = [s.text for s in test]
        for name, segs in (("model", [r["seg_text"] for r in pe_rows]), ("oracle_label", label_segs)):
            report["effective"][name] = {}
            for T in final_grid:
                cuts = [truncate(sg, T, spaced, args.min_gap)[0] for sg in segs]
                per_t = {}
                for tgt in targets:
                    tr = translators.get(tgt)
                    if tr is None:
                        tr = LocalTranslator(
                            tgt_code=to_lang_code(tgt), model_id=args.local_mt_model,
                            cache=JsonCache(run_dir / "cache" / f"translate_{to_lang_code(tgt)}.json"))
                        translators[tgt] = tr
                    full = tr.full(texts)
                    sp = score_split(cuts, texts, full, tr, adequacy, spaced,
                                     target_is_spaced(tgt), contradiction)
                    per_t[tgt] = sp.effective
                eff = []
                for i in range(len(texts)):
                    v = [per_t[t][i] for t in targets if per_t[t][i] is not None]
                    eff.append(st.mean(v) if v else None)
                ok = [x for x in eff if x is not None]
                report["effective"][name][str(T)] = {
                    "mean": round(st.mean(ok), 4) if ok else None, "n": len(ok),
                    "by_tgt": {t: round(st.mean([x for x in per_t[t] if x is not None]), 4)
                               for t in targets}}
                log(f"[test/effective] {name:12s} T={T:<3d} {report['effective'][name][str(T)]['mean']}")
    report["usage"] = gw.usage.snapshot()
    report["history"] = [{k: h.get(k) for k in ("version", "adopted", "score_train", "score_dev",
                                                 "paired_dev", "prompt_len")} for h in history]
    report["best_version"] = best["version"]
    (run_dir / "final_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1),
                                               encoding="utf-8")
    lines = [f"# 증류 루프 결과 — {args.pair_id}/{args.run_id}", "",
             f"- 채택본 iter {best['version']} / 비용 {report['usage']['cost']:.3f} / 타깃 {targets}",
             f"- 채택 기준 **{args.objective}** — 라벨 최적 절단과 같은 자리를 고른 비율 "
             f"(같은 k 로 무작위 선택은 0.189, achv 는 0.836. run15 test 실측)",
             "", f"| iter | train {args.objective} | dev {args.objective} | dev Δ (쌍체) | 길이 | 채택 |",
             "|---|---|---|---|---|---|"]
    for h in history:
        pd = h.get("paired_dev")
        d_txt = "—" if not pd else f"{pd['mean_delta']:+.4f}±{pd['se_delta']:.4f}"
        lines.append(f"| {h['version']} | {h['score_train']} | {h['score_dev']} | {d_txt} | "
                     f"{h['prompt_len']} | {'O' if h['adopted'] else 'X'} |")
    lines += ["", f"test: achv {te_m['achv']} / overlap {te_m['overlap_by_T']} / "
              f"spearman_within {te_m['spearman_within']} / fmt {te_m['format_pass_rate']}", ""]
    if report["effective"]:
        lines += ["| 정책 | " + " | ".join(f"T={T}" for T in final_grid) + " |",
                  "|---" * (len(final_grid) + 1) + "|"]
        for name, d in report["effective"].items():
            lines.append(f"| {name} | " + " | ".join(f"{d[str(T)]['mean']}" for T in final_grid) + " |")
    (run_dir / "final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    gw.close()
    for tr in translators.values():
        tr.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
