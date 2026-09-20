"""판단형 프롬프트 루프 — 설계와 근거는 [LOOP_JUDGE.md](LOOP_JUDGE.md).

`loop_distill` 과 무엇이 다른가:

| | loop_distill | loop_judge |
|---|---|---|
| 사례 | 라벨과 순위가 어긋난 경계 | **절단집합끼리의 반사실 쌍** (정책 vs 오라클) |
| Critic 출력 | 토큰 조건 + 규칙 | 판단 기준 한 줄 또는 예시 |
| 개정 위치 | `[Scoring Rules]` | `[Core Principles]` / `[Examples]` |
| 관문 | Δoverlap > 1 se, 토큰 t 검정 | `H_set` 쌍체 부트스트랩 CI 하한 |
| 짝 | 문장 | (문장, T) |

분할·라벨·캐시는 기존 런에서 물려받는다 (`--from-run`). 새로 나누면 문장이 달라져
지금까지의 측정과 비교가 끊긴다.

    PYTHONPATH=. .venv/bin/python -m core.meaning_segmentator.autoseg.loop_judge \
        --from-run en2x/en-multi/run25 --run-id en2x/en-multi/judge01 \
        --prompt core/meaning_segmentator/tools/autoseg_en2x/run24/probe/v0_coh.txt \
        --provider openai --model gpt-5-mini --iterations 5 --budget 25
"""

from __future__ import annotations

import argparse
import hashlib
import copy
import threading
from concurrent.futures import ThreadPoolExecutor
import json
import random
import statistics as st
import time
from pathlib import Path

from .infra.gateway import Gateway, add_provider_args
from .loop import target_is_spaced
from .loop_distill import evaluate
from .paths import RUNS_DIR
from .runtime import agents, agents_distill as ad, agents_judge as aj
from .runtime import data, hset, labels as L, metrics
from .runtime.pipeline import JsonCache, LocalTranslator, to_lang_code

AGENT_MAX_TOKENS = 24000
SHORTEN_PASSES = 3      # rewrite 초안이 길이만 넘을 때 PE 축소 패스 최대 횟수


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ── 절단집합 ────────────────────────────────────────────────────────────

def _sets_from_scores(sents: list, score_of, spaced: bool, min_gap: int, min_chunk: int,
                      max_k: int) -> dict[tuple[int, int], tuple[int, ...]]:
    """`{(문장 index, k): 절단집합}` — k 를 1부터 훑는다.

    **T 격자 대신 k 축을 쓴다.** T 는 문장 길이에서 k 를 정하는 함수라, 격자를 몇 점 고르면
    그 깊이의 순위만 채점된다. k 를 훑으면 순위 전체가 채점되고 LLM 비용은 안 는다 —
    점수 벡터는 문장당 한 번만 뽑기 때문이다. 지연축은 보고할 때 `chunk_len(n, k)` 로
    되돌린다.
    """
    out = {}
    for i, s in enumerate(sents):
        u = L.units_of(s.text, spaced)
        sc = score_of(i, s, u)
        if not sc:
            continue
        for k in hset.k_range(len(u), min_chunk, max_k):
            c = hset.top_k_cuts(u, sc, k, min_gap)
            if len(c) == k:
                out[(i, k)] = c
    return out


def policy_sets(rows: list[dict], sents: list, spaced: bool, min_gap: int,
                min_chunk: int = 2, max_k: int = 10) -> dict[tuple[int, int], tuple[int, ...]]:
    """채점 행 → 절단집합. 점수가 없는 문장은 건너뛴다."""
    by_id = {r["id"]: r for r in rows if r.get("scores")}

    def score_of(i, s, u):
        r = by_id.get(s.id)
        return {j: float(x) / 100.0 for j, x in zip(r["positions"], r["scores"])} if r else {}

    return _sets_from_scores(sents, score_of, spaced, min_gap, min_chunk, max_k)


def oracle_sets(lab: dict, sents: list, spaced: bool, min_gap: int,
                min_chunk: int = 2, max_k: int = 10) -> dict[tuple[int, int], tuple[int, ...]]:
    """경계별 라벨 상위 k개 — 지금 오라클이 내는 답(그리디)."""
    def score_of(i, s, u):
        return {j: L.label_value(lab, i, j) for j in range(min_gap, len(u) - min_gap + 1)}

    return _sets_from_scores(sents, score_of, spaced, min_gap, min_chunk, max_k)


def search_sets(path: Path, t_grid: list[int], limit: int) -> tuple[dict, dict]:
    """집합 탐색 산출물(`set_scores_<split>.json`)에서 T별 최적 절단집합과 그 `H_set`.

    탐색이 이미 같은 공식으로 재 뒀으므로 값을 그대로 쓴다 — 다시 번역·채점하지 않는다.
    문장 순서는 탐색을 돌린 분할과 같아야 한다 (`--split-from` 으로 고정된 dev 앞에서부터).
    """
    blob = json.loads(path.read_text(encoding="utf-8"))
    sets, vals = {}, {}
    for i_s, per in blob.items():
        i = int(i_s)
        if i >= limit:
            continue
        for T_s, d in per.items():
            T = int(T_s)
            if T not in t_grid or not d.get("sets"):
                continue
            key, v = max(d["sets"].items(), key=lambda kv: kv[1])
            sets[(i, T)] = tuple(int(x) for x in key.split(","))
            vals[(i, T)] = v
    return sets, vals


# ── 사례 ────────────────────────────────────────────────────────────────

LATENCY_BINS = (3, 5, 7, 10, 99)
CONTRA_KILL = 0.5


def latency_bin(n_units: int, k: int, bins=LATENCY_BINS) -> str:
    c = hset.chunk_len(n_units, k)
    return next((f"≤{b}" for b in bins if c <= b), f">{bins[-2]}")


def loss_by_bin(gaps: list[tuple[float, tuple[int, int]]], bin_of) -> dict[str, dict]:
    """구간별 손실 — 짝 수, 평균 격차(오라클 − 정책), 양의 격차 합의 몫(`loss_share`).

    test-A 실측(judge12 v0): 손실의 61% 가 ≤3 구간, 26% 가 ≤5 구간이었다. 기각된 개정 10건 중
    9건이 그 두 구간을 떨어뜨렸다 — "어디서 자르지 말라" 는 원칙은 절단을 성기게 해 촘촘한
    절단이 필요한 구간에서 손해다. Critic·PE 가 이 표를 보고 손실이 있는 곳을 겨누게 한다."""
    by: dict[str, list[float]] = {}
    for gap, key in gaps:
        by.setdefault(bin_of(key), []).append(gap)
    pos_total = sum(g for v in by.values() for g in v if g > 0) or 1.0
    return {b: {"pairs": len(v), "gap_mean": round(st.mean(v), 4),
                "loss_share": round(sum(g for g in v if g > 0) / pos_total, 3)}
            for b, v in sorted(by.items(), key=lambda kv: float(kv[0].strip("≤>")))}


def case_quota(lb: dict[str, dict], n_cases: int) -> dict[str, int]:
    """구간별 사례 수 — 손실 몫에 비례, 구간마다 최소 1."""
    bins = list(lb)
    q = {b: max(1, round(n_cases * lb[b]["loss_share"])) for b in bins}
    order = sorted(bins, key=lambda b: -lb[b]["loss_share"])
    while sum(q.values()) > n_cases:
        b = max((b for b in bins if q[b] > 1), key=lambda b: q[b] - n_cases * lb[b]["loss_share"])
        q[b] -= 1
    while sum(q.values()) < n_cases:
        q[order[0]] += 1
    return q


def pick_cases(gaps: list[tuple[float, tuple[int, int]]], bin_of, n_cases: int,
               exclude: set[int] = frozenset(), quota: dict[str, int] | None = None) -> list:
    """손해가 난 (문장, k) 를 지연 구간별로 고르게, 문장당 하나씩 고른다.

    손해 상위만 뽑으면 한 구간에 몰린다 — judge05 는 12개가 전부 평균 조각 2~3.5어절에 손해
    0.67~0.80 이었고, 모순 절단 하나로 집합이 0 이 된 경우뿐이었다. Critic 은 그걸 "뒤집힐 수
    있는 자리를 더 피하라" 로 일반화했고, 넣을 때마다 짧은 조각 구간이 떨어졌다(iter 2~5 전부
    기각). 구간마다 손해 큰 순으로 줄 세워 한 바퀴씩 돌며 뽑는다.

    `exclude` 는 직전 이터에 보인 문장 인덱스. 채택이 없으면 프롬프트도 손해 순위도 그대로라
    같은 문장이 또 뽑히고(judge10 iter 1·2 는 12개가 동일), Critic 은 같은 finding 을 되풀이한다.
    `quota` 를 주면 구간마다 그 수까지만 뽑는다(`case_quota`, 손실 몫 비례)."""
    by_bin: dict[str, list] = {}
    for gap, key in sorted(gaps, reverse=True):
        if gap > 0:
            by_bin.setdefault(bin_of(key), []).append((gap, key))
    names = sorted(by_bin, key=lambda b: float(b.strip("≤>")))
    queues = [by_bin[b] for b in names]
    taken = {b: 0 for b in names}
    picked, seen = [], set(exclude)
    while len(picked) < n_cases and any(queues):
        moved = False
        for b, q in zip(names, queues):
            if quota is not None and taken[b] >= quota.get(b, 0):
                continue
            while q and q[0][1][0] in seen:
                q.pop(0)
            if q and len(picked) < n_cases:
                gap, key = q.pop(0)
                picked.append((gap, key))
                seen.add(key[0])
                taken[b] += 1
                moved = True
        if not moved:
            break
    return picked


def rank_depth(rows: list[dict], max_d: int = 13) -> dict[str, float]:
    """순위 깊이 d 에서 프롬프트 상위 d 개가 라벨 상위 d 개와 겹치는 비율 / 무작위 기대.

    **H_set Δ 하나로는 개정이 순위의 어디를 고쳤는지 안 보인다.** 긴 지연은 1위만 쓰고 짧은
    지연은 8위까지 쓴다(≤3 의 평균 절단 수 8.5) — 그래서 위쪽만 고친 개정도 전체 평균은
    올린다. 실측 기준선(judge20 v0-A, dev 600): 1위 6.5배, 5위 1.6배, 8위 1.2배, 13위 1.00배.
    """
    acc: dict[int, list[float]] = {}
    for r in rows:
        sc, lb, pos = r.get("scores"), r.get("labels"), r.get("positions")
        if not sc or not lb or not pos or len(sc) != len(lb):
            continue
        n = len(pos)
        p_order = sorted(range(n), key=lambda i: -sc[i])
        l_order = sorted(range(n), key=lambda i: -lb[i])
        for d in range(1, min(max_d, n) + 1):
            hit = len(set(p_order[:d]) & set(l_order[:d])) / d
            acc.setdefault(d, []).append(hit / (d / n))      # 무작위 기대 대비 배수
    return {str(d): round(st.mean(v), 3) for d, v in sorted(acc.items())}


def misorder_cost(rows: list[dict], max_d: int = 13) -> dict[str, float]:
    """깊이 d 에서 **한 번 잘못 고를 때 잃는 양** — d번째로 좋은 자리의 라벨에서 그 시점에 아직
    순위가 안 정해진 자리들의 평균 라벨을 뺀 것. 라벨만 쓰므로 비용 0.

    `rank_depth` 는 "깊이 d 의 순위가 정보를 갖는가" 를 말하고 이 값은 "거기서 틀리면 얼마나
    비싼가" 를 말한다. 둘이 같이 있어야 판단이 된다 — 깊은 곳이 무작위여도 손해가 0 이면 고칠
    값이 없고, 손해가 얕은 곳과 같으면 반드시 고쳐야 한다. Critic 지시문은 이 값을 **읽는 법**만
    말하고 수치는 여기서 매 런 다시 계산된다(다른 코퍼스로 옮기면 값이 따라 바뀐다).
    """
    acc: dict[int, list[float]] = {}
    for r in rows:
        lb = r.get("labels")
        if not lb or len(lb) < 3:
            continue
        v = sorted(lb, reverse=True)
        for d in range(1, min(max_d, len(v) - 1) + 1):
            tail = v[d - 1:]
            if len(tail) < 2:
                continue
            acc.setdefault(d, []).append(v[d - 1] - st.mean(tail))
    return {str(d): round(st.mean(x), 4) for d, x in sorted(acc.items())}


def rejected_by_bin(history: list[dict], n_max: int = 6) -> list[dict]:
    """**이 런에서** 기각된 개정들의 구간별 Δ. 같은 방향을 말만 바꿔 되풀이하는 것을 막는다.

    부검이 붙은 항목(이터 대표 후보)만 담으므로 기각 이터당 한 줄이다. 런을 가로질러 가져오지
    않는다 — 다른 런의 기각은 데이터도 프롬프트도 달라 이 런의 근거가 못 된다.
    """
    out = []
    for h in history:
        if h.get("adopted"):
            continue
        by = (h.get("diagnosis") or {}).get("by_bin")
        if not by:
            continue
        out.append({"iter": h["iter"], "edits": h.get("edits"), "by_bin": by})
    return out[-n_max:]


def rank_inversions(rows: list[dict], sents: list, spaced: bool, n_max: int = 15,
                    band: tuple[int, int] = (4, 10), min_gap_label: float = 0.15) -> list[dict]:
    """깊은 구간의 **역전 쌍** — 프롬프트는 A 를 B 보다 높게 봤는데 라벨은 B 가 훨씬 낫다.

    Critic 에게 "8위부터 무작위" 라는 집계값만 주면 "순위 깊이를 개선하라" 같은 공허한 finding
    이 나온다. 무엇을 쓸지는 **두 자리를 가르는 표면 특징**에서 나오므로 쌍을 직접 보여준다.
    `fallback` 역할(자리끼리의 서열을 적는 역할)이 쓸 재료가 이것이다.

    라벨 순위가 `band` 안인 자리들만 본다 — 1~3위는 이미 무작위 대비 2.4~6.5배로 맞는다.
    """
    out = []
    for r in rows:
        sc, lb, pos = r.get("scores"), r.get("labels"), r.get("positions")
        if not sc or not lb or not pos or len(sc) != len(lb):
            continue
        i = next((k for k, s in enumerate(sents) if s.id == r["id"]), None)
        if i is None:
            continue
        u = L.units_of(sents[i].text, spaced)
        l_order = sorted(range(len(pos)), key=lambda k: -lb[k])
        lo, hi = band
        deep = [k for rank, k in enumerate(l_order, 1) if lo <= rank <= hi]
        best = None
        for a_ in deep:
            for b_ in deep:
                if sc[a_] <= sc[b_] or lb[b_] - lb[a_] < min_gap_label:
                    continue
                score = (lb[b_] - lb[a_]) * (sc[a_] - sc[b_])
                if best is None or score > best[0]:
                    best = (score, a_, b_)
        if best is None:
            continue
        _s, a_, b_ = best
        def ctx(k):
            j = pos[k]
            sep = " " if spaced else ""
            return (sep.join(u[max(0, j - 3):j]) + "  ||  " + sep.join(u[j:j + 3]))
        out.append({"id": r["id"],
                    "prompt_prefers": {"at": ctx(a_), "prompt_score": sc[a_], "label": lb[a_]},
                    "label_prefers": {"at": ctx(b_), "prompt_score": sc[b_], "label": lb[b_]},
                    "label_gap": round(lb[b_] - lb[a_], 3)})
    out.sort(key=lambda d: -d["label_gap"])
    return out[:n_max]


def build_cases(sents: list, lab: dict, pol: dict, ora: dict, pol_h: dict, ora_h: dict,
                spaced: bool, min_gap: int, n_cases: int, pieces_tr=None,
                exclude_ids: set[str] = frozenset(), alloc: str = "uniform", *,
                exclude_keys: set[tuple[str, str]] = frozenset(),
                shown_cuts: dict[str, list] | None = None,
                jaccard_max: float = 0.5) -> tuple[list[dict], dict]:
    """손해가 난 (문장, k) 를 지연 구간별로 골라(`pick_cases`) 반사실 쌍으로 만든다 —
    (사례, 구간별 손실표). `exclude_ids` 의 문장(앞 이터 사례)은 빼고 뽑는다.
    `alloc="loss"` 면 구간별 사례 수를 손실 몫에 비례시킨다(`case_quota`), 아니면 균등.

    **문장 통째로 빼면 풀이 금방 마른다** — train 200 에 사례 40 이면 연속 기각 5번에 바닥이다.
    `exclude_keys`(문장 id, 지연 구간)로 빼면 같은 문장이라도 다른 구간은 남는다. 20어절 문장은
    ≤3 예산에서 절단이 6개, ≤10 에서 1~2개라 오라클이 고르는 자리도 달라 **다른 실패**다.
    구간만 다르고 실질이 같은 경우는 `shown_cuts`(앞서 보여준 절단집합)와의 Jaccard 가
    `jaccard_max` 이상이면 건너뛰어 거른다 — 절단집합은 이미 있으니 계산 비용이 0 이다.
    구간별 손실표(`lb`)는 **거르기 전** 전체 격차로 낸다. 진단값이지 뽑기 대상이 아니다."""
    gaps = [(ora_h[k] - pol_h[k], k) for k in pol if k in ora_h and k in pol_h]
    n_units = {i: len(L.units_of(sents[i].text, spaced)) for i in {k[0] for _g, k in gaps}}
    bin_of = lambda key: latency_bin(n_units[key[0]], key[1])
    lb = loss_by_bin(gaps, bin_of)
    quota = case_quota(lb, n_cases) if alloc == "loss" else None
    excl = {i for i, s in enumerate(sents) if s.id in exclude_ids}
    if exclude_keys or shown_cuts:
        def fresh(key) -> bool:
            sid = sents[key[0]].id
            if (sid, bin_of(key)) in exclude_keys:
                return False
            cut = set(pol.get(key, ()))
            for prev in (shown_cuts or {}).get(sid, ()):
                p = set(prev)
                if cut and p and len(cut & p) / len(cut | p) >= jaccard_max:
                    return False
            return True
        gaps = [(g, k) for g, k in gaps if fresh(k)]
    cases = []
    for gap, (i, kk) in pick_cases(gaps, bin_of, n_cases, exclude=excl, quota=quota):
        s = sents[i]
        u = L.units_of(s.text, spaced)
        cand = list(range(min_gap, len(u) - min_gap + 1))
        hb = {j: L.label_value(lab, i, j) for j in cand}
        order = sorted(cand, key=lambda j: hb[j])
        pct = {j: round(100.0 * order.index(j) / max(1, len(cand) - 1), 1) for j in cand}

        def detail(j):
            d = {"pos": j,
                 "left": " ".join(u[max(0, j - 4):j]) if spaced else "".join(u[max(0, j - 4):j]),
                 "right": " ".join(u[j:j + 4]) if spaced else "".join(u[j:j + 4]),
                 "H": round(hb.get(j, 0.0), 4), "H_pct": pct.get(j, 0.0),
                 "contra": round(st.mean(per[i]["contra"][j - 1] for per in lab.values()), 4),
                 "cohesion": round(st.mean(per[i]["adq_l"][j - 1] for per in lab.values()), 4)}
            return d

        dropped = [detail(j) for j in pol[(i, kk)] if j not in ora[(i, kk)]]
        added = [detail(j) for j in ora[(i, kk)] if j not in pol[(i, kk)]]

        def worst(cut):
            return max((st.mean(per[i]["contra"][j - 1] for per in lab.values()) for j in cut),
                       default=0.0)

        pw, tw = worst(pol[(i, kk)]), worst(ora[(i, kk)])
        case = {"id": s.id, "cuts": kk, "avg_chunk": round(hset.chunk_len(len(u), kk), 1),
                "latency_bin": latency_bin(len(u), kk),
                "gap": round(gap, 4),
                "policy_worst_contra": round(pw, 4),
                "contra_kill": pw >= CONTRA_KILL > tw,
                "policy": {"cut": list(pol[(i, kk)]), "H_set": round(pol_h[(i, kk)], 4),
                           "text": _marked(u, pol[(i, kk)], spaced)},
                "target": {"cut": list(ora[(i, kk)]), "H_set": round(ora_h[(i, kk)], 4),
                           "text": _marked(u, ora[(i, kk)], spaced)},
                "diff": {"dropped": dropped, "added": added}}
        if pieces_tr:
            case["policy"]["pieces"] = pieces_tr(i, pol[(i, kk)])
            case["target"]["pieces"] = pieces_tr(i, ora[(i, kk)])
        cases.append(case)
    return cases, lb


def labeled_examples(sents: list, lab: dict, cases: list[dict], spaced: bool,
                     min_gap: int) -> dict[str, str]:
    """사례 문장을 실측 라벨로 채운 Input/Output 예시로 만든다 — PE 가 원칙 대신 붙일 수 있게.

    출력 숫자는 라벨의 순위 백분위(0~100)다. 분절기는 숫자의 순서만 쓰므로 절대값은 뜻이 없고,
    v0 예시도 같은 꼴이다. judge03~08 에서 PE 는 예시(E)를 한 번도 고치지 않고 원칙(C)만 늘렸다."""
    by_id = {s.id: i for i, s in enumerate(sents)}
    out = {}
    for c in cases:
        i = by_id.get(c["id"])
        if i is None:
            continue
        u = L.units_of(sents[i].text, spaced)
        cand = ad.candidate_positions(len(u), min_gap)
        if not cand:
            continue
        hb = {j: L.label_value(lab, i, j) for j in cand}
        order = sorted(cand, key=lambda j: hb[j])
        pct = {j: round(100.0 * order.index(j) / max(1, len(cand) - 1)) for j in cand}
        scored = u[0]
        for j in range(1, len(u)):
            scored += (f" <SEG:{pct[j]}> {u[j]}" if j in pct else (" " if spaced else "") + u[j])
        out[c["id"]] = f"Input: {ad.mark_candidates(u, cand)}\nOutput: {scored}"
    return out


def pick_example_ids(sents: list, spaced: bool, n: int = 4) -> list[str]:
    """v0 [Examples] 에 넣을 문장 — 길이 순으로 줄 세워 분위수 자리에서 n 개. 결정론적이다.

    Writer 가 지어낸 점수(judge10 v0 의 예시 셋)는 근거가 없다. 사례 집합(train)의 실측 라벨로
    채운 쌍을 넣는다 — 판정은 test-A/B 에서 하므로 유출이 아니다."""
    ranked = sorted(sents, key=lambda s: (len(L.units_of(s.text, spaced)), s.id))
    if len(ranked) <= n:
        return [s.id for s in ranked]
    return [ranked[int((k + 0.5) * len(ranked) / n)].id for k in range(n)]


def examples_section(examples: dict[str, str]) -> str:
    """실측 예시 쌍으로 [Examples] 섹션 본문을 만든다 — `replace_section` 에 그대로 준다."""
    return "[Examples]\n" + "\n\n".join(examples.values()) + "\n"


def example_sentences(prompt: str, sents: list) -> set[int]:
    """프롬프트 [Examples] 에 들어간 문장이 `sents` 에 있으면 그 인덱스 — 채택 판정에서 뺀다.

    실측 예시(`labeled_examples`)는 dev-A 사례 문장으로 만든다. 그 문장이 라벨과 함께 프롬프트에
    들어간 채 dev-A 에서 채점되면 그 문장만 정답을 보고 푸는 셈이다. judge09 iter 2 채택은 dev-A
    Δ +0.0105 였는데 상위 5짝이 전부 예시로 넣은 문장(en_us_1591·1102)이었다. 예시 문장은 Input
    줄에서 `<SEG:?>` 를 걷어 낸 본문으로 알아본다 — 손으로 쓴 v0 예시는 분할 밖 문장이라 안 걸린다."""
    texts = {" ".join(s.text.split()): i for i, s in enumerate(sents)}
    out = set()
    for u in aj.edit_units(prompt):
        if u["section"] != "[Examples]":
            continue
        first = u["text"].split("\n", 1)[0]
        if not first.startswith("Input:"):
            continue
        body = " ".join(first[len("Input:"):].replace("<SEG:?>", " ").split())
        if body in texts:
            out.add(texts[body])
    return out


def _marked(units: list[str], cut: tuple[int, ...], spaced: bool) -> str:
    out = units[0]
    for j in range(1, len(units)):
        sep = " ‖ " if j in cut else (" " if spaced else "")
        out += sep + units[j]
    return out


def by_latency(sents: list, h: dict, spaced: bool,
               bins=LATENCY_BINS) -> dict[str, float]:
    """(문장, k) 값들을 평균 조각 길이로 묶어 지연축 곡선으로 — 보고용.

    판정은 (문장, k) 짝으로 하고 문장 단위로 재추출한다. 이쪽은 사람이 읽을 표다.
    """
    acc: dict[str, list[float]] = {}
    for (i, k), v in h.items():
        lab = latency_bin(len(L.units_of(sents[i].text, spaced)), k, bins)
        acc.setdefault(lab, []).append(v)
    return {k: round(st.mean(v), 4) for k, v in sorted(acc.items())}


# ── 채택 판정 ───────────────────────────────────────────────────────────

def screen_indices(n: int, size: int, it: int, seed: int = 20260915) -> list[int]:
    """이터 `it` 의 선별 문장 인덱스 — 시드 고정이라 재개해도 같은 문장이다."""
    import random
    idx = list(range(n))
    random.Random(seed + it).shuffle(idx)
    return sorted(idx[:min(size, n)])


def promising(boot: dict, min_mean: float = 0.005, min_lo: float = -0.01) -> bool:
    """dev-A 에서 기각됐지만 평균이 양수이고 하한이 크게 음수는 아닌 개정 — 문장을 더 재면 판가름
    날 수 있다. judge09 iter 1: +0.0105 [-0.0045, +0.0259] 가 하한 때문에 기각됐다. dev-A 150문장의
    쌍체 CI 반폭이 약 0.015 라 +0.01 짜리 진짜 이득은 원래 통과 못 한다."""
    return boot["mean"] > min_mean and boot["lo"] > min_lo


def near_miss_base(history: list[dict], run_dir: Path) -> tuple[str, dict] | None:
    """마지막 채택 뒤의 근소 기각본 중 하한이 가장 높은 것 — 다음 이터의 편집 기반.

    judge13 iter 1: 명사구 내부 절단을 허용한 편집이 test-A +0.006, test-B +0.008, 합산 +0.0074
    [-0.0003, +0.0153] 로 하한 0.0003 차 기각. 텍스트로 "그 방향을 유지하라" 고 알려도 iter 2 후보
    셋은 전부 다른 "자르지 마라" 규칙으로 갔다 — Critic 은 기각본 문구를 replace 대상으로 찍었는데
    현재 프롬프트에 없어 버려졌고, `single_small` 은 편집 하나라 재적용+확장이 불가능했다. 그래서
    근소 기각본 자체를 편집 기반으로 준다. 판정은 그대로 현재 best 대비 같은 문턱이다."""
    last_adopt = max([h["iter"] for h in history if h.get("adopted")], default=0)
    near = [h for h in history if h["iter"] > last_adopt and aj.is_near_miss(h)]
    if not near:
        return None
    h = max(near, key=lambda x: ((x.get("gain") or x.get("delta"))["lo"], x["iter"]))
    idir = run_dir / f"iter_{h['iter']:02d}"
    p = idir / (f"candidate_{h['candidate']}.txt" if h.get("candidate") is not None else "prompt.txt")
    if not p.exists():
        return None
    return p.read_text(encoding="utf-8"), h


def bin_deltas(sents: list, old_h: dict, new_h: dict, keys: list, spaced: bool) -> dict[str, dict]:
    """구간별 Δ 평균과 짝 수 — 이미 잰 값만 쓰므로 비용이 0 이다. 퇴행 가드가 읽는다."""
    by: dict[str, list[float]] = {}
    for key in keys:
        i, k = key
        b = latency_bin(len(L.units_of(sents[i].text, spaced)), k)
        by.setdefault(b, []).append(new_h[key] - old_h[key])
    return {b: {"mean": round(st.mean(v), 4), "pairs": len(v)}
            for b, v in sorted(by.items(), key=lambda kv: float(kv[0].strip("≤>")))}


def induce_picks(cases: list[dict], excl: set, n: int, part: int = 0,
                 parts: int = 1) -> list[dict]:
    """`induce` 에게 보여줄 사례 — **구간 비중을 사례 배분과 같게** 맞춰 고른다.

    gap 내림차순으로 그냥 n 개를 뽑으면 두 가지가 어긋난다. 첫째, gap 절대값은 **긴 구간에서
    크게 나온다** — 절단이 적을수록 한 자리가 틀렸을 때 집합 전체가 받는 타격이 크다. 실측으로
    사례 100 이 ≤3 에 57개를 배분했는데 gap 상위 8 은 ≤5 가 4개, ≤3 이 2개였다(judge24 첫 시도).
    목표가 짧은 지연 개선인데 그 구간이 가장 적게 들어간다. 둘째, 프롬프트가 이미 예시로 든
    문장이 섞인다 — 그 문장은 모델이 답을 본 자리다.

    그래서 구간별 사례 수 비율대로 할당량을 정하고 각 구간 안에서 gap 상위를 뽑는다. 남는
    자리는 gap 순으로 채운다.

    `parts` 를 2 이상으로 주면 풀을 gap 순으로 그만큼 갈라 `part` 번째 조각만 쓴다. `induce` 는
    finding 을 안 받으므로(진단을 주지 않는 역할이다) 후보를 둘 두면 같은 입력이 되어 하나가
    중복으로 버려진다 — 대신 **다른 사례 묶음**을 줘서 하나는 크게 실패한 자리, 하나는 그보다
    흔한 자리를 보게 한다."""
    pool = [c for c in cases if c.get("id") not in excl]
    if not pool or n <= 0:
        return []
    # **상위 절반만 쓴다.** gap 은 그 자리에서 오라클 대비 잃은 양이라, 하위 절반은 프롬프트가
    # 거의 맞힌 자리다. 거기서 "공통점을 규칙으로 써라" 하면 설명할 차이가 없는데 설명을 만들어야
    # 하고, 그 잡음이 규칙으로 굳어 다른 문장에서 손해가 된다. judge24 네 이터가 전부 그랬다 —
    # 상위 묶음 대 하위 묶음이 +0.0056/+0.0048, −0.0028/−0.0058, −0.0076/−0.0126,
    # −0.0015/−0.0146 으로 네 번 다 상위가 낫고 하위는 세 번 음수, 두 번은 구간 전체가 0 아래였다.
    # 절대 하한(gap ≥ 0.4)은 못 쓴다 — 프롬프트가 나아지면 사례 전체가 얕아져 이터 4 에서는
    # 0.4 이상이 3개뿐이었다. 상대 기준이라야 이터가 가도 묶음이 나온다.
    ranked = sorted(pool, key=lambda x: -float(x.get("gap") or 0))
    top = ranked[:max(n * max(1, parts), len(ranked) // 2)]
    if parts > 1:
        # 상위 절반 **안에서** 겹치지 않게 자른다. 묶음마다 아래의 구간 층화가 다시 걸린다.
        size = max(n, len(top) // parts)
        pool = top[part * size:(part + 1) * size] or top[:size]
    else:
        pool = top
    by_bin: dict[str, list[dict]] = {}
    for c in sorted(pool, key=lambda x: -float(x.get("gap") or 0)):
        by_bin.setdefault(c.get("latency_bin") or "?", []).append(c)
    quota = {b: max(1, round(n * len(v) / len(pool))) if len(v) * n >= len(pool) else 0
             for b, v in by_bin.items()}

    # **모순으로 죽은 사례를 절반으로 제한한다.** gap 이 큰 것은 거의 전부 정책 H_set 이 0.05
    # 아래로 떨어진 문장이고(한 자리가 치명적으로 틀려 집합 전체가 죽은 것), 그것만 보여주면
    # 모델은 "모순 회피" 한 가지만 배운다. 실측으로 층화 후에도 8/8 이 그런 사례였다.
    # 절반은 덜 극단적인 실패에서 뽑아 흔한 자리도 보게 한다.
    def killed(c):
        return ((c.get("policy") or {}).get("H_set") or 1.0) < 0.05

    cap_killed = max(1, n // 2)
    out: list[dict] = []
    n_killed = 0
    for b in sorted(by_bin, key=lambda x: -len(by_bin[x])):
        want = quota.get(b, 0)
        took = 0
        for c in by_bin[b]:
            if took >= want:
                break
            if killed(c) and n_killed >= cap_killed:
                continue
            out.append(c)
            took += 1
            n_killed += killed(c)
    # 할당이 모자라면 gap 순으로 채운다. **상한을 먼저 지켜 보고, 그래도 모자라면 풀어서 개수를
    # 맞춘다** — 상위 절반으로 좁히면 그 안이 대부분 모순으로 죽은 사례라(gap 큰 자리가 곧 그런
    # 자리다) 상한을 끝까지 지키면 묶음이 8개에서 6개로 줄어든다. 개수를 채우는 쪽이 낫다.
    for relax in (False, True):
        if len(out) >= n:
            break
        seen = {id(c) for c in out}
        for c in sorted(pool, key=lambda x: -float(x.get("gap") or 0)):
            if len(out) >= n:
                break
            if id(c) in seen:
                continue
            if not relax and killed(c) and n_killed >= cap_killed:
                continue
            out.append(c)
            seen.add(id(c))
            n_killed += killed(c)
    return out[:n]


def adopt_keys(sents: list, keys: list, spaced: bool, bin_only: str | None) -> list:
    """채택 판정에 쓸 짝 — `bin_only` 가 있으면 그 지연 구간의 짝만 남긴다.

    H_set 은 다섯 구간의 평균이라 **한 구간의 개선이 나머지 넷의 번짐에 상쇄된다.** judge21
    홀드아웃 실측: `fallback` 순서 규칙이 test 에서 ≤3 +0.0055 인데 평균은 −0.0019 였다.
    최종 지표(CoVoST2 등지연 COMET)에서 지는 자리가 짧은 지연이므로 거기서 가른다.
    다른 구간의 퇴행은 `--guard-bin` 이 따로 막는다."""
    if not bin_only:
        return keys
    return [k for k in keys
            if latency_bin(len(L.units_of(sents[k[0]].text, spaced)), k[1]) == bin_only]


def guard_breaches(bins: dict[str, dict], guard: float, thin: int = 300) -> list[str]:
    """`guard` 보다 크게 떨어진 구간 — 짝이 `thin` 개 미만인 구간은 CI 가 넓어 두 배로 봐준다.

    문턱을 평균 > 0 으로 낮추면 **짧은 구간을 벌고 긴 구간을 파는 개정**이 통과한다. 실측으로
    judge14 iter 1 (≤99 −0.0226) 과 zh iter 4 (≤5 −0.025) 가 그런 꼴이었고, 반대로 ja·zh 의
    양수 후보 여섯 중 다섯은 다섯 구간 전부 양수라 이 가드에 걸리지 않는다."""
    if guard <= 0:
        return []
    return [f"{b} {v['mean']:+.4f}" for b, v in bins.items()
            if v["mean"] < -(guard * (2 if v["pairs"] < thin else 1))]


# v0 후보 둘의 H_set 차이가 이 값 안이면 점수로 고르지 않는다. dev 500 한 벌 채점의 표준편차
# 실측값(0.0039)이다 — 그보다 작은 차이는 프롬프트의 차이가 아니라 분절의 비결정론이다.
V0_TIE_BAND = 0.004


# 역할이 담을 수 있는 발견의 형태. **형식과 내용이 맞아야 한다** — 두 자리를 가르는 비교를
# 결속문 역할(`single_small`)에 넣으면 비교가 단항으로 납작해지고, 단항 조건을 선호문 역할
# (`fallback`)에 넣으면 비교가 아닌 것을 비교문처럼 쓰느라 군더더기가 붙는다.
# 목록에 없는 역할(`free`·`rewrite` 등)은 둘 다 받는다. `prune` 은 발견을 구현하는 역할이
# 아니라 단위를 지우는 역할이라 짝을 짓지 않고 이터당 하나 둔다.
ROLE_KINDS = {"fallback": {"order"},
              # 이항 발견이 `[Order Principles]` 아닌 집을 갖는다. 그 칸의 계약은 골격의 등급 안
              # 기본 서열을 **덮어쓰는** 것이라, 정도 기반 서열이 이득을 내는 위에서는 이득을 깎는다
              # (같은 편집이 v0 위 −0.0015, graded 위 −0.0053). `severity` 는 대신 그 발견이 가리킨
              # 원칙의 **정도 축**을 고친다 — 둘 중 어느 쪽이 덜 나쁜지를 말하는 자리다.
              "severity": {"order"},
              "single_small": {"check"},
              "induce": {"check"},
              "narrow_rule": {"check"},
              "examples_only": {"check"}}
# 발견과 짝짓지 않는 역할 — 이터당 하나다. `prune` 은 발견을 구현하는 역할이 아니라 단위를
# 지우는 역할이라 짝을 짓지 않는다.
UNPAIRED_ROLES = {"prune"}


def candidate_plan(roles: list[str], n_find: int, cross: bool, n_default: int,
                   cap: int = 16, kinds: list[str] | None = None) -> list[tuple[str, int]]:
    """후보 j 에게 줄 (역할, finding 인덱스) 표.

    기본 배정은 역할·finding 이 **각각 독립 나머지 연산**이라 주기가 LCM(역할수, finding수) 이다 —
    역할 4 · finding 4 · 후보 12 면 4쌍이 3번 반복되고 시도 폭이 안 는다. `cross` 는 (역할 × finding)
    을 한 번씩 전부 돌려 후보 수를 finding 수에 맞춰 자동으로 정한다. `cap` 은 게이트 비용 상한
    (게이트는 후보 × 홀드아웃 문장을 분절한다).

    `kinds` (finding 별 "check"/"order") 를 주면 **형식이 내용을 담을 수 있는 짝만** 곱한다
    (`ROLE_KINDS`). 후보 수가 줄어 이터당 GPU 큐와 비용이 내려가고, 관문을 들여다보는 횟수가
    줄어 우연 통과 압력도 낮아진다. 걸러낸 뒤 짝이 하나도 안 남으면 이터를 버리는 대신 종전
    곱으로 되돌린다."""
    n_find = max(1, n_find)
    if cross and roles:
        if kinds:
            pairs = [(r, i) for r in roles if r not in UNPAIRED_ROLES
                     for i, k in enumerate(kinds[:n_find])
                     if k in ROLE_KINDS.get(r, {"check", "order"})]
            pairs += [(r, 0) for r in roles if r in UNPAIRED_ROLES]
            if pairs:
                return pairs[:cap]
        n = min(len(roles) * n_find, cap)
        return [(roles[j // n_find], j % n_find) for j in range(n)]
    return [((roles[j % len(roles)] if roles else "free"), j % n_find) for j in range(n_default)]


def classify_type(gw, predicate: str, sents: list, ids: list[int], cache: JsonCache,
                  log_fn=None, chunk: int = 40) -> list[int]:
    """술어에 맞는 문장 인덱스 — 덩이마다 LLM 분류 1회. `(술어, 그 덩이의 id들)` 로 캐시한다.

    **유형 홀드아웃은 현재 프롬프트의 성적이 아니라 원문 구문으로 고른다.** 사례가 현재 프롬프트의
    실패에서 뽑히므로, 거기서 다시 고르면 선택 편향이 되돌아온다(judge05 가 걸린 함정). 술어는
    원문만 보고 판정 가능해야 하고, 그 검사는 Critic 지시문과 아래 너비 검사가 나눠 맡는다.

    한 번에 140문장을 주면 **사고 토큰이 예산을 다 먹고 본문이 빈 채로 온다** — 실측으로 빈 응답이
    두 번(원본·간결 재시도) 나고 파싱 실패가 런을 죽였다(02:24 judge17). 40문장씩 끊어 사고량을
    줄이고, 사고는 low 로 낮춘다. 분류는 구문 유무를 보는 일이라 사고를 많이 쓸 일이 아니다.
    한 덩이가 깨져도 그 덩이만 비우고 넘어간다 — 유형 하나를 놓치는 것과 런이 죽는 것은 다르다."""
    got: list[int] = []
    failed: list[str] = []
    for start in range(0, len(ids), chunk):
        part = ids[start:start + chunk]
        pset = set(part)
        key = hashlib.sha1(f"{predicate}|{','.join(map(str, part))}".encode()).hexdigest()[:16]
        hit = cache.get(key)
        if hit is not None:
            got += [i for i in hit if i in pset]
            continue
        try:
            blob = gw.chat_json(aj.CLASSIFY_SYSTEM, aj.classify_user(predicate, sents, part),
                                max_tokens=8000, reasoning_effort="low", purpose="classify")
        except Exception as e:      # noqa: BLE001 — 분류 실패는 유형 하나를 잃을 뿐이다
            failed.append(f"{start}-{start + len(part)}: {type(e).__name__}")
            continue
        part_got = [i for i in ((blob or {}).get("matching") or [])
                    if isinstance(i, int) and i in pset]
        cache.put(key, part_got)
        got += part_got
    if log_fn:
        log_fn(f"[유형] '{predicate[:60]}…' → {len(got)}/{len(ids)}문장"
               + (f" / 분류 실패 {len(failed)}덩이 ({'; '.join(failed)})" if failed else ""))
    return sorted(got)


def merge_winners(prompt: str, winners: list[dict], budget: int) -> tuple:
    """유형별 1등들의 편집을 한 프롬프트로 합친다 — (프롬프트|None, 변경목록, 편집, 건너뛴 id).

    유형마다 하나씩만 합치는 것이 전제다. 같은 유형의 편집 둘은 같은 자리를 두 번 제약해 상호작용이
    크지만, 다른 유형은 건드리는 구문이 달라 가산성이 그럴듯하다. 같은 단위(id)를 두 번 고치는
    편집은 유형 Δ 가 높은 쪽만 남긴다 — 나중 편집이 앞 편집을 덮어써 무엇이 적용됐는지 잃는다."""
    seen_ids: set[str] = set()
    edits, notes, dropped = [], [], []
    for w in sorted(winners, key=lambda x: -x["delta"]["lo"]):
        for e in (w["pe"] or {}).get("edits") or []:
            eid = str(e.get("id") or "")
            if eid in seen_ids:
                dropped.append(eid)
                continue
            seen_ids.add(eid)
            edits.append(e)
        notes += [str(x) for x in ((w["pe"] or {}).get("changelog") or [])]
    if not edits:
        return None, ["합칠 편집이 없다"], [], dropped
    # 합치는 편집이 어느 역할에서 왔는지 여기서는 모른다. S 단위를 고치는 편집이 섞여 있으면
    # 그 칸을 열어 준다 — `enforce_role` 이 이미 `procedure` 만 S 를 만질 수 있게 걸렀으므로,
    # 여기 S 편집이 있다는 것 자체가 그 역할을 통과했다는 뜻이다.
    allow = ("[Scoring Rules]",) if any(str(e.get("id") or "").startswith("S")
                                       for e in edits) else ()
    cand, note, _draft, deltas, _skipped = aj.parse_edits(
        {"changelog": notes, "edits": edits}, prompt, budget, allow)
    return cand, (notes if cand else note), deltas, dropped


def failure_brief(cases: list[dict], n: int = 8) -> str:
    """가장 크게 손해 본 사례 n개 — 원문에 현재 절단(‖)과 오라클 절단을 나란히 보인다.

    Critic 요약만 주면 Writer 는 "무엇이 틀렸다" 는 서술만 받고 **어디를 어떻게 잘랐어야 했는지**는
    못 본다. 오라클 절단은 전수 탐색으로 정해진 결정론적 값이라, 이걸 보여주는 것은 판정 집합을
    오염시키지 않는다 — 사례는 train 에서만 뽑는다."""
    worst = sorted(cases or [], key=lambda c: -float(c.get("gap") or 0))[:max(1, n)]
    out = []
    for c in worst:
        pol, tgt = c.get("policy") or {}, c.get("target") or {}
        out.append(f"[{c.get('latency_bin')}, gap {float(c.get('gap') or 0):.3f}]\n"
                   f"  now : {pol.get('text')}\n"
                   f"  best: {tgt.get('text')}")
    return "\n\n".join(out)


def pick_key(rule: str):
    """후보를 줄 세우는 자 — 채택을 가르는 자와 같아야 한다.

    선택은 하한, 채택은 평균으로 갈리면 평균이 높고 분산이 큰 후보가 선택 단계에서 조용히 떨어진다.
    합본은 편집을 여럿 담아 본질적으로 분산이 큰 후보라 그 벌점을 정면으로 맞는다 — 합쳐서 이득인지
    보려고 합본과 단일을 둘 다 본채점하는 목적이 거기서 깨진다."""
    return (lambda d: d["mean"]) if rule == "mean" else (lambda d: d["lo"])


def decide(boot: dict, strong: float = 0.005, rule: str = "lo",
           bins: dict[str, dict] | None = None, guard: float = 0.0) -> str:
    """`rule="lo"` 는 CI 하한으로 가른다 — 종전 1 se 문턱은 재채점 잡음(+0.023±0.017)이 그냥 넘었다.

    `rule="mean"` 은 평균 > 0 + 퇴행 가드다. **한 스텝의 진짜 효과(+0.003~0.01)가 200문장 se
    (0.0075~0.0095)보다 작아 하한 문턱은 원리적으로 못 넘는다** — judge08~14·de·ja·zh 열여섯
    이터에서 채택 0 이었다. 스텝을 탐색으로 쓰고 판정을 런 끝 홀드아웃 비교 한 번으로 옮긴다.
    스텝마다 잡음 위쪽을 고르므로 누적 Δ 는 위로 부풀지만, 그 편향은 최종 비교에 안 들어간다."""
    if rule == "mean":
        if boot["mean"] <= 0:
            return "reject"
        return "reject_guard" if guard_breaches(bins or {}, guard) else "accept"
    if boot["lo"] > strong:
        return "accept"
    if boot["lo"] > 0:
        return "confirm"
    return "reject"


# ── 실행 ────────────────────────────────────────────────────────────────

# ── 재개 ────────────────────────────────────────────────────────────────

STATE_FILE = "state.json"


def violation_summary(rows: list[dict]) -> dict:
    """포맷 위반 요약 — 무엇이 몇 건 깨졌나. 통과율 숫자만 로그에 찍으면 무엇이 깨졌는지 안 남아서,
    1차 통과율이 떨어져도 원인을 재현 실험으로만 찾을 수 있었다."""
    by_rule: dict[str, list[str]] = {}
    for r in rows:
        for v in r.get("violations") or []:
            by_rule.setdefault(str(v), []).append(r["id"])
    return {"n_rows": len(rows),
            "valid": sum(1 for r in rows if r.get("valid")),
            "first_pass": sum(1 for r in rows if r.get("first_pass")),
            "by_rule": {k: {"n": len(v), "ids": v[:10]}
                        for k, v in sorted(by_rule.items(), key=lambda kv: -len(kv[1]))}}


def revision_diagnosis(sents: list, old_h: dict, new_h: dict, old_sets: dict, new_sets: dict,
                       spaced: bool, n: int = 5) -> dict:
    """개정이 어디서 좋아지고 어디서 나빠졌는지 — 이미 잰 값만 쓰므로 채점 비용이 0 이다.

    이력에 평균 Δ 하나만 남기면 "무엇이 왜 나빠졌나" 를 아무도 못 본다. judge05 는 같은 방향의
    개정이 네 번 연속 기각되는 동안 Critic 이 그 사실조차 몰랐다."""
    keys = sorted(set(old_h) & set(new_h))
    by_bin: dict[str, list[float]] = {}
    for i, k in keys:
        b = latency_bin(len(L.units_of(sents[i].text, spaced)), k)
        by_bin.setdefault(b, []).append(new_h[(i, k)] - old_h[(i, k)])

    def item(key):
        i, k = key
        u = L.units_of(sents[i].text, spaced)
        return {"id": sents[i].id, "k": k, "bin": latency_bin(len(u), k),
                "delta": round(new_h[key] - old_h[key], 4),
                "before": _marked(u, old_sets[key], spaced)[:220],
                "after": _marked(u, new_sets[key], spaced)[:220]}

    ranked = sorted(keys, key=lambda key: new_h[key] - old_h[key])
    return {"by_bin": {b: round(st.mean(v), 4)
                       for b, v in sorted(by_bin.items(), key=lambda kv: float(kv[0].strip("≤>")))},
            "n_worse": sum(1 for key in keys if new_h[key] < old_h[key]),
            "n_better": sum(1 for key in keys if new_h[key] > old_h[key]),
            "worst": [item(key) for key in ranked[:n]],
            "best": [item(key) for key in reversed(ranked[-n:])]}


def checkpoint_verdict(prev: dict | None, h: dict) -> tuple[str, dict | None]:
    """dev-B 체크포인트 판정 — 직전 체크포인트와 (문장, k) 짝으로 비교해 CI 상한이 0 아래일 때만
    롤백한다. 점 비교는 같은 프롬프트를 캐시로 다시 잰 값의 흔들림(judge05: 0.5440 → 0.5416)에도
    롤백해, 채택본이 바뀐 뒤라면 멀쩡한 개정을 버린다."""
    if not prev:
        return "save", None
    old = {(i, k): v for i, k, v in prev.get("h") or []}
    keys = sorted(set(old) & set(h))
    if not keys:        # 짝 기록이 없는 체크포인트(그 기록 전 코드로 저장된 것)는 점으로 비교한다
        return ("rollback" if st.mean(h.values()) < prev["value"] - 1e-6 else "save"), None
    boot = hset.paired_bootstrap([h[k] for k in keys], [old[k] for k in keys],
                                 clusters=[i for i, _k in keys])
    return ("rollback" if boot["hi"] < 0 else "save"), boot


def length_cap(cur_len: int, v0_len: int, growth: float, ceiling: float) -> int:
    """이번 이터 PE 개정본의 길이 상한 — 직전 채택본 대비 증가율과 런 천장 중 작은 쪽.

    시작 길이에 고정하면 한 번 채택된 뒤 여유가 사라진다(judge04: 상한 8,664 에 채택본
    8,652자, iter 2~4 전부 반려). 증가율만 두면 이터마다 쌓인다(5% × 5이터 = 1.28배)."""
    return min(int(cur_len * (1 + growth)), int(v0_len * ceiling))


def save_state(path: Path, done: int, prompt: str, history: list, checkpoint: dict | None,
               v0_len: int, total_cost: float, provenance: dict) -> None:
    """이터레이션 경계의 루프 상태. 임시 파일에 쓰고 이름을 바꿔, 쓰는 도중에 죽어도 직전
    상태가 온전히 남는다."""
    blob = {"done": done, "prompt": prompt, "history": history, "checkpoint": checkpoint,
            "v0_len": v0_len, "total_cost": round(total_cost, 6), "provenance": provenance}
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(blob, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)


def load_state(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def prior_spend(run_dir: Path) -> float:
    """앞선 실행들이 이 런에 쓴 돈. 게이트웨이 누적은 실행마다 0 에서 다시 시작하므로
    `metrics.json` 에 앞선 실행 몫을 더한 `run_total_cost` 를 같이 적고, 그 최댓값을 쓴다.
    그 필드가 없는 기록은 그 실행 하나의 누적 `usage.cost` 로 본다."""
    best = 0.0
    for p in list(run_dir.glob("iter_*/metrics.json")) + list(run_dir.glob("final/metrics.json")):
        try:
            blob = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        best = max(best, float(blob.get("run_total_cost",
                                        (blob.get("usage") or {}).get("cost", 0.0))))
    state = load_state(run_dir / STATE_FILE)
    if state:
        best = max(best, float(state.get("total_cost", 0.0)))
    return best


def final_report(run_id: str, means: dict, by_bin: dict, boot_v0: dict | None,
                 history: list, cost: float, fmt: float) -> str:
    """런이 끝나고 남기는 한 장짜리 보고서 — test 에서 잰 값과 이터 이력."""
    bins = sorted({b for d in by_bin.values() for b in d},
                  key=lambda b: float(b.strip("≤>")))
    who = [k for k in ("prompt", "v0", "oracle") if k in by_bin]
    lines = [f"# {run_id} — 최종", "",
             f"채택본 test H_set **{means['prompt']:.4f}** / 오라클 {means['oracle']:.4f}"
             + (f" / v0 {means['v0']:.4f}" if "v0" in means else " (v0 그대로 — 채택된 개정 없음)"),
             ""]
    if boot_v0:
        lines += [f"v0 대비 Δ {boot_v0['mean']:+.4f} "
                  f"[{boot_v0['lo']:+.4f}, {boot_v0['hi']:+.4f}] "
                  f"(짝 {boot_v0['n']} / 문장 {boot_v0['n_clusters']})", ""]
    lines += ["## 지연 구간별 test H_set", "",
              "| 구간 | " + " | ".join(who) + " |",
              "|---|" + "---|" * len(who)]
    for b in bins:
        lines.append(f"| {b} | " + " | ".join(f"{by_bin[w].get(b, float('nan')):.4f}" for w in who)
                     + " |")
    lines += ["", f"포맷 통과율 {fmt}", "", "## 이터레이션", "",
              "| 이터 | 결과 | Δ (dev-A) | 비고 |", "|---|---|---|---|"]
    for h in history:
        d = h.get("gain") or h.get("delta")
        delta = f"{d['mean']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}]" if d else "—"
        note = "채택" if h.get("adopted") else ("반려: " + "; ".join(h["reason"])[:60]
                                               if h.get("reason") else "기각")
        if h.get("screened_out"):
            note = f"선별 탈락 (후보 {h.get('candidate')})"
        elif h.get("screened_only"):
            note = f"선별 기각 (후보 {h.get('candidate')}, 본채점 생략)"
        why = (h.get("diagnosis") or {}).get("why") or ""
        lines.append(f"| {h['iter']} | {note} | {delta} | {str(why)[:80]} |")
    lines += ["", f"비용 ${cost:.2f}", ""]
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--from-run", required=True, help="분할·라벨·캐시를 물려받을 런")
    p.add_argument("--run-id", required=True)
    p.add_argument("--prompt", default=None,
                   help="시작 프롬프트 파일. 없으면 --generate-v0 로 만든다")
    p.add_argument("--generate-v0", action="store_true",
                   help="v0 를 손으로 쓰지 않고 만든다: Profiler 가 소스 문장에서 언어 특징을 뽑고 "
                        "Writer 가 그것으로 판단형 프롬프트를 쓴다. 손으로 쓴 v0 는 영어 부정어 "
                        "목록과 영어 예시가 박혀 있어 다른 소스 언어로 못 옮긴다")
    p.add_argument("--v0-candidates", type=int, default=2,
                   help="생성할 v0 후보 수. dev-A H_set 으로 골라 시작한다")
    p.add_argument("--dev-a", type=int, default=150, help="dev 앞 N 문장을 채택 판정에 쓴다")
    p.add_argument("--min-gap", type=int, default=1,
                   help="절단 사이 최소 조각 길이. **기본 1 — 제약을 걸지 않는다.** 종전 루프는 "
                        "발화 속도에서 유도한 3 을 썼는데, 그건 ASR 쪽 노브지 이 지표의 일부가 "
                        "아니다. 너무 짧은 조각은 H_set 이 알아서 벌한다")
    p.add_argument("--min-chunk", type=int, default=2,
                   help="평균 조각이 이보다 짧아지는 k 는 안 잰다 — 1어절 조각 구간은 잡음뿐")
    p.add_argument("--max-k", type=int, default=10, help="문장당 잴 절단 수 상한")
    p.add_argument("--iterations", type=int, default=5)
    p.add_argument("--n-cases", type=int, default=12)
    p.add_argument("--target-sets", default=None,
                   help="집합 탐색 산출물(set_scores_dev.json). 주면 목표가 그리디 오라클 대신 "
                        "탐색 최적이 된다")
    p.add_argument("--checkpoint-every", type=int, default=2, help="0 이면 dev-B 를 안 본다")
    p.add_argument("--skip-final", action="store_true",
                   help="런 끝의 test 평가를 건너뛴다. 기본은 채택본과 v0 를 test 에서 한 번씩 재고 "
                        "지연 구간별 표를 남긴다 (약 $2~4)")
    p.add_argument("--growth-per-iter", type=float, default=0.10,
                   help="PE 개정본이 직전 채택본보다 길어질 수 있는 비율. 넘으면 PE 가 근거 약한 "
                        "단위를 갈아끼우거나 압축해야 한다")
    p.add_argument("--growth-ceiling", type=float, default=1.3,
                   help="런 전체 길이 천장 — 시작 프롬프트 길이의 배수. 증가율이 이터마다 쌓이는 "
                        "것을 막는다")
    p.add_argument("--k-samples", type=int, default=3)
    p.add_argument("--pe-candidates", type=int, default=1,
                   help="이터마다 받는 PE 개정 후보 수. 2 이상이면 dev-A 앞 --screen-n 문장에서 "
                        "먼저 재고 Δ 평균이 가장 높은 후보만 본채점한다")
    p.add_argument("--screen-n", type=int, default=50,
                   help="후보 선별에 쓰는 판정 분할 문장 수. **0 이면 선별을 건너뛰고 곧바로 "
                        "본채점한다.** 문장은 이터마다 무작위로 흩어 뽑으므로(`screen_indices`) "
                        "본채점의 배치 경계와 맞지 않아 캐시가 거의 재사용되지 않는다 — 선별은 "
                        "비용·시간 순증이다. 필터는 judge20 에서 폐지했고(선별·본채점 아홉 쌍 중 "
                        "다섯이 부호 뒤집힘) 남은 값은 중간 보고뿐이다.")
    p.add_argument("--screen-skip", action="store_true",
                   help="선별 최고 후보의 Δ 평균이 0 이하면 본채점 없이 기각한다 (부검은 선별 문장으로)")
    p.add_argument("--candidate-roles", default="",
                   help="후보별 역할을 쉼표로: free / examples_only / single_small / narrow_rule / "
                        "replace / rewrite. replace 는 원칙 하나를 지우고 그 자리에 발견을 넣는다 "
                        "(delete 1개 + insert_after 1개, 순증가 상한). 문장을 더한 스물세 번 중 "
                        "스물두 번이 음수인데 지운 편집은 0 이었으므로 길이 중립을 시험한다. "
                        "narrow_rule 은 [Core Principles] 에 **추가**만 한다(insert_after 1개) — "
                        "실측상 채택된 둘은 좁은 점검 추가였고 원칙 재조정 다섯은 전부 실패했다. 후보 j 는 "
                        "roles[j %% len] 을 받는다. rewrite 는 Writer 가 [Core Principles]·[Examples] "
                        "를 새로 쓴다(--generate-v0 런에서만). severity 는 [Core Principles] 원칙 "
                        "하나의 **정도 절**(첫 물음표 뒤)만 다시 쓴다 — 질문은 한 글자도 못 바꾼다. "
                        "질문은 등급 배정을 정하고 그쪽은 이미 작동하며(1위 6.16배), 없는 것은 등급 "
                        "안 서열이다(8위 1.27배). 정도 절을 손으로 넣어 본 값이 dev 3벌 +0.0076 "
                        "[+0.0028, +0.0127] 이고 이득이 ≤3 구간에 몰렸다. 이항 발견과 짝짓는다. "
                        "예: free,examples_only,rewrite")
    p.add_argument("--case-alloc", default="uniform", choices=("uniform", "loss"),
                   help="구간별 사례 수 — uniform: 한 바퀴씩 균등 / loss: 구간별 손실 몫에 비례(최소 1). "
                        "test-A 실측은 손실의 61%% 가 ≤3, 26%% 가 ≤5 인데 균등 배분은 그 둘에 6/12 만 준다")
    p.add_argument("--full-score-max", type=int, default=1,
                   help="선별 뒤 본채점할 후보 수 상한 — 선별 평균이 양수인 후보를 Δ 순으로 이만큼. "
                        "1 이면 종전처럼 최선 하나. judge12 는 5이터 전부 후보 CI 가 겹쳐 선별이 고르지 못했다")
    p.add_argument("--verify-dev-b", action="store_true",
                   help="dev-A 에서 채택된 개정을 dev-B 에서 **한 번 더 재서** 확인한다. 고르기가 "
                        "없는 단계라 선택 편향이 없다 — judge21 이 dev 600 하나로 12번 재고 고른 "
                        "개정이 최종 홀드아웃에서 −0.0089 로 뒤집혔다. --confirm-dev-b 의 합산과 "
                        "다르다(합산은 표본만 늘려 구간을 좁힌다).")
    p.add_argument("--verify-tries", type=int, default=1,
                   help="dev-B 확인을 몇 순위까지 시도하는가. 1 이면 dev-A 1등만 본다. 2 이상이면 "
                        "1등이 탈락할 때 다음 순위로 내려가되 **문턱이 --verify-step 씩 올라간다** — "
                        "확인 분할에서 여러 번 재는 것 자체가 고르기이므로 뒤 순위는 더 강한 증거가 "
                        "필요하다. dev-A 하한 > 0 을 통과한 후보만 후보군이 된다.")
    p.add_argument("--verify-step", type=float, default=0.005,
                   help="dev-B 확인 시도가 한 순위 내려갈 때 올리는 문턱 폭.")
    p.add_argument("--verify-min", type=float, default=0.0,
                   help="dev-B 확인 문턱 — 평균이 이 값을 넘어야 채택을 유지한다.")
    p.add_argument("--confirm-dev-b", action="store_true",
                   help="dev-A 에서 기각됐지만 평균 > 0.005, 하한 > -0.01 인 개정은 dev-B 를 더 재서 "
                        "합산(415문장) CI 하한으로 다시 가른다. 유망한 개정 하나에 약 $5")
    p.add_argument("--labeled-examples", action="store_true",
                   help="사례 문장을 실측 라벨로 채운 Input/Output 예시로 만들어 PE 에 준다. PE 는 "
                        "편집에 labeled_example: <사례 id> 로 그 예시를 그대로 붙일 수 있다")
    p.add_argument("--adopt-strong", type=float, default=0.005,
                   help="**재추출 확인 없이 바로 채택하는 하한 문턱이다.** 하한이 0 과 이 값 "
                        "사이면 양쪽을 새 분절로 다시 재고(verdict 'confirm') 거기서도 하한 > 0 "
                        "이어야 채택한다. **0 으로 두면 확인이 아예 안 탄다** — judge20~25 여섯 "
                        "런이 그렇게 돌아 관문이 꺼진 채였고, judge25 의 채택(dev 하한 +0.0032)이 "
                        "test 560 에서 −0.0070 으로 뒤집혔다. 사후 재추출로 그 dev Δ 는 "
                        "+0.0138 → +0.0037 이었다. 확인을 늘 켜려면 도달 불가능한 값(0.05)을 준다")
    p.add_argument("--baseline-draws", type=int, default=1,
                   help="기준선(현재 프롬프트)을 몇 벌의 분절로 재서 (문장, k) 별로 평균할지. "
                        "**기준선 한 벌은 후보 전부가 공유하므로 그 벌의 오차가 모든 후보 Δ 에 "
                        "같은 방향으로 얹힌다** — 2026-09-19 실측으로 v0 를 dev 500 에서 네 번 뽑으니 "
                        "0.5514 / 0.5578 / 0.5588 / 0.5591 이었고, judge24~26 이 판정에 쓴 첫 벌이 "
                        "나머지 평균보다 0.0072 낮아 후보 Δ 전부가 그만큼 올려 잡혔다. 3 이면 그 "
                        "공통 편향이 √3 로 줄고 짝 비교 sd 도 함께 내려간다")
    p.add_argument("--confirm-draws", type=int, default=1,
                   help="1차를 통과한 후보를 **총 몇 벌**로 다시 재는지(1차에서 쓴 벌 포함). 2 이상이면 "
                        "1차는 선별이 되고 판정은 이 다벌 평균이 한다 — 통과자 전부를 재고, 하한 > 0 인 "
                        "것 중 하한 최고를 채택한다. 1 이면 종전 경로(최고 하나만 --adopt-strong 구간에서 "
                        "새 추출 확인). 벌을 3 으로 하고 기준선도 3 벌이면 Δ 의 sd 가 0.0058 → 0.0033 "
                        "으로 줄어 하한 > 0 문턱이 +0.011 에서 +0.0065 로 내려간다")
    p.add_argument("--draw-tag", default=None,
                   help="벌별 캐시 파일 이름에 쓰는 꼬리표. 기본은 런 이름이다. **다른 런에서 뽑아 둔 "
                        "벌을 재사용하려면 그 런과 같은 값을 준다** — 기준선 3벌을 다시 뽑지 않아도 된다")
    p.add_argument("--contra-agg", default="max", choices=("max", "mean"),
                   help="절단집합의 contra 를 모으는 법. **기본 max 가 지표다** — 청자를 모형화한 "
                        "선택이고 지금까지의 모든 값이 그 위에 있다. mean 은 진단용이다: max 는 "
                        "최악 하나에 지배돼 기울기가 희소하고, 여러 자리를 조금씩 낫게 만든 개정이 "
                        "보이지 않는다. 같은 개정을 두 집계로 재면 '루프가 못 찾은 것인가 목적함수가 "
                        "못 본 것인가' 가 갈린다. 분절·번역이 캐시돼 있으면 API 비용은 0 이다. "
                        "**이 값으로 잰 수치는 다른 런의 값과 비교하면 안 된다**")
    p.add_argument("--score-prompts", default=None,
                   help="--score-only 에서 기준선과 맞붙일 프롬프트 파일 여러 개를 쉼표로. 한 프로세스에서 "
                        "전부 병렬로 재므로 기준선을 한 번만 뽑고 GPU 경합도 없다. --prompt 는 무시된다")
    p.add_argument("--final-draws", type=int, default=1,
                   help="최종 test 를 몇 벌로 재서 평균할지. 판정을 다벌로 해 놓고 최종 숫자만 한 벌로 "
                        "재면 결론이 다시 ±0.011 잡음에 묻힌다 — 채택본과 v0 를 같은 벌 수로 잰다")
    p.add_argument("--gate-rule", default="lo", choices=("lo", "mean"),
                   help="--confirm-draws 2 이상일 때 1차 문턱. mean=평균 > 0 인 후보 **전부**를 2차로 "
                        "올린다(선별). lo=하한 > 0 만 올린다. **기준선을 새로 뽑으면 하한 > 0 은 거의 "
                        "안 나온다** — 세 런 후보 66개를 기준선 보정해 세어 보니 하한 > 0 은 1개(1.5%%)고 "
                        "평균 > 0 은 13개(20%%)였다. lo 로 두면 2차가 한 번도 안 돌 수 있다")
    p.add_argument("--gate-min", type=float, default=0.0,
                   help="1차 문턱의 값. 기본 0 은 '부호만 맞으면 올린다' 라서 **효과가 없는 후보의 "
                        "절반이 통과한다** — 잡음이 양수로 떨어진 쪽이 전부 올라간다. 2차가 채택할 수 "
                        "있는 하한(3벌 대 3벌이면 +0.0062)의 절반쯤을 요구하면, 거기 닿을 수 없는 "
                        "후보에 2차 채점을 쓰지 않는다. 후보당 2벌이 약 $5 라 이 문턱이 곧 비용이다")
    p.add_argument("--adopt-rule", default="lo", choices=("lo", "mean"),
                   help="채택 문턱 — lo: CI 하한 > 0.005 (종전) / mean: 평균 > 0 + 퇴행 가드. "
                        "스텝 효과(+0.003~0.01)가 200문장 se 보다 작아 lo 는 원리적으로 못 넘는다")
    p.add_argument("--adopt-bin", default=None,
                   help="채택 판정을 이 지연 구간의 짝으로만 한다 (예: ≤3). 비우면 전체 평균. "
                        "H_set 평균은 다섯 구간 평균이라 한 구간의 개선이 나머지 번짐에 상쇄된다 — "
                        "judge21 홀드아웃에서 순서 규칙이 ≤3 +0.0055 인데 평균은 −0.0019 였다. "
                        "채점은 그대로 전체를 재고 로그에 두 값을 다 남긴다.")
    p.add_argument("--guard-bin", type=float, default=0.01,
                   help="퇴행 가드 — 어느 지연 구간이든 Δ 가 이 값보다 크게 떨어지면 기각한다 "
                        "(짝 300 미만 구간은 두 배로 봐준다). 0 이면 끈다. --adopt-rule mean 전용")
    p.add_argument("--case-holdout", type=int, default=0,
                   help="사례 중 Critic·PE 에 **안 보여주고** 후보 게이트에만 쓸 개수. 편집을 쓴 "
                        "사례로 그 편집을 심사하면 사례를 외운 후보가 이긴다")
    p.add_argument("--case-gate", action="store_true",
                   help="유료 선별(dev-A 부분집합 재채점) 대신 --case-holdout 사례에서 오라클과의 "
                        "거리로 후보를 고른다. k_samples 1 로 돌아 선별보다 훨씬 싸다")
    p.add_argument("--type-select", action="store_true",
                   help="게이트 대신 **유형 Δ** 로 후보를 고른다 — Critic 이 finding 마다 쓴 "
                        "type_predicate 로 train 문장을 분류해 그 유형 문장에서만 재고, 유형별 1등을 "
                        "합쳐 본채점한다. 전체 평균은 겨냥한 효과를 나머지 문장의 잡음으로 희석한다")
    p.add_argument("--no-near-miss", action="store_true",
                   help="근소 기각본을 다음 이터의 편집 기반으로 삼지 않는다. judge20 은 이터 1 의 "
                        "rewrite 기각본(1차 포맷 통과율 0.50)이 이터 2·3 의 기반이 되면서 후속 "
                        "후보 통과율이 0.18 까지 떨어졌다 — 기반 선택에 통과율 조건이 없다. "
                        "끄면 편집 기반이 v0 로 고정되고 train 채점도 한 번으로 끝난다")
    p.add_argument("--score-workers", type=int, default=1,
                   help="본채점을 몇 후보씩 동시에 돌리는가. 후보 하나는 (문장수/batch-size) 콜밖에 "
                        "안 써서 --workers 를 다 못 채운다 — dev 500·batch 6 이면 84콜에 워커 128 로 "
                        "활용률 66%%. 2 로 두면 후보당 벽시계가 약 65%% 로 준다. GPU 채점(QE·NLI)은 "
                        "락으로 직렬화되므로 3 이상은 이득이 작다.")
    p.add_argument("--induce-cases", type=int, default=24,
                   help="induce 역할에게 보여줄 사례 수. 오라클 절단과 현재 절단을 조각 번역까지 "
                        "나란히 주고 그 선택을 재현하는 규칙을 귀납하게 한다. 사례 풀의 **상위 절반**에서 "
                        "구간 층화로 뽑는다(`induce_picks`). 건당 약 750 토큰이고 PE 입력 여유는 "
                        "330K 쯤이라 100건도 들어가지만, 8건에서는 Critic(사례 100개를 본다)보다 증거가 "
                        "12배 적어 압축 우회의 이득이 상쇄됐다 — judge24 에서 네 이터 내내 중간 성적이었다.")
    p.add_argument("--inversions-max", type=int, default=15,
                   help="Critic 에 넘길 깊은 구간 역전 쌍 수. 사례와 총량을 맞춰 쓴다 — 사례 60 "
                        "그대로 두고 더하면 Critic 입력이 그만큼 길어진다")
    p.add_argument("--findings-max", type=int, default=3,
                   help="Critic finding 상한 — 유형 수가 곧 후보 폭(역할 × 유형)이 된다")
    p.add_argument("--pe-verbatim", action="store_true",
                   help="PE 가 쓴 문면을 Critic 의 finding 문면으로 되돌린다 — 형태(단항/이항)를 "
                        "고쳐 쓰지 못하게. PE 에 남는 일은 어느 단위 옆에 넣을지와 무엇을 지울지다")
    p.add_argument("--critic-both-kinds", action="store_true",
                   help="Critic 이 check·order 를 각각 최소 하나 내게 한다. 한쪽이 비고 사유도 "
                        "없으면 한 번 다시 부른다 — 한쪽이 비면 그 형태를 쓰는 역할이 안 돈다")
    p.add_argument("--type-holdout-max", type=int, default=40,
                   help="유형 홀드아웃 문장 수 상한. 너무 작으면 유형 Δ 의 se 가 커진다")
    p.add_argument("--score-baseline",
                   help="--score-only 의 비교 기준선 프롬프트 파일. 주지 않으면 런의 prompt_v0.txt 를 "
                        "쓴다 — 그 파일은 --prompt 로 시작한 런이 덮어쓰므로, 여러 프롬프트를 같은 "
                        "런 디렉토리에서 비교할 때는 이 플래그로 기준선을 따로 준다")
    p.add_argument("--final-split",
                   help="최종 표를 낼 분할 이름. 기본은 분할 구성이 정한다(4·5분할은 test). "
                        "v0 갈래를 sel 에서만 재고 test 를 아끼려면 --final-split sel")
    p.add_argument("--score-only", action="store_true",
                   help="이터를 돌지 않고 --prompt 와 런 디렉토리의 prompt_v0.txt 를 최종 분할에서만 "
                        "재고 끝낸다 — 미리 정한 프롬프트를 홀드아웃 한 번으로 비교하는 용도. 후보를 "
                        "고르지 않으므로 선별 편향이 없다. test-A 기준선 채점($3)도 건너뛴다")
    p.add_argument("--type-control", action="store_true",
                   help="유형별 1등을 술어에 **안 맞는** 문장에서도 재서 손해가 겨냥한 구문에 몰렸는지 "
                        "본다. 진단 전용이고 유형 3개면 25분/이터가 더 든다. judge18 에서 여섯 쌍을 "
                        "받은 결과는 방향 없음(유형 평균 -0.0116 / 유형 밖 -0.0089, 3:3)이라 "
                        "기본은 끈다 — 전체에서 어떤지는 test-A 200문장 본채점이 반폭 0.015 로 답한다")
    p.add_argument("--type-rank-only", action="store_true",
                   help="유형 Δ 를 진단으로만 쓴다 — 통과 문턱을 두지 않고 상위 --full-score-max 개를 "
                        "본채점으로 올린다. 19~25문장 유형 홀드아웃의 CI 반폭이 0.03~0.05 로 실제 "
                        "효과(±0.01)보다 커서 절대 문턱은 통과를 원리적으로 막는다 — judge17·18 에서 "
                        "후보 24개가 전부 음수였고 채택이 0 이었다")
    p.add_argument("--merge-max", type=int, default=4,
                   help="한 이터에 합칠 유형별 1등 수 상한. 길이 예산(--growth-per-iter)이 실질 상한을 "
                        "따로 건다")
    p.add_argument("--candidates-cross", action="store_true",
                   help="후보를 (역할 × finding) 교차곱으로 낸다 — 역할마다 finding 전부를 한 번씩. "
                        "--pe-candidates 는 무시된다. 독립 나머지 연산은 주기가 LCM 이라 역할 4·"
                        "finding 4 면 4쌍만 돌고 나머지가 낭비된다")
    p.add_argument("--candidates-cap", type=int, default=16,
                   help="--candidates-cross 의 후보 수 상한. finding 이 많이 나온 이터에서 게이트 비용이 "
                        "터지는 것을 막는다 (게이트는 후보 × --case-holdout 문장)")
    p.add_argument("--gate-calibrate", type=int, default=0,
                   help="N 이터마다 게이트 **꼴찌** 후보도 본채점해 게이트 순위가 본채점과 맞는지 잰다. "
                        "0 이면 끈다. 이 자료가 없으면 게이트를 키울지 버릴지 판단할 수 없다")
    p.add_argument("--case-exclude", default="sent", choices=("sent", "bin"),
                   help="이터 간 사례 제외 단위 — sent: 문장 통째(종전) / bin: (문장, 지연 구간). "
                        "bin 이면 같은 문장의 다른 구간이 남아 풀이 200문장에서 수백 짝으로 는다")
    p.add_argument("--model", default="gpt-5-mini")
    p.add_argument("--seg-reasoning-effort", default="medium")
    p.add_argument("--agent-reasoning-effort", default="medium")
    p.add_argument("--batch-size", type=int, default=6)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--comet-batch-size", type=int, default=32)
    p.add_argument("--budget", type=float, default=25.0,
                   help="런 전체 비용 상한($). --resume 이면 앞선 실행 지출을 빼고 남은 만큼만 쓴다")
    p.add_argument("--resume", action="store_true",
                   help="같은 --run-id 의 state.json 에서 마지막으로 끝난 이터레이션 다음부터 잇는다. "
                        "state.json 이 없으면 v0 단계부터 — 이미 쓴 v0 후보 파일은 Writer 를 다시 "
                        "부르지 않고 그대로 채점한다(분절 캐시 적중이라 공짜)")
    add_provider_args(p)
    a = p.parse_args()

    # **역할 이름은 자유 문자열이라 오타가 조용히 통과한다.** `enforce_role` 은 모르는 역할에
    # 제약을 걸지 않으므로 `procedre` 라고 쓰면 아무 제약 없는 후보가 되고, 그 후보의 Δ 는 어느
    # 역할의 값도 아니다. 여기서 죽인다 — 뒤로 미루면 v0 생성이 끝난 다음이라 이미 돈을 썼다.
    unknown = [r.strip() for r in (a.candidate_roles or "").split(",")
               if r.strip() and r.strip() not in aj.ROLES]
    if unknown:
        print(f"[stop] 모르는 후보 역할: {unknown} — 쓸 수 있는 것은 {list(aj.ROLES)}")
        return 2

    src = RUNS_DIR / a.from_run
    run_dir = RUNS_DIR / a.run_id
    (run_dir / "cache").parent.mkdir(parents=True, exist_ok=True)
    if not (run_dir / "cache").exists():
        (run_dir / "cache").symlink_to(Path("..") / src.name / "cache")
    cfg = json.loads((src / "config.json").read_text(encoding="utf-8"))
    spaced = cfg["spaced"]
    min_gap = a.min_gap if a.min_gap is not None else cfg["min_gap"]
    t_grid = cfg["t_grid"]          # 채점(evaluate)의 라벨 지표용. 판정은 k 축으로 한다
    targets = cfg["targets"]

    def load(split):
        return [data.Sentence(**r) for r in
                json.loads((src / f"data/{split}.json").read_text(encoding="utf-8"))]

    def load_labels(split):
        return json.loads((src / f"oracle_labels_{split}.json").read_text(encoding="utf-8"))

    # 3분할 런(run26~): train 은 사례·실측 예시의 출처, test_a 는 채택 판정, test_b 는 체크포인트·
    # 유망 확인·최종 표. 2분할 런(run25)은 dev-A 가 사례와 판정을 겸해 실측 예시가 들어간 문장을
    # 그 문장에서 채점했다(judge09 iter 2: dev-A Δ +0.0105 중 +0.006 이 예시 3문장 몫).
    # 5분할 런(run28~): train 사례·예시 / sel v0 후보 선별 / dev 채택 판정 / test 최종.
    # test_a·test_b 로 갈라 체크포인트를 돌리던 구조를 dev 하나로 합쳤다 — judge15~19 다섯
    # 런에서 **롤백이 한 번도 걸리지 않았고**, 이터 3 에 저장만 하고 그대로 끝났다(값 없이
    # dev-B 채점 한 번을 태웠다). 대신 판정 집합을 200 -> 600 으로 키운다: 같은 프롬프트를
    # 두 번 재기만 해도 200문장 반폭이 0.0178 인데 루프가 잡으려는 효과가 0.008 이다
    # (`artifacts/en2x/en-multi/noise/sample_noise.json`).
    scheme_sel = (src / "data/dev.json").exists() and (src / "data/test.json").exists() \
        and not (src / "data/test_a.json").exists()
    scheme3 = (src / "data/test_a.json").exists()
    v0_sel = None       # v0 후보 선별 전용 분할. 없으면 사례 집합에서 고른다
    if scheme_sel:
        case_sents, case_lab = load("train"), load_labels("train")
        devA, labA = load("dev"), load_labels("dev")
        devB, labB = [], {t: [] for t in targets}
        # sel 은 **있을 때만** v0 후보 선별에 쓴다. 후보가 하나면 선별 자체가 없으므로
        # 분할을 따로 떼어 둘 이유도 없다 — 그 문장들은 사례 풀(train)로 보내는 편이 낫다.
        v0_sel = ((load("sel"), load_labels("sel"), "sel")
                  if (src / "data/sel.json").exists() else None)
        dev, train = devA, case_sents
        names = ("dev", "없음")
        test_sents, lab_test = load("test"), load_labels("test")
        final_split = "test"
        log(f"[data] train {len(case_sents)} (사례·예시)"
            + (f" / sel {len(v0_sel[0])} (v0 선별)" if v0_sel else "")
            + f" / dev {len(devA)} (채택 판정) / test {len(test_sents)} (최종 홀드아웃)")
        if a.checkpoint_every or a.confirm_dev_b:
            log("[data] dev-B 가 없는 분할이다 — 체크포인트·유망확인을 끈다")
            a.checkpoint_every, a.confirm_dev_b = 0, False
    elif scheme3:
        case_sents, case_lab = load("train"), load_labels("train")
        devA, labA = load("test_a"), load_labels("test_a")
        devB, labB = load("test_b"), load_labels("test_b")
        dev, train = devA + devB, case_sents
        names = ("test-A", "test-B")
        if (src / "data/test.json").exists():
            # 4분할(run27~): 최종 표는 어느 판정에도 안 쓴 test 에서 잰다. test-B 는 체크포인트·
            # 유망 확인에 쓰여 완전히 안 본 집합이 아니다.
            test_sents, lab_test = load("test"), load_labels("test")
            final_split = "test"
            log(f"[data] 4분할 — train {len(case_sents)} (사례·예시) / test-A {len(devA)} (판정) / "
                f"test-B {len(devB)} (체크포인트·확인) / test {len(test_sents)} (최종 홀드아웃)")
        else:
            test_sents, lab_test = devB, labB
            final_split = "test_b"
            log(f"[data] 3분할 — train {len(case_sents)} (사례·예시) / test-A {len(devA)} (판정) / "
                f"test-B {len(devB)} (체크포인트·확인·최종)")
    else:
        dev, train = load("dev"), load("train")
        test_sents = load("test")
        lab_test = load_labels("test")
        lab_dev, lab_train = load_labels("dev"), load_labels("train")
        devA, devB = dev[:a.dev_a], dev[a.dev_a:] + train
        labA = {t: per[:a.dev_a] for t, per in lab_dev.items()}
        labB = {t: lab_dev[t][a.dev_a:] + lab_train[t] for t in lab_dev}
        case_sents, case_lab = devA, labA
        final_split = "test"
        names = ("dev-A", "dev-B")
    # 어느 분할에서 최종 표를 낼지 명령줄로 덮어쓴다 — `--score-only` 로 프롬프트 몇 개를
    # sel 에서만 재고 싶을 때 쓴다. 최종 test 를 건드리지 않고 비교를 끝낼 수 있다.
    if a.final_split and a.final_split != final_split:
        test_sents, lab_test = load(a.final_split), load_labels(a.final_split)
        final_split = a.final_split
        log(f"[data] 최종 분할을 {final_split} 로 덮어씀 ({len(test_sents)}문장)")
    # **사례 분할이 판정 분할과 따로인가.** 이 조건으로 갈려야 할 자리를 `scheme3`(= test_a.json
    # 존재)로 물어보면, test_a 없이 train/dev/test 로 도는 구성에서 사례가 판정 집합 것으로
    # 조용히 바뀐다 — judge21 이 그렇게 죽었다(사례 train 300 에 dev 600 인덱스).
    sep_cases = scheme3 or scheme_sel
    log(f"[data] {names[0]} {len(devA)} / {names[1]} {len(devB)} / k 1~{a.max_k} "
        f"(평균 조각 ≥ {a.min_chunk}) / min_gap {min_gap}"
        f"{' (런 설정 %d 을 덮어씀)' % cfg['min_gap'] if min_gap != cfg['min_gap'] else ''}"
        f" / 타깃 {targets}")

    gw = Gateway.from_args(a, model=a.model, budget=a.budget,
                           reasoning_effort=a.agent_reasoning_effort,
                           max_connections=max(16, a.workers))
    if len(gw._keys) > 1:
        log(f"[gateway] 키 {len(gw._keys)}개 라운드로빈 / 동시 연결 {max(16, a.workers)}")
    spent_before = prior_spend(run_dir) if a.resume else 0.0
    state = load_state(run_dir / STATE_FILE) if a.resume else None
    if a.resume:
        gw.budget = a.budget - spent_before
        log(f"[resume] 앞선 실행 지출 ${spent_before:.2f} — 이번 실행 예산 ${gw.budget:.2f} / "
            + (f"이터 {state['done']} 까지 완료" if state else "state.json 없음, v0 단계부터"))
        if gw.budget <= 0:
            log("[stop] 남은 예산이 없다 — --budget 을 올려라")
            return 2
    seg_cache = JsonCache(run_dir / "cache" / "segment.json")
    # 유형 분류는 (술어, 문장 집합) 키라 같은 유형이 다음 이터에 또 나오면 공짜다.
    type_cache = JsonCache(run_dir / "cache" / "type_class.json")
    seg_effort = None if a.seg_reasoning_effort == "none" else a.seg_reasoning_effort
    translators = {t: LocalTranslator(tgt_code=to_lang_code(t),
                                      cache=JsonCache(run_dir / "cache" /
                                                      f"translate_{to_lang_code(t)}.json"))
                   for t in targets}
    scorer = hset.HsetScorer(contra_agg=a.contra_agg, translators=translators,
                             qe=metrics.make_adequacy_backend(cfg.get("adequacy_backend",
                                                                      "cometkiwi"),
                                                              batch_size=a.comet_batch_size),
                             spaced=spaced,
                             target_spaced={t: target_is_spaced(t) for t in targets})

    def save_usage(d: Path) -> None:
        """런 누적 usage 를 `iter_NN/metrics.json` 에 덮어쓴다 — `infra.cost_report` 가 읽는 형식.
        에이전트 호출과 채점마다 부르므로 크래시해도 직전 호출까지의 지출이 남는다."""
        d.mkdir(parents=True, exist_ok=True)
        u = gw.usage.snapshot()
        (d / "metrics.json").write_text(json.dumps({"usage": u,
                                                    "run_total_cost": spent_before + u["cost"]},
                                                   ensure_ascii=False, indent=1), encoding="utf-8")

    # 후보를 병렬로 채점할 때 **GPU 쪽만 직렬화한다.** 분절은 API 호출이라 동시에 던질수록
    # 이득이지만, 조각 번역 QE 와 NLI 는 GPU 하나를 쓰고 `HsetScorer` 에 락이 없다. 여기서 막으면
    # API 구간은 겹치고 GPU 구간만 줄을 선다 — judge23 실측으로 이터 시간의 66% 가 후보 채점이다.
    score_lock = threading.Lock()

    def hset_of(sents, lab, sets: dict) -> dict:
        keys = sorted(sets)
        texts = [s.text for s in sents]

        def contra_of(i, j):
            return lab[targets[0]][i]["contra"][j - 1]

        with score_lock:
            vals = scorer.score(texts, [(i, sets[(i, T)]) for i, T in keys], contra_of)
        return dict(zip(keys, vals))

    history: list[dict] = []
    cur_rows, cur_h = None, None
    if a.target_sets:
        ora_A, ora_hA = search_sets(Path(a.target_sets), t_grid, len(devA))
        kind = f"탐색 최적 ({Path(a.target_sets).name})"
    else:
        ora_A = oracle_sets(labA, devA, spaced, min_gap, a.min_chunk, a.max_k)
        ora_hA = hset_of(devA, labA, ora_A)
        kind = "그리디 오라클"
    log(f"[목표] {kind}: {names[0]} H_set 평균 {st.mean(ora_hA.values()):.4f} ({len(ora_hA)} 짝)")
    if sep_cases:
        ora_C = oracle_sets(case_lab, case_sents, spaced, min_gap, a.min_chunk, a.max_k)
        ora_hC = hset_of(case_sents, case_lab, ora_C)
        log(f"[목표] train H_set 오라클 평균 {st.mean(ora_hC.values()):.4f} ({len(ora_hC)} 짝)")
    else:
        ora_C, ora_hC = ora_A, ora_hA
    checkpoint = None

    def segment_rows(pr, sents, lab, cache=None):
        # `cache` 를 따로 주면 **같은 프롬프트라도 모델이 분절을 다시 뽑는다** — 캐시 키에
        # 프롬프트 해시가 들어가므로 파일이 다르면 전부 미스다. 재추출 확인이 이걸로 돈다.
        return evaluate(gw, pr, sents, lab, spaced, min_gap, t_grid, cache or seg_cache,
                        a.workers, a.batch_size, seg_effort, k_samples=a.k_samples)

    def draw_cache(tag: str) -> JsonCache:
        """벌마다 다른 캐시 파일. **런 이름이 들어가야 한다** — `cache/` 는 `--from-run` 쪽으로의
        심볼릭 링크라 여러 런이 한 디렉토리를 쓰고, 이름이 겹치면 다음 런이 그 벌을 그대로 읽어
        "새로 뽑은 것" 이 아니게 된다.

        **`shared` 여야 한다.** 한 벌을 여러 프롬프트가 동시에 쓰므로(변형 넷을 병렬로 재면 네
        스레드가 같은 파일을 만진다) 인스턴스를 따로 만들면 각자 같은 `.tmp` 에 쓰고 rename 해서,
        먼저 rename 한 쪽이 남의 tmp 를 없애 `FileNotFoundError` 로 런이 죽는다(2026-09-19
        judge29 첫 시도). 죽지 않는 경우에도 서로의 항목을 덮어 캐시가 조용히 사라진다."""
        return JsonCache.shared(run_dir / "cache"
                                / f"segment_{tag}_{a.draw_tag or run_dir.name}.json")

    def score_prompt(pr, sents, lab, tag, rows_m=None, cache=None):
        """분절(API) → 절단 집합 → H_set. `rows_m` 을 주면 이미 끝난 분절을 쓴다 — 후보 여럿을
        동시에 분절해 두고 GPU 채점만 차례로 할 때."""
        rows, m = rows_m or segment_rows(pr, sents, lab, cache)
        (cache or seg_cache).flush()
        sets = policy_sets(rows, sents, spaced, min_gap, a.min_chunk, a.max_k)
        h = hset_of(sents, lab, sets)
        # 번역 캐시는 20건마다만 쓴다. 채점이 끝날 때 비우지 않으면 프로세스가 죽을 때 남은 번역이
        # 사라져 다음 실행이 다시 만든다.
        for tr in translators.values():
            if tr.cache is not None:
                tr.cache.flush()
        log(f"[{tag}] H_set {st.mean(h.values()):.4f} {by_latency(sents, h, spaced)} / "
            f"overlap {m['overlap']} / fmt {m['format_pass_rate']} "
            f"(1차 {m['format_pass_rate_no_retry']}, 재정렬 {m['first_pass_violations'].get('realigned', 0)}) / "
            f"누적 ${gw.usage.snapshot()['cost']:.2f}")
        return rows, sets, h, m

    def score_avg(pr, sents, lab, tag, draws, kind):
        """분절을 `draws` 벌 뽑아 (문장, k) 별 H 를 평균한다. 첫 벌은 공용 캐시를 쓰고(다른 경로와
        공유되어 공짜일 수 있다), 둘째부터는 `kind` 전용 캐시라 반드시 새로 뽑힌다. rows·sets·metrics
        는 첫 벌 것을 그대로 돌려준다 — 위반 집계와 예시 판정에만 쓰고 판정 수치에는 안 들어간다."""
        if draws <= 1:
            return score_prompt(pr, sents, lab, tag)

        def one(d):
            # 첫 벌만 공용 캐시 — 다른 경로가 이미 뽑아 뒀으면 공짜다. 둘째부터는 전용 캐시라 새로 뽑힌다.
            return score_prompt(pr, sents, lab, f"{tag} {d}/{draws}",
                                cache=None if d == 1 else draw_cache(f"{kind}{d}"))

        # **벌끼리 병렬로 던진다.** 한 벌이 dev 500 에 6분인데 대부분이 분절 API 왕복 대기라,
        # 겹치면 3벌이 한 벌 시간에 가깝게 끝난다. GPU 채점은 `hset_of` 의 락이 줄을 세운다.
        with ThreadPoolExecutor(max_workers=min(draws, max(1, a.score_workers))) as ex:
            res = list(ex.map(one, range(1, draws + 1)))
        rows, sets, _h0, m = res[0]
        hs = [r[2] for r in res]
        ks = set(hs[0])
        for h_d in hs[1:]:
            ks &= set(h_d)
        avg = {k: sum(h_d[k] for h_d in hs) / len(hs) for k in ks}
        log(f"[{tag}] {draws}벌 평균 H_set {st.mean(avg.values()):.4f} "
            f"(벌별 {[round(st.mean(x.values()), 4) for x in hs]})")
        return rows, sets, avg, m

    v0_examples = labeled_examples(case_sents, case_lab,
                                   [{"id": i} for i in pick_example_ids(case_sents, spaced)],
                                   spaced, min_gap)

    def writer_material() -> str:
        """Writer 에게 주는 재료 — 프로파일·실측 사실·실측 예시·[Output Rules]. v0 생성과
        후보 역할 "rewrite" 가 같이 쓴다. 프로파일은 파일이 있으면 읽으므로 재개해도 돈이 안 든다.

        **프로파일·예시 문장은 어느 분할에도 안 쓰인 문장에서 뽑는다.** dev 에서 뽑으면 그
        문장이 프롬프트 안에 그대로 들어간 채로 dev 에서 채점된다 — 라벨 유출은 아니지만
        그 문장에서만 유리해진다."""
        pool_ids = {x.id for x in dev} | {x.id for x in train} | {x.id for x in test_sents}
        spare = [x for x in data.load(cfg["dataset"]) if x.id not in pool_ids]
        if len(spare) < 20:
            log(f"[v0] 분할 밖 문장이 {len(spare)}개뿐 — dev-A 에서 뽑는다")
            spare = devA
        log(f"[v0] 분할 밖 문장 {len(spare)}개에서 프로파일 20 / 실측 예시 {len(v0_examples)}개는 "
            f"train 에서: {list(v0_examples)}")
        measured = json.loads((src / "measured_profile.json").read_text(encoding="utf-8"))
        prof_path = run_dir / "language_profile.json"
        if prof_path.exists():
            profile = json.loads(prof_path.read_text(encoding="utf-8"))
        else:
            profile = agents.Profiler(gw).profile([x.text for x in spare[:20]])
            prof_path.write_text(json.dumps(profile, ensure_ascii=False, indent=2),
                                 encoding="utf-8")
        out_rules = ad.output_rules(spaced, "source", meaning_in_scoring_rules=True)
        facts = agents.measured_facts(measured)
        return (f"Language profile:\n{json.dumps(profile, ensure_ascii=False, indent=2)}\n\n"
                + (facts + "\n\n" if facts else "")
                + "Measured examples — paste them verbatim as the whole [Examples] section:\n\n"
                + "\n\n".join(v0_examples.values()) + "\n\n"
                + "Copy these two sections verbatim into the prompt — they are fixed, and the "
                  "Scoring Rules block is the procedure your two judgement sections have to "
                  f"feed:\n\n{aj.scoring_rules(targets)}\n\n{out_rules}")

    v0_material = None      # Writer 에게 준 재료 — 후보 역할 "rewrite" 가 다시 쓴다
    v0_path = run_dir / "prompt_v0.txt"
    if a.prompt:
        prompt = Path(a.prompt).read_text(encoding="utf-8")
        if "rewrite" in (a.candidate_roles or ""):
            # v0 을 파일로 받아도 rewrite 는 돌 수 있어야 한다 — 재료는 소스 런의 measured_profile 과
            # 런 디렉토리의 language_profile 로 다시 만들 수 있다. 프로파일 파일을 같이 복사해 두면
            # Profiler 호출도 없고 v0 을 쓸 때와 **같은 재료**가 된다.
            v0_material = writer_material()
    elif v0_path.exists():
        prompt = v0_path.read_text(encoding="utf-8")
        log("[v0] 런 산출물 재사용")
        if a.generate_v0:
            v0_material = writer_material()
    elif a.generate_v0:
        user = v0_material = writer_material()
        out_rules = ad.output_rules(spaced, "source", meaning_in_scoring_rules=True)
        cands = []
        for c in range(a.v0_candidates):
            cand_path = run_dir / f"prompt_v0_cand{c}.txt"
            if a.resume and cand_path.exists():
                pr = cand_path.read_text(encoding="utf-8")
                log(f"[v0] 후보 {c} 파일 재사용")
            else:
                pr = gw.chat(aj.writer_system(spaced, targets), user, max_tokens=16000,
                             reasoning_effort=(None if a.agent_reasoning_effort == "none"
                                               else a.agent_reasoning_effort),
                             purpose="prompt_v0").strip()
                save_usage(run_dir / "iter_00")
                pr = aj.strip_inline_headers(pr)
                pr = aj.replace_section(pr, "[Scoring Rules]", aj.scoring_rules(targets))
                pr = aj.replace_section(pr, "[Output Rules]", out_rules)
                pr = aj.replace_section(pr, "[Examples]", examples_section(v0_examples))
                cand_path.write_text(pr, encoding="utf-8")
            errs = aj.check_skeleton(pr)
            if errs:
                log(f"[v0] 후보 {c} 골격 실패: {errs}")
                continue
            cands.append(pr)
        if not cands:
            log("[stop] v0 후보가 전부 골격 검증에 걸렸다")
            return 2
        if len(cands) == 1:
            prompt = cands[0]
        else:
            # 후보는 판정 집합이 아니라 사례 집합(3분할: train)에서 고른다. 판정 집합에서 고르면
            # 그 v0 의 판정값이 낙관 편향이 돼 이후 개정이 부풀려진 기준선을 넘어야 한다.
            # 2분할 런은 둘이 같은 dev-A 라 편향이 남는다.
            sel_sents, sel_lab, sel_name = (v0_sel if v0_sel
                                            else (case_sents, case_lab, "train") if sep_cases
                                            else (devA, labA, names[0]))
            scored = []
            for c, pr in enumerate(cands):
                _r, _s, h, m = score_prompt(pr, sel_sents, sel_lab, f"v0 후보 {c} ({sel_name})")
                save_usage(run_dir / "iter_00")
                scored.append((st.mean(h.values()), c, pr,
                               m.get("format_pass_rate_no_retry", 0.0),
                               -m.get("first_pass_violations", {}).get("realigned", 0)))
            scored.sort(reverse=True)
            # **잡음 폭 안에서는 H_set 으로 고르지 않는다.** judge33b 에서 후보 둘이 0.5589 / 0.5588
            # 로 갈렸다 — 한 벌 sd 가 0.0039 이므로 그 차이는 sd 의 1/40 이고, 사실상 동전 던지기다.
            # 그 자리에서는 **결정론적인 자로** 고르는 것이 낫고, 1차 포맷 통과율이 그 자를 한다:
            # 재시도(`segment_retry`)를 덜 부르고, 출력 규약을 스스로 지키는 프롬프트다.
            #
            # 재정렬이 Δ 를 편향시키지는 **않는다.** `realign_tags` 는 옮긴 자리가 입력 마커 자리와
            # 하나라도 다르면 포기하므로(`mapped != want`), 성공한 재정렬의 절단 자리와 점수는
            # 원래 출력과 같다. 고치는 것은 주변 글자뿐이다. 그래서 이 자는 "품질을 예측한다" 가
            # 아니라 "잡음이 아니다" 로만 정당하다. 둘 다 같으면 먼저 생성된 후보다.
            band = [x for x in scored if scored[0][0] - x[0] <= V0_TIE_BAND]
            pick = max(band, key=lambda x: (x[3], x[4], -x[1]))
            log(f"[v0] 후보 {sel_name} H_set {[round(x[0], 4) for x in scored]} / "
                f"1차 {[(x[1], x[3], -x[4]) for x in scored]} "
                f"→ 동점대 {[x[1] for x in band]} 중 {pick[1]} 채택")
            prompt = pick[2]
    else:
        log("[stop] --prompt 또는 --generate-v0 가 필요하다")
        return 2
    if not a.score_only:
        v0_path.write_text(prompt, encoding="utf-8")
    v0_len = len(prompt)
    provenance = aj.init_provenance(prompt)
    start = 1
    if state:
        prompt, history, checkpoint = state["prompt"], state["history"], state["checkpoint"]
        # prompt_budget: 길이 상한이 시작 길이에 고정돼 있던 때의 state 는 그 값이 곧 시작 길이다
        v0_len = state.get("v0_len") or state["prompt_budget"]
        provenance = state.get("provenance") or aj.init_provenance(prompt)
        start = state["done"] + 1
        log(f"[resume] 이터 {start} 부터 — 현재 프롬프트 {len(prompt)}자 / 시작 길이 {v0_len}자")
        # 편집 내용이 없는 이력(그 기록 전 코드로 돈 이터)은 pe_edits.json 에서 채운다. 단위 id 는
        # 그때의 프롬프트 기준이라, 마지막 채택 뒤의 이터만 현재 프롬프트로 풀 수 있다.
        last = max((h["iter"] for h in history if h.get("adopted")), default=0)
        for h in history:
            p = run_dir / f"iter_{h['iter']:02d}" / "pe_edits.json"
            if h["iter"] > last and "edits" not in h and p.exists():
                tries = json.loads(p.read_text(encoding="utf-8"))
                h["edits"] = aj.edit_summary(prompt, tries[-1].get("edits"))
    log(f"[길이] 이터당 직전 채택본 +{a.growth_per_iter:.0%}, 천장 "
        f"{int(v0_len * a.growth_ceiling)}자 (시작 {v0_len}자 × {a.growth_ceiling})")

    if a.score_only:
        # 판정 분할(test-A)은 건드리지 않는다 — 잴 것은 최종 분할에서의 프롬프트 두 개뿐이다.
        a.iterations = 0
        cur_rows, cur_sets, cur_h, cur_m = [], {}, {}, {"format_pass_rate": 1.0}
        log("[score-only] 이터를 돌지 않는다 — 최종 분할에서 --prompt 와 prompt_v0.txt 만 잰다")
    else:
        # 재개 때도 다시 잰다 — 현재 프롬프트의 분절은 캐시에 있어 LLM 호출이 없고 QE 만 돈다.
        cur_rows, cur_sets, cur_h, cur_m = score_avg(
            prompt, devA, labA, "iter 0" if start == 1 else f"iter {start - 1} 재개",
            a.baseline_draws, "base")
    # v0 를 생성하지 않고 `--prompt` 로 받은 런은 이 디렉토리를 만든 적이 없다 — 생성 경로만
    # iter_00 을 만들어 두므로, 여기서 보장해야 한다.
    (run_dir / f"iter_{start - 1:02d}").mkdir(parents=True, exist_ok=True)
    (run_dir / f"iter_{start - 1:02d}" / "violations.json").write_text(
        json.dumps({"current": violation_summary(cur_rows)}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    save_usage(run_dir / f"iter_{start - 1:02d}")

    def snapshot_state(done: int) -> None:
        save_state(run_dir / STATE_FILE, done, prompt, history, checkpoint, v0_len,
                   spent_before + gw.usage.snapshot()["cost"], provenance)

    def cur_best() -> str:
        return prompt      # propose() 안에서 `prompt` 는 편집 기반으로 다시 묶이므로 채택본은 이걸로 읽는다

    def propose(it: int, idir: Path, cases: list, budget: int, target: int, timing: dict,
                examples: dict | None = None, loss_bins: dict | None = None,
                base: tuple[str, dict] | None = None, gate_cases: list | None = None,
                inversions: list | None = None, depth: dict | None = None,
                misorder: dict | None = None):
        nonlocal checkpoint
        """Critic → PE → dev-A 채점·판정 한 번. `budget` 은 코드가 검사하는 상한, `target` 은
        모델에게 알리는 목표(상한의 95%, `agents_judge.soft_target`). 채택이면 (개정본, 절단집합, H, 지표, 채택 근거 Δ),
        아니면 None. 반려·기각 어느 쪽이든 돌아와야 뒤의 체크포인트가 돈다 — 반려 경로의
        continue 가 체크포인트를 건너뛰어 judge03·04 는 dev-B 를 한 번도 못 쟀다.

        `base` 가 있으면 후보는 그 본문(근소 기각본, `near_miss_base`)을 편집한다. 아래에서
        `prompt` 는 편집 기반이고 `best` 가 현재 채택본이다 — 판정·dev-B 현재값·예시 제외는 `best`."""
        best = cur_best()
        prompt = best
        base_tag: dict = {}
        prov = provenance

        def add_vs_base(diag: dict, c_h: dict, c_sets: dict) -> dict:
            """기반(근소 기각본)이 있으면 그 대비 차이도 잰다 — 부검이 기반본이 벌어 둔 이득을 새
            편집의 공으로 돌리지 않게(judge13 iter 2: "서술어 보호" 편집이 ≤3 을 올렸다고 썼는데
            기반 대비로는 −0.008). 기반본의 test-A 채점은 캐시라 비용 0."""
            if not base:
                return diag
            _r, b_sets, b_h, _m = score_prompt(prompt, devA, labA, f"iter {it} 기반")
            vb = revision_diagnosis(devA, b_h, c_h, b_sets, c_sets, spaced)
            kb = sorted(set(b_h) & set(c_h))
            boot_b = hset.paired_bootstrap([c_h[k] for k in kb], [b_h[k] for k in kb],
                                           clusters=[i for i, _k in kb])
            diag["vs_base"] = {"delta": boot_b, **{k: vb[k] for k in ("by_bin", "n_worse", "n_better")}}
            log(f"[iter {it}] 기반 대비 Δ {boot_b['mean']:+.4f} [{boot_b['lo']:+.4f}, {boot_b['hi']:+.4f}] "
                f"구간별 {vb['by_bin']}")
            return diag
        if base:
            prompt, nm = base
            d = nm.get("gain") or nm.get("delta")
            base_tag = {"built_on": f"iter {nm['iter']} near miss ({d['mean']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}])"}
            prov = {**provenance, **{u["text"]: {"origin": f"iter {nm['iter']} near miss",
                                                 "adopted_delta": None, "adopted_ci_lo": None,
                                                 "near_miss_delta": round(d["mean"], 4),
                                                 "near_miss_ci_lo": round(d["lo"], 4)}
                                     for u in aj.edit_units(prompt) if u["text"] not in provenance}}
        t0 = time.perf_counter()
        crit_user = {"cases": cases, "prompt": prompt}
        if base_tag:
            crit_user["base"] = base_tag["built_on"] + " — this prompt is that revision; it is measured against the current best"
        if loss_bins:
            crit_user["loss_by_bin"] = loss_bins
        if depth:
            crit_user["rank_depth"] = depth
        if misorder:
            crit_user["misorder_cost"] = misorder
        rej = rejected_by_bin(history)
        if rej:
            crit_user["rejected_by_bin"] = rej
        if inversions:
            # 깊은 구간의 역전 쌍 — 집계값("8위부터 무작위")은 어디를 보라는 말일 뿐이고,
            # 무엇을 쓸지는 두 자리를 가르는 표면 특징에서 나온다.
            crit_user["rank_inversions"] = inversions
        prev = next((h for h in reversed(history) if h.get("diagnosis")), None)
        if prev:        # 직전 개정이 어디서 무너졌는지 Critic 도 본다 — 같은 방향의 반복을 끊는다
            crit_user["last_revision"] = {"iter": prev["iter"], "adopted": prev["adopted"],
                                          "delta": prev.get("gain") or prev.get("delta"),
                                          "edits": prev.get("edits"), **prev["diagnosis"]}
            if aj.is_near_miss(prev) and not a.no_near_miss:
                crit_user["last_revision"]["near_miss"] = True
                log(f"[iter {it}] 직전 개정은 근소 기각(평균 {crit_user['last_revision']['delta']['mean']:+.4f}) "
                    f"— Critic·PE 에 near_miss 로 알린다")
        crit = gw.chat_json(aj.critic_system(a.findings_max),
                            json.dumps(crit_user, ensure_ascii=False),
                            max_tokens=AGENT_MAX_TOKENS, purpose="critic")
        timing["critic"] = round(time.perf_counter() - t0, 1)
        findings = aj.clean_findings(crit, prompt, cap=a.findings_max)
        # **종류별로 최소 하나를 요구한다.** 한쪽이 비면 그 형태를 쓰는 역할이 짝을 못 찾아 아예
        # 돌지 않는다 — judge31 이터 1 은 `check` 하나만 나와 후보가 9개 계획에서 2개로 줄었다.
        # 이유를 적어 비운 것은 그대로 받는다(없는 것을 억지로 내게 하면 잡음이 굳는다).
        if a.critic_both_kinds and findings:
            miss = [k for k, key in (("order", "orders_skipped"), ("check", "checks_skipped"))
                    if not any(f["kind"] == k for f in findings)
                    and not str((crit or {}).get(key) or "").strip()]
            if miss:
                log(f"[iter {it}] Critic 이 {'/'.join(miss)} 형태를 안 냈고 사유도 없다 — 한 번 더")
                t1 = time.perf_counter()
                crit2 = gw.chat_json(
                    aj.critic_system(a.findings_max),
                    json.dumps({**crit_user, "missing_shapes": miss,
                                "your_previous_findings": crit}, ensure_ascii=False),
                    max_tokens=AGENT_MAX_TOKENS, purpose="critic:shape_retry")
                timing["critic"] = round(timing["critic"] + time.perf_counter() - t1, 1)
                f2 = aj.clean_findings(crit2, prompt, cap=a.findings_max)
                got = {f["kind"] for f in f2}
                if f2 and all(k in got for k in miss):
                    crit, findings = crit2, f2
                    log(f"[iter {it}] 두 번째 호출이 {len(findings)}개를 냈다 — "
                        + " ".join(f"{f['kind']}" for f in findings))
                else:
                    log(f"[iter {it}] 두 번째 호출도 {'/'.join(miss)} 를 안 냈다 — 첫 결과로 간다")
        n_raw = len(aj.clean_findings(crit, cap=a.findings_max))
        if n_raw > len(findings):
            log(f"[iter {it}] Critic finding {n_raw - len(findings)}개 버림 — replace 대상이 현재 "
                f"프롬프트에 없다")
        (idir / "critique.json").write_text(json.dumps({"raw": crit, "findings": findings},
                                                       ensure_ascii=False, indent=1),
                                            encoding="utf-8")
        save_usage(idir)
        if not findings:
            log("[iter] Critic 이 쓸 수 있는 finding 을 안 냈다 — 건너뛴다")
            return None

        # ── 유형 홀드아웃 — finding 마다 "그 구문이 있는 문장" 을 train 에서 고른다 ──────────────
        # 후보를 전체 200문장 평균으로 재면 겨냥한 실패가 나머지 문장의 잡음에 희석된다. 유형 안에서
        # 재면 같은 개정의 효과가 몇 배로 보인다. 사례로 쓴 문장은 빼야 편집을 외운 후보가 안 이긴다.
        types: list[dict] = []
        if a.type_select:
            t0 = time.perf_counter()
            case_ids = {c["id"] for c in cases}
            pool = [i for i, s_ in enumerate(case_sents) if s_.id not in case_ids]
            # 경계는 **분류 대상 기준**이다. train 200 을 분모로 쓰면 사례 60 을 뺀 pool 140 에서
            # 의도한 10~35% 가 실제로 14~50% 로 벌어져, 절반을 덮는 유형이 좁히기 없이 통과한다.
            lo_n, hi_n = max(8, int(0.10 * len(pool))), int(0.35 * len(pool))
            for fi, f in enumerate(findings):
                pred = (f.get("type_predicate") or "").strip()
                if not pred:
                    log(f"[iter {it}] finding {fi}: type_predicate 가 없다 — 유형 제외")
                    continue
                got = classify_type(gw, pred, case_sents, pool, type_cache, log_fn=log)
                if len(got) > hi_n:
                    log(f"[iter {it}] finding {fi}: 유형이 너무 넓다 ({len(got)}/{len(pool)}) — "
                        f"좁힌 술어를 한 번 더 받는다")
                    try:
                        nb = gw.chat_json(aj.NARROW_SYSTEM,
                                          json.dumps({"predicate": pred, "matched": len(got),
                                                      "pool": len(pool),
                                                      "diagnosis": f["diagnosis"]},
                                                     ensure_ascii=False),
                                          max_tokens=6000, reasoning_effort="low",
                                          purpose="narrow")
                    except Exception as e:      # noqa: BLE001 — 좁히기 실패는 유형 제외로 끝낸다
                        log(f"[iter {it}] finding {fi}: 좁히기 실패 ({type(e).__name__}) — 제외")
                        continue
                    pred2 = ((nb or {}).get("type_predicate") or "").strip()
                    got = (classify_type(gw, pred2, case_sents, pool, type_cache, log_fn=log)
                           if pred2 else [])
                    pred = pred2 or pred
                if not (lo_n <= len(got) <= hi_n):
                    log(f"[iter {it}] finding {fi}: 유형 크기 {len(got)} 가 [{lo_n}, {hi_n}] 밖 — 제외")
                    continue
                types.append({"finding": fi, "predicate": pred,
                              "ids": got[:a.type_holdout_max], "diagnosis": f["diagnosis"]})
            # 유형 문장에서 **현재 프롬프트가 이미 얼마나 잘하는지**를 남긴다. 유형 Δ 만 찍으면
            # 음수가 나왔을 때 "개정이 나빴다" 와 "그 자리가 이미 높았다" 를 구분할 수 없다.
            for t in types:
                lvl = [v for (i, _k), v in case_h.items() if i in set(t["ids"])]
                t["level"] = round(st.mean(lvl), 5) if lvl else None
            # 대조군 — 술어에 **안 맞는** 문장을 같은 개수 뽑는다. 유형별 1등에 한해 여기서도 재서
            # 손해가 겨냥한 구문에 몰린 것인지 전역인지 가린다. 어느 유형에도 안 들어간 문장만 쓴다.
            taken = {i for t in types for i in t["ids"]}
            rest = [i for i in pool if i not in taken]
            for t in types:
                rng = random.Random(hashlib.sha1(t["predicate"].encode()).hexdigest()[:8])
                t["control"] = sorted(rng.sample(rest, min(len(t["ids"]), len(rest))))
            timing["type"] = round(time.perf_counter() - t0, 1)
            type_cache.flush()
            (idir / "types.json").write_text(json.dumps(types, ensure_ascii=False, indent=1),
                                             encoding="utf-8")
            save_usage(idir)
            log(f"[iter {it}] 유형 {len(types)}/{len(findings)} 통과 — "
                + ", ".join(f"f{t['finding']}:{len(t['ids'])}문장" for t in types))
            for t in types:
                log(f"[iter {it}] 유형 f{t['finding']}: 현재 H_set {t['level']} / train "
                    f"{st.mean(case_h.values()):.4f} / 대조군 {len(t['control'])}문장")
            if not types and not a.type_rank_only:
                log(f"[iter {it}] 쓸 수 있는 유형이 없다 — 이터 건너뜀 (본채점 안 함)")
                history.append({"iter": it, "adopted": False, "reason": ["유형 없음"], **base_tag})
                return None
            if not types:
                # 순위만 쓰는 모드에서 유형이 없다고 이터를 버리면, 거부권을 후보 단위에서 치워 놓고
                # 한 층 위에 남겨 둔 셈이 된다. 순위를 못 매길 뿐이니 후보는 그대로 만들고 앞의
                # --full-score-max 개를 본채점한다. judge19 이터 2 가 이 구멍으로 Critic 값만 쓰고
                # 후보를 하나도 안 만든 채 닫혔다 (술어가 140문장 중 87개를 잡아 너비 검사 탈락).
                log(f"[iter {it}] 쓸 수 있는 유형이 없다 — 순위 없이 앞 "
                    f"{max(1, a.full_score_max)}개를 본채점한다")
            else:
                findings = [findings[t["finding"]] for t in types]

        # PE 는 프롬프트를 다시 쓰지 않고 번호 붙은 단위를 편집한다. 적용과 길이는 코드가 하고,
        # 단위마다 출처(들어온 이터·채택 Δ·Critic 지적 횟수)를 붙여 무엇을 갈아끼울지 고르게 한다.
        # 편집 단위(units)가 두 섹션 본문을 이미 담으므로 프롬프트 전체는 안 준다 — 동결 섹션 중
        # 측정 정의만 붙인다. 이력은 방향 반복을 막는 데 필요한 것만 요약한다.
        # **이미 시도한 삭제를 모은다.** 같은 이터의 앞 후보(`used_deletes` 로 누적)와 이전
        # 이터의 이력을 함께 본다. `edit_summary` 가 `was` 에 지워진 단위 본문 앞 80자를 남겨
        # 두는데, 그 주석이 "id 는 이터마다 다시 매겨지므로 원문 앞부분을 싣는다" 로 이 쓰임을
        # 이미 예견했다. judge23·24 에서 `prune` 이 일곱 번 전부 같은 원칙을 지웠다.
        spent_deletes: set[str] = set()
        for h in history:
            for e in (h.get("edits") or []):
                if isinstance(e, dict) and e.get("kind") == "delete" and e.get("was"):
                    spent_deletes.add(str(e["was"]))

        # `[Scoring Rules]` 는 빼고 units 로만 준다 — 같은 2.9KB 를 두 번 실을 일이 없고, "고정
        # 섹션" 이라는 이름이 `procedure` 역할과 모순된다. 못 고치는 줄은 S 태그 쪽에서 가린다.
        pe_user = {"fixed_sections": {h: aj.section_of(prompt, h) for h in ("[Role]",)},
                   "units": aj.units_with_provenance(prompt, prov),
                   "findings": findings, "history": aj.history_brief(history),
                   "size": aj.size_brief(prompt, target)}
        if base_tag:
            pe_user["base"] = base_tag["built_on"] + " — the units you were given already contain that revision's edit; candidates are measured against the current best prompt"
        if loss_bins:
            pe_user["loss_by_bin"] = loss_bins
        if depth:
            pe_user["rank_depth"] = depth
        if misorder:
            pe_user["misorder_cost"] = misorder
        if examples:
            by_id = {c["id"]: c for c in cases}
            pe_user["labeled_examples"] = [
                {"id": cid, "latency_bin": by_id[cid]["latency_bin"], "gap": by_id[cid]["gap"],
                 "unit": ex} for cid, ex in examples.items() if cid in by_id]

        def record(j, pe_blob, deltas, skipped, draft, cand, note):
            if skipped:
                log(f"[iter {it}] 후보 {j}: PE 편집 {len(skipped)}개 무시: "
                    + "; ".join(f"edit {s['edit']} {s['reason']}" for s in skipped[:3]))
            return {"candidate": j, "edits": (pe_blob or {}).get("edits"), "deltas": deltas,
                    "skipped": skipped, "result_chars": len(draft) if draft else None,
                    "errors": None if cand else note}

        used_examples: set[str] = set()     # 같은 이터에서 앞 후보가 이미 넣은 실측 예시 id

        def resolve(pe_blob, role="free", finding=None, pin=True):
            """`labeled_example` 참조를 실측 예시 본문으로 바꾸고 후보 역할 제약을 건다. 어기거나
            없는 id, 앞 후보가 이미 쓴 id 를 가리킨 편집은 로그에 남기고 뺀다."""
            if not isinstance(pe_blob, dict):
                return pe_blob, []
            # 삭제 편집에 **지워질 단위의 본문**을 붙인다 — `enforce_role` 이 이미 시도한 삭제를
            # 막는 데 쓴다. id 는 이터마다 다시 매겨져 못 쓴다(`C2` 가 매번 다른 원칙이다).
            units_now = {u["id"]: u["text"] for u in aj.edit_units(prompt)}
            for e in (pe_blob.get("edits") or []):
                if isinstance(e, dict) and e.get("op") in ("delete", "replace"):
                    e["_was"] = units_now.get(str(e.get("id")), "")
            pending = pe_blob.get("edits")
            if a.pe_verbatim and finding and pin:
                # **PE 가 쓴 문면을 Critic 문면으로 되돌린다.** 형태(단항/이항)가 역할 배분의
                # 근거인데 judge31 에서 PE 가 그것을 바꿔 썼다.
                pending, pinned = aj.pin_finding_text(pending, finding, role)
                for c in pinned:
                    log(f"[iter {it}] PE 문면을 발견 문면으로 되돌렸다: edit {c['edit']} "
                        f"{c['id']} — PE 가 쓴 것은 {c['was']!r}")
            edits, bad = aj.enforce_role(
                pending, role,
                {"deleted": spent_deletes,
                 # 그 칸의 마지막 한 줄은 지울 수 없다 — 비면 골격이 참조할 것이 없다.
                 "order_units": sum(1 for k in units_now if k.startswith("O"))})
            if finding:
                edits, bad_k = aj.enforce_kind(edits, finding.get("kind"),
                                               "[Order Principles]" in prompt, role)
                bad += bad_k
            if examples:
                # 앞 후보가 쓴 예시는 없는 것으로 친다 — judge12 iter 1 은 후보 넷 중 셋이 같은
                # 예시(E4 → en_us_738)를 넣었다. 제약이 강한 역할은 선택지가 그것뿐이었다.
                offered = {k: v for k, v in examples.items() if k not in used_examples}
                edits, bad2 = aj.resolve_labeled_examples(edits, offered)
                bad += bad2
                used_examples.update(str(e["labeled_example"]) for e in edits
                                     if e.get("labeled_example"))
            # 같은 이터의 뒤 후보가 같은 원칙을 또 지우지 못하게 누적한다.
            spent_deletes.update(str(e["_was"]) for e in edits
                                 if e.get("op") in ("delete", "replace") and e.get("_was"))
            for b in bad:
                log(f"[iter {it}] PE 편집 무시: edit {b['edit']} {b['reason']}")
            return {**pe_blob, "edits": edits}, bad

        def revise(j: int, extra: dict, role: str = "free", finding=None):
            """PE 호출 한 번. 길이만 넘으면 편집별 실측 증감과 초과량을 붙여 한 번 되돌린다 — 콜
            하나가 이터레이션 하나를 통째로 날리는 것보다 싸다. (pe, cand, note, deltas, tries)."""
            t0 = time.perf_counter()
            pe = gw.chat_json(aj.engineer_system(target, len(prompt)),
                              json.dumps({**pe_user, **extra}, ensure_ascii=False),
                              max_tokens=AGENT_MAX_TOKENS, purpose="engineer")
            timing["engineer"] = round(timing.get("engineer", 0) + time.perf_counter() - t0, 1)
            save_usage(idir)
            pe, bad = resolve(pe, role, finding)
            cand, note, draft, deltas, skipped = aj.parse_edits(
                pe, prompt, budget, aj.unfreezes(role))
            tries = [record(j, pe, deltas, skipped, draft, cand, note)]
            # 역할 제약을 어겨 **편집이 하나도 안 남으면** 그 후보는 원본 그대로가 되어 슬롯을
            # 통째로 버린다. 사유를 돌려주고 한 번 더 시킨다 — 길이 초과에 이미 있는 경로와 같다.
            # `fallback` 은 조건이 둘(금지문 아님 + 비교어 있음)이라 특히 걸리기 쉽다.
            if not pe.get("edits") and bad:
                log(f"[iter {it}] 후보 {j}: 역할 '{role}' 제약으로 편집이 전부 빠졌다 — "
                    f"사유를 알려 한 번 더: {'; '.join(b['reason'] for b in bad[:3])}")
                pe = gw.chat_json(aj.engineer_system(target, len(prompt)),
                                  json.dumps({**pe_user, **extra,
                                              "your_previous_edits": tries[0]["edits"],
                                              "rejected_because": [b["reason"] for b in bad]},
                                             ensure_ascii=False),
                                  max_tokens=AGENT_MAX_TOKENS, purpose="engineer:role_retry")
                save_usage(idir)
                pe, _bad2 = resolve(pe, role, finding)
                cand, note, draft, deltas, skipped = aj.parse_edits(
                    pe, prompt, budget, aj.unfreezes(role))
                tries.append(record(j, pe, deltas, skipped, draft, cand, note))
            if cand is None and aj.only_too_long(note):
                fb = aj.edit_feedback(draft, prompt, target, deltas)
                log(f"[iter {it}] 후보 {j}: PE 편집 결과 {len(draft)} > 목표 {target} — 편집별 증감과 "
                    f"초과량 {fb['over_by']}자({fb['over_by_words']}단어)를 알려 한 번 더")
                pe = gw.chat_json(aj.engineer_system(target, len(prompt)),
                                  json.dumps({**pe_user, **extra,
                                              "your_previous_edits": tries[0]["edits"],
                                              "size_feedback": fb}, ensure_ascii=False),
                                  max_tokens=AGENT_MAX_TOKENS, purpose="engineer:shorten")
                save_usage(idir)
                pe, _bad3 = resolve(pe, role, finding, pin=False)
                cand, note, draft, deltas, skipped = aj.parse_edits(
                    pe, prompt, budget, aj.unfreezes(role))
                tries.append(record(j, pe, deltas, skipped, draft, cand, note))
            return pe, cand, note, deltas, tries

        # 개정 후보를 여러 개 받아 dev-A 앞부분에서 선별한다. 채택률이 낮을 때(judge03~08: 측정된
        # 개정 9건 중 채택 1건) 후보 하나에 이터레이션 전체 채점($3)을 거는 것보다, 셋을 50문장에서
        # 재고($1씩) 가장 나은 것만 끝까지 재는 쪽이 같은 돈으로 채택 기회를 늘린다. 선별에 쓴
        # 문장은 캐시에 남아 본채점에서 다시 돈이 들지 않는다.
        cands, all_tries = [], []
        def rewrite(j: int):
            """Writer 가 [Core Principles]·[Examples] 를 새로 쓴다 — 편집이 아니라 큰 보폭.

            Writer 표본 사이의 H_set 차이(judge07 0.5918/0.5764, judge08 0.6027/0.5932)가 편집 하나의
            효과(±0.01)보다 크다. 편집만으로는 그 분산을 못 쓰므로 후보 하나는 다시 쓴 것으로 채운다.
            동결 섹션은 현재 프롬프트 것으로 되돌려 붙인다."""
            lessons = [h["diagnosis"]["lesson"] for h in history
                       if h.get("diagnosis") and h["diagnosis"].get("lesson")]
            cpw = aj.chars_per_word(prompt)
            msg = (v0_material
                   + "\n\nA prompt of this skeleton is already in use (below) and was measured. "
                     "Write a NEW prompt with the same skeleton whose [Core Principles] and "
                     "[Examples] address the critic findings and the lessons below. Do not copy "
                     "the current [Core Principles] — rewrite the judgements from scratch in your "
                     "own structure; you may keep examples that still teach the right ranking. "
                     f"The whole prompt must stay under {target} characters "
                     f"(about {int(target / cpw)} words; it is {len(prompt)} now).\n\n"
                   + f"Current prompt:\n{prompt}\n\n"
                   + f"Critic findings:\n{json.dumps(findings, ensure_ascii=False)}\n\n"
                   + ("Measured failures — the cut set the CURRENT prompt produced (‖) against the "
                      "best cut set an exhaustive search found, on the sentences where the loss is "
                      "largest. Read them as what your ranking has to change:\n"
                      + failure_brief(cases) + "\n\n" if cases else "")
                   + (f"Lessons from measured revisions:\n{json.dumps(lessons, ensure_ascii=False)}\n\n"
                      if lessons else "")
                   + ("Measured examples you may paste into [Examples] verbatim:\n"
                      + "\n\n".join(examples.values()) + "\n" if examples else ""))
            def write(user_msg: str) -> str:
                t0 = time.perf_counter()
                pr = gw.chat(aj.writer_system(spaced, targets), user_msg, max_tokens=16000,
                             reasoning_effort=(None if a.agent_reasoning_effort == "none"
                                               else a.agent_reasoning_effort),
                             purpose="rewrite").strip()
                timing["engineer"] = round(timing.get("engineer", 0) + time.perf_counter() - t0, 1)
                save_usage(idir)
                pr = aj.strip_inline_headers(pr)
                for h in aj.FROZEN:
                    pr = aj.replace_section(pr, h, aj.section_of(prompt, h))
                # 사람이 골격에 넣은 칸은 Writer 가 다시 쓰지 않는다 — 복원하지 않으면 통째로
                # 사라지고(Writer 는 그 칸을 모른다) 골격이 참조하는 자리가 비게 된다.
                for h in aj.ALWAYS_OPTIONAL:
                    if h in prompt:
                        pr = aj.replace_section(pr, h, aj.section_of(prompt, h))
                return pr

            pr = write(msg)
            blob = {"prompt": pr, "changelog": ["rewrite: [Core Principles]/[Examples] 를 새로 씀"]}
            cand, note = aj.parse_prompt(blob, prompt, budget)
            deltas = [{"edit": 0, "op": "rewrite", "id": None, "kind": "rewrite",
                       "chars_change": len(pr) - len(prompt)}]
            pe = {"changelog": blob["changelog"], "edits": [], "rewrite": True}
            tries = [{"candidate": j, "edits": [], "rewrite": True, "deltas": deltas,
                      "skipped": [], "result_chars": len(pr), "errors": None if cand else note}]
            # Writer 는 못 줄인다 — 통째로 다시 쓰므로 길이가 손을 떠난다(judge09 iter 2: 상한
            # 9,050 에 11,463자, judge10 iter 1: 8074 → 재시도 8104). PE 는 단위별 실측을 받아
            # 편집만 내니 맞춘다. 초안을 단위로 쪼개 PE 에게 줄이게 하되, 한 패스에 15% 남짓만
            # 깎이므로(judge10 iter 2: 9748 → 8339) 맞을 때까지 SHORTEN_PASSES 번 반복한다.
            # 축소 패스에서 insert 는 코드가 뺀다(같은 곳: 삭제 −1596 에 예시 추가 +262).
            passes = 0
            while cand is None and aj.only_too_long(note) and passes < SHORTEN_PASSES:
                passes += 1
                over_w = -(-(len(pr) - target) // int(cpw))
                log(f"[iter {it}] 후보 {j}: rewrite 결과 {len(pr)} > 목표 {target} — 축소 패스 "
                    f"{passes}/{SHORTEN_PASSES}: 초안을 단위로 쪼개 PE 에게 초과량 "
                    f"{len(pr) - target}자({over_w}단어)를 줄이게 한다")
                t0 = time.perf_counter()
                pe2 = gw.chat_json(aj.engineer_system(target, len(pr)),
                                   json.dumps(aj.shorten_user(pr, target, findings),
                                              ensure_ascii=False),
                                   max_tokens=AGENT_MAX_TOKENS, purpose="engineer:shorten")
                timing["engineer"] = round(timing.get("engineer", 0) + time.perf_counter() - t0, 1)
                save_usage(idir)
                pe2, _bad4 = resolve(pe2, aj.SHORTEN_ROLE)
                cand, note, draft, pe_deltas, skipped = aj.parse_edits(pe2, pr, budget)
                tries.append(record(j, pe2, pe_deltas, skipped, draft, cand, note))
                deltas = deltas + pe_deltas
                pe = {"changelog": pe["changelog"]
                      + [str(x) for x in ((pe2 or {}).get("changelog") or [])],
                      "edits": [], "rewrite": True}
                if draft:
                    pr = draft      # 다음 패스는 줄어든 초안에서 잇는다
            return pe, cand, note, deltas, tries

        roles = [r.strip() for r in (a.candidate_roles or "").split(",") if r.strip()]
        # 후보 배정 — 기본은 역할·finding 각각 독립 나머지 연산이라 주기가 LCM(역할수, finding수) 이다.
        # 역할 4 · finding 4 면 4쌍이 반복돼 후보를 늘려도 실질 시도가 안 는다. `--candidates-cross`
        # 는 (역할 × finding) 을 한 번씩 전부 돌려 시도 폭을 finding 수에 맞춰 자동으로 맞춘다.
        n_find = max(1, len(findings))
        cross = bool(a.candidates_cross and roles and findings)
        kinds = [str(f.get("kind") or "check") for f in findings]
        plan = candidate_plan(roles, n_find, cross, a.pe_candidates, a.candidates_cap, kinds)
        if cross:
            log(f"[iter {it}] 후보 {len(plan)} — finding {n_find}개 "
                f"({'/'.join(kinds)}) 와 역할 {len(roles)}개를 형태가 맞는 짝만 곱했다: "
                + ", ".join(f"{r}×f{i}" for r, i in plan))
        # 후보 생성은 **순차여야 한다.** 앞 후보의 편집을 `sibling_candidates` 로 보여주고 앞
        # 후보가 쓴 실측 예시를 `used_examples` 로 빼는데, 그게 다양성 장치다 — judge12 iter 1 은
        # 후보 넷 중 셋이 같은 예시(E4 → en_us_738)를 넣었고 그래서 이 의존을 넣었다. 병렬로
        # 바꾸면 8콜 90초가 15초로 줄지만 그 장치를 잃는다.
        for j, (role, f_idx) in enumerate(plan):
            extra = {}
            if role == "rewrite" and not v0_material:
                log(f"[iter {it}] 후보 {j}: rewrite 역할인데 v0 재료가 없다(--prompt 로 시작) — free 로")
                role = "free"
            if role not in ("free", "rewrite"):
                extra["constraint"] = role
            if findings and role != "induce":
                extra["primary_finding"] = findings[f_idx]["diagnosis"]
                extra["primary_finding_kind"] = findings[f_idx].get("kind") or "check"
                # 이항 발견에는 **압축 전 쌍**을 같이 준다 — 비교문을 쓰려면 어느 두 자리를
                # 갈라야 하는지가 필요한데 진단 한 문장에는 그것이 없다. 단항 발견에는 넣지
                # 않는다(토큰만 늘고 쓸 자리가 없다).
                if inversions and extra["primary_finding_kind"] == "order":
                    extra["rank_inversions"] = inversions
            if role == "induce":
                # **진단을 주지 않는다.** 이 역할의 요지가 "압축 전 증거만 보고 귀납한다" 이므로
                # Critic 의 한 문장을 같이 주면 그 압축에 끌려가고, 결과가 나와도 귀납 덕인지
                # 진단 덕인지 가릴 수 없다. 대신 사례를 직접 준다.
                # **압축 전 자료를 준다.** 다른 역할은 Critic 이 사례를 한 문장으로 줄인
                # `primary_finding` 만 읽는데, 잘못된 일반화가 그 압축에서 들어온다. 여기서는
                # 어디를 잘랐고 어디를 잘라야 했는지를 그대로 보여주고 공통점을 찾게 한다.
                # 전부 주면 토큰이 커지므로 `--induce-cases` 개만 준다 — 구간 비중을 사례
                # 배분과 맞추고, 프롬프트가 예시로 든 문장은 뺀다(`induce_picks` 주석 참고).
                n_ind = sum(1 for r, _f in plan if r == "induce")
                picked = induce_picks(cases, example_sentences(prompt, case_sents),
                                      a.induce_cases, part=f_idx, parts=max(1, n_ind))
                if picked:
                    log(f"[iter {it}] 후보 {j} induce 사례 {len(picked)}개 — 구간 "
                        + " ".join(f"{b}:{sum(1 for c in picked if c['latency_bin'] == b)}"
                                   for b in ("≤3", "≤5", "≤7", "≤10", "≤99")
                                   if any(c["latency_bin"] == b for c in picked))
                        + f" / gap {min(c['gap'] for c in picked):.3f}~"
                          f"{max(c['gap'] for c in picked):.3f}")
                # **조각 번역을 함께 준다.** 이것이 "왜 그 자리가 나쁜가" 의 증거다 — 조각을 따로
                # 번역해 이어붙인 결과를 보면 모델이 원문만 보고 추측하지 않는다. 건당 356 → 750
                # 토큰이 되지만 PE 입력에 여유가 크다(실측: 호출 하나 37,700 토큰 중 사례가 2,851,
                # 나머지 34,849 가 시스템·단위·이력·예시다. 컨텍스트 400K).
                extra["measured_cases"] = [
                    {"id": c["id"], "latency_bin": c["latency_bin"], "gap": c["gap"],
                     "current_cuts": (c.get("policy") or {}).get("text"),
                     "current_pieces": (c.get("policy") or {}).get("pieces"),
                     "target_cuts": (c.get("target") or {}).get("text"),
                     "target_pieces": (c.get("target") or {}).get("pieces"),
                     "cuts_the_target_drops": [d for d in (c.get("diff") or {}).get("dropped", [])][:4],
                     "cuts_the_target_adds": [d for d in (c.get("diff") or {}).get("added", [])][:4]}
                    for c in picked]
            if cands:
                extra["sibling_candidates"] = [
                    {"changelog": c[2], "edits": aj.edit_summary(prompt, (c[0] or {}).get("edits"))}
                    for c in cands if c[1]]
                if pe_user.get("labeled_examples"):
                    extra["labeled_examples"] = [x for x in pe_user["labeled_examples"]
                                                 if x["id"] not in used_examples]
            if role != "free":
                log(f"[iter {it}] 후보 {j} 역할 {role}")
            f_cur = (findings[f_idx] if findings and role not in ("induce", "rewrite")
                     else None)
            pe, cand, note, deltas, tries = (rewrite(j) if role == "rewrite"
                                             else revise(j, extra, role, f_cur))
            cands.append((pe, cand, note, deltas))
            all_tries += tries
            if cand is None:
                log(f"[iter {it}] 후보 {j}: 반려 — {'; '.join(map(str, note))}")
        (idir / "pe_edits.json").write_text(json.dumps(all_tries, ensure_ascii=False, indent=1),
                                            encoding="utf-8")
        valid = [(j, *c) for j, c in enumerate(cands) if c[1]]
        keep, dup = aj.dedupe_candidates([(v[0], v[2]) for v in valid])
        for j, same in dup:
            log(f"[iter {it}] 후보 {j}: 후보 {same} 와 본문이 같다 — 선별에서 뺀다")
        valid = [v for v in valid if v[0] in keep]
        if not valid:
            pe, _c, note, _d = cands[-1]
            log(f"[iter] PE 출력 반려: {note} / 누적 ${gw.usage.snapshot()['cost']:.2f}")
            history.append({"iter": it, "adopted": False, "reason": note,
                            "edits": aj.edit_summary(prompt, (pe or {}).get("edits")), **base_tag})
            return None
        for j, _pe, cand, _note, deltas in valid:
            log(f"[iter {it}] 후보 {j}: PE 편집 {len(deltas)}개 → {len(cand)}자 / 상한 {budget}자 "
                f"(목표 {target}자)")

        if a.type_select and types and valid:
            # ── 유형 Δ 선별 — 후보를 **자기 유형 문장에서만** 재고, 유형마다 1등을 뽑아 합친다 ──
            # 게이트(오라클 거리)는 보정 3점에서 본채점 순위와 무관했다(+0.0036/+0.0089/−0.0061,
            # 전부 본채점 CI 반폭보다 작다). 여기서는 같은 자(H_set 쌍체 Δ)를 쓰되 재는 문장을
            # 그 개정이 겨냥한 구문으로 좁힌다 — 효과가 희석되지 않는다.
            t0 = time.perf_counter()
            scored: list[dict] = []
            for j, pe_j, cand_j, note_j, _d in valid:
                (idir / f"candidate_{j}.txt").write_text(cand_j, encoding="utf-8")
                ty = types[plan[j][1]] if plan[j][1] < len(types) else types[0]
                ids = ty["ids"]
                tsents = [case_sents[i] for i in ids]
                tlab = {t: [per[i] for i in ids] for t, per in case_lab.items()}
                rows_t, _m = segment_rows(cand_j, tsents, tlab)
                seg_cache.flush()
                t_sets = policy_sets(rows_t, tsents, spaced, min_gap, a.min_chunk, a.max_k)
                t_h = {(ids[n], k): v for (n, k), v in hset_of(tsents, tlab, t_sets).items()}
                for tr in translators.values():
                    if tr.cache is not None:
                        tr.cache.flush()
                keys = sorted(k for k in set(t_h) & set(case_h))
                boot = hset.paired_bootstrap([t_h[k] for k in keys], [case_h[k] for k in keys],
                                             clusters=[i for i, _k in keys])
                log(f"[iter {it}] 후보 {j} (유형 f{ty['finding']}, {len(ids)}문장) 유형 Δ "
                    f"{boot['mean']:+.4f} [{boot['lo']:+.4f}, {boot['hi']:+.4f}] 짝 {boot['n']}")
                scored.append({"candidate": j, "type": ty["finding"], "delta": boot,
                               "pe": pe_j, "cand": cand_j, "note": note_j})
            timing["type_score"] = round(time.perf_counter() - t0, 1)
            # 1등은 **채택을 가르는 자와 같은 자**로 뽑는다. 유형 홀드아웃은 25문장이라 se 가 200문장의
            # 약 2.8배(≈0.021)다 — 하한 > 0 은 유형 안 평균이 +0.04 를 넘어야 서는 문턱이라, 선별이
            # 아니라 차단으로 작동한다. 선별은 관대하게, 판정은 test-A 200문장 본채점이 엄격하게.
            sel = pick_key(a.adopt_rule)
            kname = "평균" if a.adopt_rule == "mean" else "하한"
            winners = []
            for ty in types:
                same = [x for x in scored if x["type"] == ty["finding"]]
                if not same:
                    continue
                top = max(same, key=lambda x: sel(x["delta"]))
                # **1등만** 대조군에서 한 번 더 잰다. 후보 전부를 재면 선별 시간이 두 배가 되는데,
                # 가려야 하는 것은 "가장 나은 후보조차 유형에서 손해인가, 아니면 어디서든 손해인가"다.
                ctrl = (ty.get("control") or []) if a.type_control else []
                if ctrl:
                    csents = [case_sents[i] for i in ctrl]
                    clab = {t_: [per[i] for i in ctrl] for t_, per in case_lab.items()}
                    crows, _cm = segment_rows(top["cand"], csents, clab)
                    seg_cache.flush()
                    csets = policy_sets(crows, csents, spaced, min_gap, a.min_chunk, a.max_k)
                    ch = {(ctrl[n], k): v for (n, k), v in hset_of(csents, clab, csets).items()}
                    for tr in translators.values():
                        if tr.cache is not None:
                            tr.cache.flush()
                    ckeys = sorted(k for k in set(ch) & set(case_h))
                    cboot = hset.paired_bootstrap([ch[k] for k in ckeys],
                                                  [case_h[k] for k in ckeys],
                                                  clusters=[i for i, _k in ckeys])
                    top["control_delta"] = cboot
                    log(f"[iter {it}] 유형 f{ty['finding']} 1등 후보 {top['candidate']}: "
                        f"유형 Δ {top['delta']['mean']:+.4f} vs 대조 Δ {cboot['mean']:+.4f} "
                        f"[{cboot['lo']:+.4f}, {cboot['hi']:+.4f}] 짝 {cboot['n']} — "
                        + ("손해가 유형에 몰렸다" if cboot["mean"] - top["delta"]["mean"] > 0.01
                           else "대조군도 같이 내려간다"))
                if sel(top["delta"]) > 0:
                    winners.append(top)
                else:
                    log(f"[iter {it}] 유형 f{ty['finding']}: 1등 후보 {top['candidate']} {kname} "
                        f"{sel(top['delta']):+.4f} ≤ 0 — 합치기에서 제외")
            # 대조 Δ 까지 채운 뒤에 저장한다 — 먼저 쓰면 control_delta 가 파일에 안 남는다
            (idir / "type_scores.json").write_text(
                json.dumps([{k: v for k, v in x.items() if k not in ("pe", "cand")}
                            for x in scored], ensure_ascii=False, indent=1), encoding="utf-8")
            if a.type_rank_only:
                # 순위만 쓴다 — 유형 Δ 가 음수여도 상위 K개를 본채점으로 올린다. 유형 홀드아웃은
                # 후보를 줄 세우는 데까지만 쓸 수 있고(judge17 이터 1 에서 구간이 갈렸다), 통과를
                # 가르는 자로는 못 쓴다(judge17·18 이터 다섯에서 통과 0). 판정은 test-A 200 이 한다.
                winners = sorted(scored, key=lambda x: -sel(x["delta"]))[:max(1, a.full_score_max)]
            else:
                winners = sorted(winners, key=lambda x: -sel(x["delta"]))[:a.merge_max]
            if not winners:
                log(f"[iter {it}] 유형 Δ {kname} > 0 인 후보가 없다 — 본채점 없이 기각")
                history.append({"iter": it, "adopted": False, "reason": ["유형 Δ 미달"],
                                "type_scores": [{"candidate": x["candidate"], "type": x["type"],
                                                 "delta": x["delta"]} for x in scored],
                                **base_tag})
                (idir / "result.json").write_text(json.dumps(history[-1], ensure_ascii=False,
                                                             indent=1), encoding="utf-8")
                return None
            best_single = winners[0]
            to_full = [best_single["candidate"]]
            log(f"[iter {it}] 유형별 1등 {len(winners)}개 — "
                + ", ".join(f"후보 {w['candidate']}(f{w['type']}, {w['delta']['mean']:+.4f})"
                            for w in winners))
            if a.type_rank_only:
                to_full = [w["candidate"] for w in winners]
                log(f"[iter {it}] 유형 Δ 순위로 본채점 {len(to_full)}개 — 합치지 않고 따로 잰다")
            elif len(winners) > 1:
                m_cand, m_note, m_deltas, m_drop = merge_winners(prompt, winners, budget)
                if m_cand:
                    MERGED = 900
                    valid.append((MERGED, {"changelog": m_note,
                                           "edits": [e for w in winners
                                                     for e in (w["pe"] or {}).get("edits") or []]},
                                  m_cand, m_note, m_deltas))
                    to_full = [MERGED, best_single["candidate"]]
                    (idir / "candidate_merged.txt").write_text(m_cand, encoding="utf-8")
                    log(f"[iter {it}] 합본 {len(winners)}개 편집 → {len(m_cand)}자"
                        + (f" (중복 단위 {m_drop} 버림)" if m_drop else "")
                        + " — 합본과 단일 최고를 둘 다 본채점한다")
                else:
                    log(f"[iter {it}] 합치기 실패({m_note}) — 단일 최고만 본채점")
            _j, pe, cand, note, deltas = next(v for v in valid if v[0] == to_full[0])
        elif a.type_select and a.type_rank_only and valid:
            # 유형이 하나도 안 선 이터 — 순위 없이 앞에서부터 올린다. 역할 순서가 곧 우선순위라
            # --candidate-roles 의 앞쪽에 기대가 큰 역할을 둔다.
            to_full = [v[0] for v in valid[:max(1, a.full_score_max)]]
            log(f"[iter {it}] 유형 없이 본채점 {len(to_full)}개 — 후보 "
                + ", ".join(str(j) for j in to_full))
            _j, pe, cand, note, deltas = next(v for v in valid if v[0] == to_full[0])
        elif a.case_gate and gate_cases and len(valid) > 1:
            # 사례 게이트 — 유료 선별(dev-A 50문장 재채점, 후보마다 k_samples 3벌)을 대신한다.
            #
            # **현재 프롬프트 대비로 재면 안 된다.** 사례는 현재 프롬프트가 못한 자리로 뽑은 것이라
            # 그 문장에서 현재 값은 아래로 치우쳐 있다(선택 + 분절 잡음). 아무것도 안 바꾼 후보도
            # 양수로 나온다 — judge05 가 걸린 함정이고, 선별이 순위를 못 매기는 이유다
            # (de OOM 재개 전후로 같은 후보 0 이 +0.0088 → −0.0050 으로 뒤집혔다).
            # 그래서 **결정론적인 오라클과의 거리**로 후보끼리만 줄 세운다. 나아졌는지는 본채점이 본다.
            t0 = time.perf_counter()
            gkeys = {c["id"]: c for c in gate_cases}
            gidx = [i for i, s in enumerate(case_sents) if s.id in gkeys]
            gsents = [case_sents[i] for i in gidx]
            glab = {t: [per[i] for i in gidx] for t, per in case_lab.items()}
            want = {(n, gkeys[s.id]["cuts"]) for n, s in enumerate(gsents)}
            with ThreadPoolExecutor(max_workers=len(valid)) as ex:
                futs = {j: ex.submit(evaluate, gw, cand_j, gsents, glab, spaced, min_gap, t_grid,
                                     seg_cache, a.workers, a.batch_size, seg_effort, 1)
                        for j, _pe, cand_j, _n, _d in valid}
                grows = {j: f.result() for j, f in futs.items()}
            seg_cache.flush()
            gated = []
            for j, pe_j, cand_j, note_j, deltas_j in valid:
                (idir / f"candidate_{j}.txt").write_text(cand_j, encoding="utf-8")
                g_sets = policy_sets(grows[j][0], gsents, spaced, min_gap, a.min_chunk, a.max_k)
                g_h = hset_of(gsents, glab, g_sets)
                for tr in translators.values():
                    if tr.cache is not None:
                        tr.cache.flush()
                # 본채점과 같은 방어 — 예시로 프롬프트에 박힌 문장은 정답을 보고 푸는 셈이다.
                # 게이트 문장은 이터 간에 돌아올 수 있어(--case-exclude bin) 이 경로가 열려 있다.
                gex = {n for n, s_ in enumerate(gsents)
                       if n in example_sentences(cand_j, gsents) | example_sentences(prompt, gsents)}
                got = sorted(k for k in set(g_h) & want if k[0] not in gex)
                dist = st.mean(ora_hC[(gidx[n], kk)] - g_h[(n, kk)] for n, kk in got) if got else 9.9
                log(f"[iter {it}] 후보 {j} 게이트 오라클 거리 {dist:+.4f} (짝 {len(got)}/{len(want)})")
                gated.append({"candidate": j, "oracle_dist": round(dist, 4), "pairs": len(got),
                              "chars": len(cand_j), "changelog": note_j})
            timing["gate"] = round(time.perf_counter() - t0, 1)
            best_j = min(gated, key=lambda x: x["oracle_dist"])["candidate"]
            (idir / "gate.json").write_text(
                json.dumps({"n_sentences": len(gsents), "ids": [s.id for s in gsents],
                            "best": best_j, "candidates": gated}, ensure_ascii=False, indent=1),
                encoding="utf-8")
            log(f"[iter {it}] 게이트 통과 후보 {best_j} — 본채점으로")
            for sc, (j, pe_j, _c, note_j, _d) in zip(gated, valid):
                if j != best_j:
                    history.append({"iter": it, "candidate": j, "adopted": False,
                                    "screened_out": True, "oracle_dist": sc["oracle_dist"],
                                    "changelog": note_j,
                                    "edits": aj.edit_summary(prompt, (pe_j or {}).get("edits")),
                                    **base_tag})
            _j, pe, cand, note, deltas = next(v for v in valid if v[0] == best_j)
            to_full = [best_j]
            if a.gate_calibrate and it % a.gate_calibrate == 0 and len(gated) > 1:
                # 게이트 보정 — 꼴찌도 200문장에서 잰다. 이걸 안 하면 "게이트가 못 골라서 기각"
                # 인지 "고를 게 없어서 기각" 인지 영영 구별할 수 없다. 꼴찌가 실제로 더 나으면
                # 아래 판정이 그쪽을 고른다(하한 최고).
                worst_j = max(gated, key=lambda x: x["oracle_dist"])["candidate"]
                to_full.append(worst_j)
                log(f"[iter {it}] 게이트 보정 — 꼴찌 후보 {worst_j}"
                    f"(거리 {max(g['oracle_dist'] for g in gated):+.4f})도 본채점한다")
        elif len(valid) > 1 and a.screen_n > 0:
            # 선별 문장은 이터마다 돌려 뽑는다. 늘 같은 앞 50문장이면 거기에 맞는 후보가 이터마다
            # 살아남고, 그 50이 든 본판정 Δ 도 그만큼 부풀려진다(선택 편향). 현재 프롬프트 값은
            # 판정 집합 전체에 있으므로 어느 부분집합이든 비용이 같다.
            idx = screen_indices(len(devA), a.screen_n, it)
            sub = [devA[i] for i in idx]
            labS = {t: [per[i] for i in idx] for t, per in labA.items()}
            t0 = time.perf_counter()
            # 분절(API 대기)은 후보들을 동시에, GPU 채점(번역·QE)은 차례로. 선별 한 후보에 23분이
            # 걸렸는데(judge08 iter 3) 대부분이 API 왕복 대기라 셋을 겹치면 그 시간에 다 끝난다.
            with ThreadPoolExecutor(max_workers=len(valid)) as ex:
                futs = {j: ex.submit(segment_rows, cand_j, sub, labS)
                        for j, _pe, cand_j, _n, _d in valid}
                rows_m = {j: f.result() for j, f in futs.items()}
            screened = []
            for j, pe_j, cand_j, note_j, deltas_j in valid:
                (idir / f"candidate_{j}.txt").write_text(cand_j, encoding="utf-8")
                _r, s_s, h_s, _m = score_prompt(cand_j, sub, labS, f"iter {it} 후보 {j} 선별",
                                                rows_m=rows_m[j])
                s_s = {(idx[i], k): v for (i, k), v in s_s.items()}     # 판정 집합 인덱스로
                h_s = {(idx[i], k): v for (i, k), v in h_s.items()}
                excl = (example_sentences(best, devA) | example_sentences(prompt, devA)
                        | example_sentences(cand_j, devA))
                keys = sorted(k for k in set(h_s) & set(cur_h) if k[0] not in excl)
                boot = hset.paired_bootstrap([h_s[k] for k in keys], [cur_h[k] for k in keys],
                                             clusters=[i for i, _k in keys])
                log(f"[iter {it}] 후보 {j} 선별 Δ {boot['mean']:+.4f} [{boot['lo']:+.4f}, "
                    f"{boot['hi']:+.4f}] 짝 {boot['n']} / 문장 {boot['n_clusters']}")
                screened.append({"candidate": j, "delta": boot, "chars": len(cand_j),
                                 "changelog": note_j, "h": h_s, "sets": s_s})
            timing["screen"] = round(time.perf_counter() - t0, 1)
            best_sc = max(screened, key=lambda x: x["delta"]["mean"])
            best_j = best_sc["candidate"]
            (idir / "screen.json").write_text(
                json.dumps({"n_sentences": len(sub), "indices": idx, "best": best_j,
                            "candidates": [{k: v for k, v in sc.items() if k not in ("h", "sets")}
                                           for sc in screened]}, ensure_ascii=False, indent=1),
                encoding="utf-8")
            for sc, (j, pe_j, _c, note_j, _d) in zip(screened, valid):
                if j != best_j:   # 선별에서 진 후보도 이력에 남긴다 — PE 가 같은 방향을 또 내지 않게
                    history.append({"iter": it, "candidate": j, "adopted": False,
                                    "screened_out": True, "delta": sc["delta"],
                                    "changelog": note_j,
                                    "edits": aj.edit_summary(prompt, (pe_j or {}).get("edits")),
                                    **base_tag})
            _j, pe, cand, note, deltas = next(v for v in valid if v[0] == best_j)
            # 선별 평균이 양수인 후보를 Δ 순으로 --full-score-max 개까지 본채점한다. judge12 는
            # 5이터 전부 후보 CI(±0.025)가 겹쳐 선별 1등이 잡음으로 정해졌고, 1등 셋이 본채점에서
            # 전부 음수였다(사이드 채점한 2등도 음수). 200문장이 고르게 한다.
            # **`평균 > 0` 은 본채점을 다 감당 못 할 때만 쓴다.** 선별 50문장의 반폭은 실측
            # 0.038 로 가르려는 효과(0.008)의 다섯 배다 — 부호는 동전 던지기다. judge20 iter 1
            # 에서 prune −0.0010 / rewrite −0.0077 이 그렇게 잘리고 narrow_rule +0.0050 하나만
            # 남았다(셋 다 CI 가 0 을 품는다). 예산이 후보 수만큼 있으면 전부 본채점한다.
            afford = a.full_score_max >= len(screened)
            to_full = [sc["candidate"] for sc in sorted(screened, key=lambda x: -x["delta"]["mean"])
                       if afford or sc["delta"]["mean"] > 0][:max(1, a.full_score_max)]
            if afford:
                log(f"[iter {it}] 선별로 거르지 않는다 — 후보 {len(screened)}개를 전부 본채점"
                    f"(--full-score-max {a.full_score_max})")
            if not to_full:
                to_full = [best_j]
            if a.screen_skip and best_sc["delta"]["mean"] <= 0:
                # 선별에서 최고 후보조차 평균 Δ 가 0 이하면 본채점($2, 30분)을 아낀다. 부검은 선별
                # 문장으로 한다 — 다음 Critic 이 무엇이 무너졌는지는 봐야 한다.
                log(f"[iter {it}] 후보 {best_j} 선별 Δ {best_sc['delta']['mean']:+.4f} ≤ 0 — "
                    f"본채점 생략, 기각")
                diag = revision_diagnosis(devA, cur_h, best_sc["h"], cur_sets, best_sc["sets"],
                                          spaced)
                diag = add_vs_base(diag, best_sc["h"], best_sc["sets"])
                edits_now = aj.edit_summary(prompt, (pe or {}).get("edits"))
                pm = gw.chat_json(aj.postmortem_system(),
                                  json.dumps({"verdict": "reject", "delta": best_sc["delta"],
                                              "changelog": note, "edits": edits_now, **diag},
                                             ensure_ascii=False),
                                  max_tokens=AGENT_MAX_TOKENS, purpose="postmortem")
                save_usage(idir)
                for k in ("why", "lesson", "blamed"):
                    diag[k] = (pm or {}).get(k)
                diag["blamed"] = aj.blamed_with_text(diag["blamed"], cand)
                (idir / "regression.json").write_text(
                    json.dumps({"verdict": "reject", "screened_only": True,
                                "delta": best_sc["delta"], **diag}, ensure_ascii=False, indent=1),
                    encoding="utf-8")
                log(f"[iter {it}] 부검(선별 {len(sub)}문장) 구간별 Δ {diag['by_bin']} / 나빠진 짝 "
                    f"{diag['n_worse']} 좋아진 {diag['n_better']} / {str(diag.get('why'))[:90]}")
                history.append({"iter": it, "candidate": best_j, "adopted": False,
                                "screened_only": True, "delta": best_sc["delta"],
                                "changelog": note, "edits": edits_now, **base_tag,
                                "findings": [f["diagnosis"] for f in findings],
                                "diagnosis": {k: diag[k] for k in ("by_bin", "n_worse",
                                                                   "n_better", "why", "lesson",
                                                                   "blamed", "vs_base")
                                              if k in diag}})
                (idir / "result.json").write_text(json.dumps(history[-1], ensure_ascii=False,
                                                             indent=1), encoding="utf-8")
                (idir / "prompt.txt").write_text(cand, encoding="utf-8")
                return None
        elif a.screen_n <= 0 and len(valid) > 1:
            # **선별을 끈 경로.** 후보 전부를 곧바로 본채점한다 — `--full-score-max` 만큼만.
            # 선별은 필터로 쓰지 않기로 한 뒤(judge20) 중간 보고만 남았는데, 문장을 이터마다
            # 무작위로 흩어 뽑아 본채점 배치와 경계가 안 맞으니 캐시도 거의 안 살아난다.
            # 실측으로 이터당 4~5분, $1.5 순증이었다(judge24 이터 1: 7후보 선별 4분 44초).
            to_full = [j for j, *_ in valid][:max(1, a.full_score_max)]
            log(f"[iter {it}] 선별을 건너뛴다(--screen-n 0) — 후보 {len(to_full)}개 곧바로 본채점")
        else:
            to_full = [valid[0][0]]

        def full_score(j):
            _j, pe_j, cand_j, note_j, deltas_j = next(v for v in valid if v[0] == j)
            c_rows, c_sets, c_h, c_m = score_prompt(cand_j, devA, labA, f"iter {it} 후보 {j}")
            excl = (example_sentences(best, devA) | example_sentences(prompt, devA)
                    | example_sentences(cand_j, devA))
            keys = sorted(k for k in set(c_h) & set(cur_h) if k[0] not in excl)
            jkeys = adopt_keys(devA, keys, spaced, a.adopt_bin)
            boot = hset.paired_bootstrap([c_h[k] for k in jkeys], [cur_h[k] for k in jkeys],
                                         clusters=[i for i, _k in jkeys])
            log(f"[iter {it}] 후보 {j} 본채점 Δ {boot['mean']:+.4f} [{boot['lo']:+.4f}, "
                f"{boot['hi']:+.4f}] 짝 {boot['n']} / 문장 {boot['n_clusters']}"
                + (f" ({a.adopt_bin} 만)" if a.adopt_bin else ""))
            if a.adopt_bin:         # 판정은 한 구간으로 하되 전체도 남긴다 — 사후에 둘을 비교한다
                whole = hset.paired_bootstrap([c_h[k] for k in keys], [cur_h[k] for k in keys],
                                              clusters=[i for i, _k in keys])
                log(f"[iter {it}] 후보 {j} 전체 Δ {whole['mean']:+.4f} [{whole['lo']:+.4f}, "
                    f"{whole['hi']:+.4f}] 짝 {whole['n']} (판정에는 안 쓴다)")
            return {"j": j, "pe": pe_j, "cand": cand_j, "note": note_j, "deltas": deltas_j,
                    "rows": c_rows, "sets": c_sets, "h": c_h, "m": c_m, "excl": excl, "boot": boot,
                    "keys": keys, "bins": bin_deltas(devA, cur_h, c_h, keys, spaced)}

        def confirm_draws(passers):
            """1차를 통과한 후보들을 추가 벌로 다시 재서 (1차 벌 포함) `--confirm-draws` 벌 평균으로
            기준선과 짝 비교한다. 1차의 한 벌로 판정하면 **고르기 편향이 그대로 들어간다** —
            judge26 에서 1차 +0.0143 [+0.0029] 이던 후보가 새 벌에서 −0.0019 였다. 문장은 같게 두고
            분절만 새로 뽑는 것이 핵심이다(같은 문장이라 짝 비교가 유지되고, 가릴 대상이 추출이다).

            **(후보 × 벌) 을 한 풀에 던진다.** 통과자 3개에 2벌씩이면 6회 채점인데 차례로 하면
            36분이고 겹치면 분절 대기가 포개져 한 자릿수 분이 된다."""
            t0 = time.perf_counter()
            tasks = [(f, d) for f in passers for d in range(2, a.confirm_draws + 1)]

            def one(t):
                f, d = t
                _r, _s, h_d, _m = score_prompt(
                    f["cand"], devA, labA,
                    f"iter {it} 후보 {f['j']} 확인 {d}/{a.confirm_draws}",
                    cache=draw_cache(f"conf{d}"))
                return f["j"], h_d

            got = []
            if tasks:
                with ThreadPoolExecutor(max_workers=min(len(tasks),
                                                        max(1, a.score_workers))) as ex:
                    got = list(ex.map(one, tasks))
            save_usage(idir)            # 크래시해도 여기까지의 지출이 남는다
            extra: dict = {}
            for j, h_d in got:
                extra.setdefault(j, []).append(h_d)
            for f in passers:
                hs = [f["h"]] + extra.get(f["j"], [])
                ks = set(hs[0])
                for h_d in hs[1:]:
                    ks &= set(h_d)
                avg = {k: sum(h_d[k] for h_d in hs) / len(hs) for k in ks}
                jk = sorted(k for k in set(avg) & set(cur_h) if k[0] not in f["excl"])
                boot = hset.paired_bootstrap([avg[k] for k in jk], [cur_h[k] for k in jk],
                                             clusters=[i for i, _k in jk])
                f["h_avg"], f["boot_avg"] = avg, boot
                f["bins_avg"] = bin_deltas(devA, cur_h, avg, jk, spaced)
                log(f"[iter {it}] 후보 {f['j']} {a.confirm_draws}벌 확인 Δ {boot['mean']:+.4f} "
                    f"[{boot['lo']:+.4f}, {boot['hi']:+.4f}] 짝 {boot['n']} / 벌별 H "
                    f"{[round(st.mean(x.values()), 4) for x in hs]}")
            timing["confirm_draws"] = round(time.perf_counter() - t0, 1)
            log(f"[iter {it}] 2차 {len(passers)}개 × 추가 {a.confirm_draws - 1}벌 = "
                f"{len(tasks)}회 채점을 병렬로 — {timing['confirm_draws']:.0f}초")

        t0 = time.perf_counter()
        if a.score_workers > 1 and len(to_full) > 1:
            # 후보 하나는 dev 문장수/batch_size 만큼의 콜밖에 안 써서 워커가 남는다 — dev 500·
            # batch 6 이면 84콜에 워커 128 로 활용률 66% 다. 둘씩 묶으면 168콜이 128 연결로 나가
            # 1.3 라운드가 되어 후보당 벽시계가 약 65% 로 준다. 게이트웨이의 `max_connections`
            # 가 실제 천장이므로 워커만 올리는 것으로는 안 빨라진다(judge12: 16→64 에 시간 불변).
            with ThreadPoolExecutor(max_workers=a.score_workers) as ex:
                fulls = list(ex.map(full_score, to_full))      # map 은 입력 순서를 지킨다
        else:
            fulls = [full_score(j) for j in to_full]
        timing["score_candidate"] = round(time.perf_counter() - t0, 1)
        if a.score_workers > 1 and len(to_full) > 1:
            log(f"[iter {it}] 후보 {len(to_full)}개를 {a.score_workers}개씩 병렬 채점 — "
                f"{timing['score_candidate']:.0f}초")
        # 채택을 가르는 자로 고르고, 퇴행 가드는 **선택 전에** 후보마다 매긴다. 뒤에 매기면 뽑힌 하나가
        # 구간을 무너뜨릴 때 통과하는 다른 후보를 남겨두고도 이터 전체가 기각된다 — 본채점 두 번이
        # 통째로 버려진다. 구간 Δ 는 이미 잰 값을 나누기만 하므로 추가 채점 비용은 없다.
        fkey = pick_key(a.adopt_rule)
        clean = [f for f in fulls if not guard_breaches(f["bins"], a.guard_bin)]
        if len(fulls) > 1 and clean and len(clean) < len(fulls):
            log(f"[iter {it}] 퇴행 가드에 걸린 후보 "
                + ", ".join(str(f["j"]) for f in fulls if f not in clean) + " 는 선택에서 뺀다")
        pool = clean or fulls
        multi = a.confirm_draws > 1
        if multi:
            # **1차는 선별, 2차가 판정이다.** 1차 문턱을 하한 > 0 으로 두면 기준선을 제대로 뽑은 뒤엔
            # 통과가 거의 안 난다(세 런 후보 66개를 보정해 세니 1개, 1.5%). 평균 > 0 이면 20% 다.
            gkey = (lambda b: b["mean"]) if a.gate_rule == "mean" else (lambda b: b["lo"])
            gname = "평균" if a.gate_rule == "mean" else "하한"
            passers = [f for f in pool if gkey(f["boot"]) > a.gate_min]
            log(f"[iter {it}] 1차({gname} > {a.gate_min:+.4f}) 통과 {len(passers)}/{len(pool)}개"
                + (" — " + ", ".join(f"후보 {f['j']}" for f in passers) if passers else ""))
            confirm_draws(passers)
            ok = [f for f in passers if f["boot_avg"]["lo"] > 0]
            if ok:
                chosen = max(ok, key=lambda f: f["boot_avg"]["lo"])
                log(f"[iter {it}] 2차 통과 {len(ok)}/{len(passers)}개 — 하한 최고 후보 "
                    f"{chosen['j']} 채택")
            else:
                chosen = max(pool, key=lambda f: fkey(f["boot"]))
                log(f"[iter {it}] 2차 통과 0/{len(passers)}개 — 기각 "
                    f"(부검은 1차 최고 후보 {chosen['j']} 로 한다)")
        else:
            chosen = max(pool, key=lambda f: fkey(f["boot"]))
        for f in fulls:
            if f is not chosen:     # 본채점까지 갔지만 뽑히지 않은 후보 — 이력에 남긴다
                history.append({"iter": it, "candidate": f["j"], "adopted": False,
                                "full_scored": True, "delta": f["boot"], "changelog": f["note"],
                                "delta_confirm": f.get("boot_avg"),
                                "edits": aj.edit_summary(prompt, (f["pe"] or {}).get("edits")),
                                **base_tag})
        if len(fulls) > 1 and not multi:
            kn = "평균" if a.adopt_rule == "mean" else "하한"
            log(f"[iter {it}] 본채점 {len(fulls)}개 중 {kn} 최고 후보 {chosen['j']} 선택")
        pe, cand, note, deltas = chosen["pe"], chosen["cand"], chosen["note"], chosen["deltas"]
        c_rows, c_sets, c_h, c_m, excl, boot = (chosen["rows"], chosen["sets"], chosen["h"],
                                                 chosen["m"], chosen["excl"], chosen["boot"])
        if "boot_avg" in chosen:
            # 채택되면 이 값이 **다음 이터의 기준선**이 된다 — 1차 한 벌을 물려주면 그 벌의 오차가
            # 다음 이터 후보 전부에 다시 얹힌다. 다벌 평균을 넘긴다.
            c_h, boot = chosen["h_avg"], chosen["boot_avg"]
        (idir / "violations.json").write_text(
            json.dumps({"candidate": violation_summary(c_rows)}, ensure_ascii=False, indent=1),
            encoding="utf-8")
        if excl:
            log(f"[iter {it}] 예시로 들어간 dev-A 문장 {len(excl)}개는 판정에서 뺀다")
        keys, vbins = chosen["keys"], chosen.get("bins_avg") or chosen["bins"]
        if multi:
            # 2차가 이미 갈랐다 — 하한 > 0 인 후보 중에서만 chosen 이 나온다.
            verdict = "accept" if "boot_avg" in chosen and boot["lo"] > 0 else "reject"
        else:
            verdict = decide(boot, strong=a.adopt_strong, rule=a.adopt_rule,
                             bins=vbins, guard=a.guard_bin)
        gain = boot
        log(f"[iter {it}] Δ H_set {boot['mean']:+.4f} [{boot['lo']:+.4f}, {boot['hi']:+.4f}] "
            f"짝 {boot['n']} / 문장 {boot['n_clusters']} → {verdict}")
        if verdict == "reject_guard":
            log(f"[iter {it}] 퇴행 가드 — 평균은 양수인데 무너진 구간이 있다: "
                + ", ".join(guard_breaches(vbins, a.guard_bin)))
            verdict = "reject"

        if verdict == "confirm":
            # **양쪽을 새로 뽑아 다시 잰다.** 후보만 다시 뽑으면 반쪽이다 — 2026-09-19 judge25 를
            # 사후에 재현해 보니 사라진 이득 0.0101 중 **0.0064 가 기준선 쪽**이었다(같은 dev 500·
            # 같은 v0 인데 분절만 새로 뽑으니 0.5514 → 0.5578). 기준선 한 벌은 후보 전부가 공유해서
            # 그게 낮게 뽑히면 후보가 다 같이 올라간다 — judge25 이터 2 는 서로 다른 편집 넷 중
            # **셋이 동시에** 하한 > 0 이었다. 그 채택이 test 560 에서 −0.0070 으로 뒤집혔다.
            log(f"[iter {it}] 경계선(하한 {boot['lo']:+.4f}) — 양쪽을 새 추출로 재채점")
            t0 = time.perf_counter()
            # 파일 이름에 **런 이름**이 들어가야 한다 — `cache/` 는 from-run 으로의 심볼릭
            # 링크라 여러 런이 한 디렉토리를 쓴다. 이터 번호만 넣으면 다음 런의 같은 이터가
            # 이 파일을 그대로 읽어 "새 추출" 이 아니게 된다(v0 가 고정이라 기준선에서 반드시
            # 부딪힌다).
            tmp = JsonCache.shared(run_dir / "cache"
                                    / f"segment_confirm_{run_dir.name}_{it}.json")
            with ThreadPoolExecutor(max_workers=2) as ex:
                f_c = ex.submit(segment_rows, cand, devA, labA, tmp)
                f_b = ex.submit(segment_rows, best, devA, labA, tmp)
                rows2, _m2 = f_c.result()
                rows2b, _m2b = f_b.result()
            tmp.flush()
            h2 = hset_of(devA, labA, policy_sets(rows2, devA, spaced, min_gap,
                                                 a.min_chunk, a.max_k))
            h2b = hset_of(devA, labA, policy_sets(rows2b, devA, spaced, min_gap,
                                                  a.min_chunk, a.max_k))
            k2 = sorted(k for k in set(h2) & set(h2b) if k[0] not in excl)
            gain = hset.paired_bootstrap([h2[k] for k in k2], [h2b[k] for k in k2],
                                         clusters=[i for i, _k in k2])
            timing["confirm_redraw"] = round(time.perf_counter() - t0, 1)
            log(f"[iter {it}] 재추출 확인 Δ {gain['mean']:+.4f} [{gain['lo']:+.4f}, "
                f"{gain['hi']:+.4f}] 짝 {gain['n']} / 기준선 {st.mean(h2b.values()):.4f} "
                f"(원 추출 {st.mean(cur_h.values()):.4f}) / {timing['confirm_redraw']:.0f}초")
            verdict = "accept" if gain["lo"] > 0 else "reject"

        if verdict == "reject" and a.confirm_dev_b and promising(boot):
            # 유망한 기각은 dev-B(265문장)를 더 재서 dev-A 와 합산해 다시 가른다. 문장이 415 로 늘면
            # CI 반폭이 약 0.009 로 준다. 현재 프롬프트의 dev-B 값은 체크포인트에 있으면 그걸 쓰고,
            # 없으면 재서 체크포인트로 남긴다(다음 정기 체크포인트가 그대로 쓴다).
            log(f"[iter {it}] 유망(평균 {boot['mean']:+.4f}, 하한 {boot['lo']:+.4f}) — dev-B 로 확인")
            t0 = time.perf_counter()
            if checkpoint and checkpoint["prompt"] == best:
                hB_cur = {(i, k): v for i, k, v in checkpoint["h"]}
            else:
                _, _s, hB_cur, _m = score_prompt(best, devB, labB, f"iter {it} dev-B 현재")
                checkpoint = {"iter": it, "value": round(st.mean(hB_cur.values()), 5),
                              "prompt": best, "provenance": copy.deepcopy(provenance),
                              "h": [[i, k, v] for (i, k), v in sorted(hB_cur.items())]}
                log(f"[체크포인트] dev-B {checkpoint['value']:.4f} 저장 (확인 채점을 겸함)")
            _, _s, hB_c, _m = score_prompt(cand, devB, labB, f"iter {it} dev-B 후보")
            exclB = (example_sentences(best, devB) | example_sentences(prompt, devB)
                     | example_sentences(cand, devB))
            kB = sorted(k for k in set(hB_c) & set(hB_cur) if k[0] not in exclB)
            bootB = hset.paired_bootstrap([hB_c[k] for k in kB], [hB_cur[k] for k in kB],
                                          clusters=[i for i, _k in kB])
            pooled = hset.paired_bootstrap(
                [c_h[k] for k in keys] + [hB_c[k] for k in kB],
                [cur_h[k] for k in keys] + [hB_cur[k] for k in kB],
                clusters=[i for i, _k in keys] + [100000 + i for i, _k in kB])
            timing["confirm_dev_b"] = round(time.perf_counter() - t0, 1)
            verdict = "accept" if pooled["lo"] > 0 else "reject"
            gain = {**pooled, "dev_a": boot, "dev_b": bootB, "pooled": True}
            log(f"[iter {it}] dev-B Δ {bootB['mean']:+.4f} [{bootB['lo']:+.4f}, {bootB['hi']:+.4f}] "
                f"짝 {bootB['n']} / 합산 Δ {pooled['mean']:+.4f} [{pooled['lo']:+.4f}, "
                f"{pooled['hi']:+.4f}] 짝 {pooled['n']} / 문장 {pooled['n_clusters']} → {verdict}")

        def verify_on_b(f, tries_used: int):
            """dev-B 로 확인한다 — 통과하면 bootB, 아니면 None.

            **시도마다 문턱을 올린다.** 확인 분할에서 여러 후보를 재면 그 분할도 고르기 분할이
            되어 편향이 돌아온다(1개만 재면 우연 통과 2.5%×50%=1.25%, 넷을 다 재고 하나만
            통과해도 채택하면 약 2.3% 로 judge21 수준이다). k번째 시도는 "앞의 것들이 떨어졌는데도
            통과했다" 는 것이므로 더 강한 증거를 요구한다 — 문턱이 `verify_min + (k-1)*verify_step`
            으로 올라간다."""
            nonlocal checkpoint
            need = a.verify_min + tries_used * a.verify_step
            log(f"[iter {it}] dev-B {len(devB)}문장으로 후보 {f['j']} 확인 "
                f"(시도 {tries_used + 1}, 문턱 평균 > {need:+.4f})")
            t1 = time.perf_counter()
            if checkpoint and checkpoint["prompt"] == best:
                hB_cur = {(i, k): v for i, k, v in checkpoint["h"]}
            else:
                _, _s, hB_cur, _m = score_prompt(best, devB, labB, f"iter {it} dev-B 현재")
                checkpoint = {"iter": it, "value": round(st.mean(hB_cur.values()), 5),
                              "prompt": best, "provenance": copy.deepcopy(provenance),
                              "h": [[i, k, v] for (i, k), v in sorted(hB_cur.items())]}
                log(f"[체크포인트] dev-B {checkpoint['value']:.4f} 저장 (확인 채점을 겸함)")
            _, _s, hB_c, _m = score_prompt(f["cand"], devB, labB,
                                           f"iter {it} dev-B 확인 후보 {f['j']}")
            exclB = (example_sentences(best, devB) | example_sentences(prompt, devB)
                     | example_sentences(f["cand"], devB))
            kB = sorted(k for k in set(hB_c) & set(hB_cur) if k[0] not in exclB)
            bB = hset.paired_bootstrap([hB_c[k] for k in kB], [hB_cur[k] for k in kB],
                                       clusters=[i for i, _k in kB])
            timing[f"verify_dev_b_{f['j']}"] = round(time.perf_counter() - t1, 1)
            ok = bB["mean"] > need
            log(f"[iter {it}] dev-B 확인 후보 {f['j']} Δ {bB['mean']:+.4f} [{bB['lo']:+.4f}, "
                f"{bB['hi']:+.4f}] 짝 {bB['n']} / 문장 {bB['n_clusters']} "
                f"→ {'통과' if ok else '탈락'}")
            return ({**bB, "tries": tries_used + 1, "threshold": round(need, 5)} if ok else None)

        if verdict == "accept" and a.verify_dev_b and devB:
            # **고른 뒤에 다른 문장으로 확인한다.** dev-A 에서 후보 여러 개를 재고 최고를 고르면
            # 그 값은 위로 치우친다 — judge21 이 dev 600 하나로 12번 재고 고른 개정이 최종
            # 홀드아웃에서 −0.0089 로 뒤집혔다. 합산(`--confirm-dev-b`)과 다르다 — 합산은 표본만
            # 늘려 구간을 좁히지 선택 편향을 풀지 못한다.
            #
            # 1등이 탈락하면 `--verify-tries` 만큼 다음 후보로 내려간다. 내려갈수록 문턱이
            # `verify_step` 씩 올라간다 — 확인 분할에서 여러 번 재는 것 자체가 고르기이므로,
            # 뒤 순위가 통과하려면 더 강한 증거를 내야 한다.
            order = sorted(clean or fulls, key=lambda f: -fkey(f["boot"]))
            order = [f for f in order if f["boot"]["lo"] > 0] or order[:1]
            bootB, picked = None, None
            for n_try, f in enumerate(order[:max(1, a.verify_tries)]):
                bootB = verify_on_b(f, n_try)
                if bootB:
                    picked = f
                    break
            if picked is None:
                verdict = "reject"
                gain = {**boot, "dev_a": boot, "verified": False,
                        "tries": min(len(order), max(1, a.verify_tries))}
                log(f"[iter {it}] dev-A 최고는 {boot['mean']:+.4f} 였는데 확인 분할을 "
                    f"{gain['tries']}번 시도해 전부 탈락했다 — 고르기가 만든 편향이다")
            else:
                if picked is not chosen:
                    log(f"[iter {it}] 확인 분할을 통과한 것은 dev-A {order.index(picked) + 1}위 "
                        f"후보 {picked['j']} 다 — 이 후보로 채택한다")
                    chosen = picked
                    pe, cand, note, deltas = (chosen["pe"], chosen["cand"], chosen["note"],
                                              chosen["deltas"])
                    c_rows, c_sets, c_h, c_m, excl, boot = (
                        chosen["rows"], chosen["sets"], chosen["h"], chosen["m"],
                        chosen["excl"], chosen["boot"])
                    keys, vbins = chosen["keys"], chosen["bins"]
                verdict = "accept"
                gain = {**boot, "dev_a": boot, "dev_b": bootB, "verified": True,
                        "tries": bootB["tries"]}

        # 부검 — 어느 (문장, k) 가 어떻게 움직였는지 세고, 왜 그랬는지 한 문단을 받는다.
        edits_now = aj.edit_summary(prompt, (pe or {}).get("edits"))
        diag = revision_diagnosis(devA, cur_h, c_h, cur_sets, c_sets, spaced)
        diag = add_vs_base(diag, c_h, c_sets)
        # 순위 깊이 — **개정이 순위의 어디를 고쳤는지**를 부검에도 남긴다. 구간별 Δ 만으로는
        # 긴 지연에서 번 것과 짧은 지연에서 번 것이 갈리지만, "1위를 정리했나 8위를 정리했나"
        # 는 안 보인다. 짧은 지연은 8위까지 쓰고 긴 지연은 1위만 쓴다.
        d_old, d_new = rank_depth(cur_rows), rank_depth(c_rows)
        if d_old and d_new:
            diag["rank_depth"] = {"base": d_old, "revision": d_new,
                                  "delta": {k: round(d_new[k] - d_old.get(k, 0), 3)
                                            for k in d_new if k in d_old}}
            log(f"[iter {it}] 부검 순위 깊이 배수 변화 "
                + " ".join(f"{k}:{diag['rank_depth']['delta'][k]:+.2f}"
                           for k in ("1", "3", "5", "8", "13")
                           if k in diag["rank_depth"]["delta"]))
        t0 = time.perf_counter()
        pm = gw.chat_json(aj.postmortem_system(),
                          json.dumps({"verdict": verdict, "delta": gain, "changelog": note,
                                      "edits": edits_now, **diag}, ensure_ascii=False),
                          max_tokens=AGENT_MAX_TOKENS, purpose="postmortem")
        timing["postmortem"] = round(time.perf_counter() - t0, 1)
        save_usage(idir)
        for k in ("why", "lesson", "blamed"):
            diag[k] = (pm or {}).get(k)
        diag["blamed"] = aj.blamed_with_text(diag["blamed"], cand)
        (idir / "regression.json").write_text(json.dumps({"verdict": verdict, "delta": gain,
                                                          **diag}, ensure_ascii=False, indent=1),
                                              encoding="utf-8")
        log(f"[iter {it}] 부검 구간별 Δ {diag['by_bin']} / 나빠진 짝 {diag['n_worse']} "
            f"좋아진 {diag['n_better']} / {str(diag.get('why'))[:90]}")
        history.append({"iter": it, "adopted": verdict == "accept", "delta": boot, "gain": gain,
                        "changelog": note, "findings": [f["diagnosis"] for f in findings],
                        "edits": edits_now, **base_tag,
                        "diagnosis": {k: diag[k] for k in ("by_bin", "n_worse", "n_better",
                                                           "why", "lesson", "blamed", "vs_base")
                                      if k in diag}})
        (idir / "result.json").write_text(json.dumps(history[-1], ensure_ascii=False, indent=1),
                                          encoding="utf-8")
        (idir / "prompt.txt").write_text(cand, encoding="utf-8")
        return (cand, c_sets, c_h, c_m, gain) if verdict == "accept" else None

    case_state = {"prompt": None, "sets": None, "h": None, "rows": None}
    # 이미 보인 사례 문장 — 채택될 때까지 누적해서 뺀다. 직전 이터만 빼면 프롬프트가 안 바뀌는
    # 동안 두 이터 주기로 같은 12문장이 돌아온다(judge11 iter 1 = iter 3). 채택되면 손해 순위가
    # 새로 생기므로 비운다.
    seen_case_ids: set[str] = set()
    seen_case_keys: set[tuple[str, str]] = set()      # --case-exclude bin: (문장 id, 지연 구간)
    shown_cuts: dict[str, list] = {}                  # 문장 id -> 앞서 보여준 절단집합들

    def remember_cases(cs: list[dict]) -> None:
        seen_case_ids.update(c["id"] for c in cs)
        seen_case_keys.update((c["id"], c["latency_bin"]) for c in cs)
        for c in cs:
            shown_cuts.setdefault(c["id"], []).append(tuple(c["policy"]["cut"]))

    if start > 1:
        # 재개 — 집합은 메모리에만 있으므로 마지막 채택 뒤 이터들의 cases.json 에서 다시 모은다
        # (judge12 iter 2 재개에서 iter 1 과 같은 12문장이 다시 뽑혔다).
        last_adopt = max([h["iter"] for h in history if h.get("adopted")], default=0)
        for k in range(last_adopt + 1, start):
            cp = run_dir / f"iter_{k:02d}" / "cases.json"
            if cp.exists():
                remember_cases(json.loads(cp.read_text(encoding="utf-8")))
        if seen_case_ids:
            log(f"[resume] 이미 보인 사례 문장 {len(seen_case_ids)}개 복원 (이터 {last_adopt + 1}~{start - 1})")
    for it in range(start, a.iterations + 1):
        snapshot_state(it - 1)
        idir = run_dir / f"iter_{it:02d}"
        idir.mkdir(parents=True, exist_ok=True)
        t_iter = time.perf_counter()

        base = None if a.no_near_miss else near_miss_base(history, run_dir)
        edit_src = prompt
        if base:
            d = base[1].get("gain") or base[1].get("delta")
            edit_src = base[0]
            log(f"[iter {it}] 근소 기각본(iter {base[1]['iter']}, {d['mean']:+.4f} [{d['lo']:+.4f}, "
                f"{d['hi']:+.4f}]) 을 편집 기반으로 — 판정은 현재 채택본 대비 그대로")
        if sep_cases:
            # 사례는 train 에서 뽑는다. 편집 기반이 바뀐 이터에만 다시 잰다(같으면 캐시 적중).
            if case_state["prompt"] != edit_src:
                case_state["rows"], case_state["sets"], case_state["h"], _cm = score_prompt(
                    edit_src, case_sents, case_lab, f"iter {it} train")
                case_state["prompt"] = edit_src
            case_sets, case_h = case_state["sets"], case_state["h"]
            case_rows = case_state.get("rows") or []
        else:
            case_sets, case_h = cur_sets, cur_h
            case_rows = cur_rows

        def pieces_tr(i, cut):
            u = L.units_of(case_sents[i].text, spaced)
            out = {}
            for tgt in targets:
                out[tgt] = [scorer.piece(tgt, u, x, y)
                            for x, y in hset.pieces_of(u, cut, spaced)]
            return out

        by_bin_excl = a.case_exclude == "bin"
        cases, loss_bins = build_cases(
            case_sents, case_lab, case_sets, ora_C, case_h, ora_hC,
            spaced, min_gap, a.n_cases, pieces_tr,
            exclude_ids=frozenset() if by_bin_excl else seen_case_ids, alloc=a.case_alloc,
            exclude_keys=seen_case_keys if by_bin_excl else frozenset(),
            shown_cuts=shown_cuts if by_bin_excl else None)
        if len(cases) < a.n_cases and (seen_case_ids or seen_case_keys):
            # 풀이 말랐다 — 직전 이터 것만 남기고 비운다. 그대로 두면 사례가 조용히 줄어든다
            # (`pick_cases` 는 큐가 안 움직이면 break 한다).
            prev = json.loads((run_dir / f"iter_{it - 1:02d}" / "cases.json").read_text(
                encoding="utf-8")) if (run_dir / f"iter_{it - 1:02d}" / "cases.json").exists() else []
            log(f"[iter {it}] 사례 풀 소진 ({len(cases)}/{a.n_cases}) — 직전 이터 "
                f"{len(prev)}개만 남기고 제외 목록을 비운다")
            seen_case_ids.clear(); seen_case_keys.clear(); shown_cuts.clear()
            remember_cases(prev)
            cases, loss_bins = build_cases(
                case_sents, case_lab, case_sets, ora_C, case_h, ora_hC,
                spaced, min_gap, a.n_cases, pieces_tr,
                exclude_ids=frozenset() if by_bin_excl else seen_case_ids, alloc=a.case_alloc,
                exclude_keys=seen_case_keys if by_bin_excl else frozenset(),
                shown_cuts=shown_cuts if by_bin_excl else None)
        remember_cases(cases)
        # 순위 깊이 — 개정이 순위의 **어디를** 고쳤는지 남긴다. 라벨과 이미 잰 점수만 쓰므로 공짜다.
        depth = rank_depth(case_rows)
        if depth:
            log(f"[iter {it}] 순위 깊이별 무작위 대비 배수 "
                + " ".join(f"{d}:{v}" for d, v in depth.items() if d in ("1", "3", "5", "8", "13")))
        inversions = rank_inversions(case_rows, case_sents, spaced, n_max=a.inversions_max)
        if inversions:
            log(f"[iter {it}] 깊은 구간 역전 쌍 {len(inversions)}개 — Critic 에 넘긴다 "
                f"(라벨 차 최대 {inversions[0]['label_gap']})")
        # 게이트 몫은 Critic·PE 에 안 보여준다. 구간 배분을 유지하려고 앞에서 몇 개를 떼지 않고
        # 일정 간격으로 고른다 (`pick_cases` 가 구간을 한 바퀴씩 돌며 채우므로 간격 추출이 섞인다).
        gate_cases: list[dict] = []
        if a.case_holdout > 0 and len(cases) > a.case_holdout:
            stride = max(1, len(cases) // a.case_holdout)
            gate_idx = list(range(0, len(cases), stride))[:a.case_holdout]
            gate_cases = [cases[i] for i in gate_idx]
            cases = [c for i, c in enumerate(cases) if i not in set(gate_idx)]
            log(f"[iter {it}] 사례 {len(cases)} (Critic·PE) + 게이트 홀드아웃 {len(gate_cases)}")
            (idir / "gate_cases.json").write_text(json.dumps(gate_cases, ensure_ascii=False,
                                                             indent=1), encoding="utf-8")
        log(f"[iter {it}] 구간별 손실 몫 " + " ".join(
            f"{b}:{v['loss_share']:.2f}({v['pairs']}짝)" for b, v in loss_bins.items()))
        (idir / "cases.json").write_text(json.dumps(cases, ensure_ascii=False, indent=1),
                                         encoding="utf-8")
        log(f"[iter {it}] 사례 {len(cases)} / 손해 상위 {cases[0]['gap'] if cases else 0} / 구간 "
            + ", ".join(f"{b}:{n}" for b, n in sorted(
                {c["latency_bin"]: sum(1 for x in cases if x["latency_bin"] == c["latency_bin"])
                 for c in cases}.items(), key=lambda kv: float(kv[0].strip("≤>")))))
        if not cases:
            log("[stop] 오라클보다 못한 (문장,T) 가 없다")
            break

        budget = length_cap(len(prompt), v0_len, a.growth_per_iter, a.growth_ceiling)
        target = aj.soft_target(budget, len(prompt))
        log(f"[iter {it}] 길이 상한 {budget}자 / 모델에 알리는 목표 {target}자 (직전 채택본 {len(prompt)}자)")
        timing = {"cases": round(time.perf_counter() - t_iter, 1)}
        examples = (labeled_examples(case_sents, case_lab, cases, spaced, min_gap)
                    if a.labeled_examples else None)
        if examples:
            # 기각된 개정(`iter_NN/prompt.txt` 는 그 이터에 잰 후보)이 넣었던 예시는 다시 안 준다
            rejected = [run_dir / f"iter_{h['iter']:02d}" / "prompt.txt" for h in history
                        if not h.get("adopted") and not h.get("reason") and not h.get("screened_out")]
            examples, dropped = aj.exclude_used_examples(
                examples, [prompt] + [q.read_text(encoding="utf-8") for q in rejected if q.exists()])
            if dropped:
                log(f"[iter {it}] 현재 프롬프트나 기각된 개정에 이미 있는 실측 예시 {len(dropped)}개 "
                    f"제외: {dropped}")
        adopted = propose(it, idir, cases, budget, target, timing, examples, loss_bins,
                          base=base, gate_cases=gate_cases, inversions=inversions,
                          depth=depth, misorder=misorder_cost(case_rows))
        if adopted:
            cand, c_sets, c_h, c_m, gain = adopted
            provenance = aj.adopt_provenance(provenance, cand, it, gain)
            prompt, cur_sets, cur_h, cur_m = cand, c_sets, c_h, c_m
            seen_case_ids.clear(); seen_case_keys.clear(); shown_cuts.clear()
            (run_dir / "best_prompt.txt").write_text(prompt, encoding="utf-8")

        if a.checkpoint_every and it % a.checkpoint_every == 0:
            if checkpoint and checkpoint["prompt"] == prompt:
                log(f"[체크포인트] 직전 체크포인트(iter {checkpoint['iter']})와 같은 프롬프트 — "
                    f"다시 재지 않는다")
            else:
                _, _s, hB, _mB = score_prompt(prompt, devB, labB, f"iter {it} dev-B")
                verdict, cb = checkpoint_verdict(checkpoint, hB)
                vs = ("" if cb is None else
                      f" / 직전 대비 Δ {cb['mean']:+.4f} [{cb['lo']:+.4f}, {cb['hi']:+.4f}]")
                if verdict == "rollback":
                    log(f"[체크포인트] dev-B {st.mean(hB.values()):.4f}{vs} — 롤백")
                    prompt = checkpoint["prompt"]
                    provenance = copy.deepcopy(checkpoint.get("provenance")
                                               or aj.init_provenance(prompt))
                    cur_rows, cur_sets, cur_h, cur_m = score_avg(prompt, devA, labA,
                                                                 f"iter {it} 롤백 후",
                                                                 a.baseline_draws, "base")
                else:
                    checkpoint = {"iter": it, "value": round(st.mean(hB.values()), 5),
                                  "prompt": prompt, "provenance": copy.deepcopy(provenance),
                                  "h": [[i, k, v] for (i, k), v in sorted(hB.items())]}
                    log(f"[체크포인트] dev-B {checkpoint['value']:.4f}{vs} 저장")
                (run_dir / "checkpoint.json").write_text(
                    json.dumps({k: v for k, v in (checkpoint or {}).items()
                                if k not in ("prompt", "provenance", "h")},
                               ensure_ascii=False, indent=1), encoding="utf-8")
        timing["total"] = round(time.perf_counter() - t_iter, 1)
        (idir / "timing.json").write_text(json.dumps(timing, ensure_ascii=False, indent=1),
                                          encoding="utf-8")
        save_usage(idir)
    else:
        snapshot_state(a.iterations)

    (run_dir / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=1),
                                          encoding="utf-8")
    (run_dir / "best_prompt.txt").write_text(prompt, encoding="utf-8")

    # ── 최종 test — 런 밖에서 손으로 돌리던 것을 루프 안에 둔다. 채택본과 v0 를 같은 자로 재야
    #    "이 런이 실제로 올린 폭" 이 남는다. dev 는 개정을 고르는 데 이미 소진됐다.
    base_path = Path(a.score_baseline) if a.score_baseline else v0_path
    if a.score_only and a.score_prompts:
        # **여러 변형을 한 프로세스에서 병렬로 잰다.** 따로 띄우면 기준선을 변형마다 다시 뽑고
        # (벌 캐시 이름이 런마다 달라진다) GPU 를 프로세스끼리 다투게 된다. 여기서는 기준선 3벌을
        # 한 번만 뽑고, 변형들은 같은 기준선에 대고 짝 비교한다 — 후보를 고르는 단계가 없으므로
        # 고르기 편향도 없다.
        v0_prompt = base_path.read_text(encoding="utf-8")
        cands = [(Path(x.strip()).stem, Path(x.strip()).read_text(encoding="utf-8"))
                 for x in a.score_prompts.split(",") if x.strip()]
        f_excl = example_sentences(v0_prompt, test_sents)
        for _n, pr in cands:
            f_excl |= example_sentences(pr, test_sents)
        if f_excl:
            log(f"[최종] 프롬프트에 예시로 들어간 문장 {len(f_excl)}개를 채점에서 뺀다")
        t0 = time.perf_counter()
        jobs = [("v0", v0_prompt)] + cands
        with ThreadPoolExecutor(max_workers=len(jobs)) as ex:
            got = list(ex.map(lambda j: score_avg(j[1], test_sents, lab_test,
                                                 f"최종 {final_split} {j[0]}",
                                                 a.final_draws, "fin"), jobs))
        hs = {n: {k: v for k, v in g[2].items() if k[0] not in f_excl}
              for (n, _p), g in zip(jobs, got)}
        ora_t = oracle_sets(lab_test, test_sents, spaced, min_gap, a.min_chunk, a.max_k)
        ora_h_t = {k: v for k, v in hset_of(test_sents, lab_test, ora_t).items()
                   if k[0] not in f_excl}
        rows_out = []
        for n, _pr in jobs[1:]:
            kk = sorted(set(hs[n]) & set(hs["v0"]))
            b = hset.paired_bootstrap([hs[n][k] for k in kk], [hs["v0"][k] for k in kk],
                                      clusters=[i for i, _k in kk])
            bins = bin_deltas(test_sents, hs["v0"], hs[n], kk, spaced)
            rows_out.append({"name": n, "mean": round(st.mean(hs[n].values()), 4),
                             "delta": b, "by_bin": {x: y["mean"] for x, y in bins.items()},
                             "by_latency": by_latency(test_sents, hs[n], spaced)})
            log(f"[최종] {n}: H_set {st.mean(hs[n].values()):.4f} / Δ {b['mean']:+.4f} "
                f"[{b['lo']:+.4f}, {b['hi']:+.4f}] 짝 {b['n']} / 구간 "
                + str({x: y['mean'] for x, y in bins.items()}))
        log(f"[최종] 기준선 {st.mean(hs['v0'].values()):.4f} / 오라클 "
            f"{st.mean(ora_h_t.values()):.4f} / {a.final_draws}벌 × {len(jobs)}프롬프트 = "
            f"{a.final_draws * len(jobs)}회 채점 {time.perf_counter() - t0:.0f}초")
        (run_dir / "variants.json").write_text(json.dumps({
            "split": final_split, "draws": a.final_draws, "baseline": str(base_path),
            "baseline_mean": round(st.mean(hs["v0"].values()), 4),
            "oracle_mean": round(st.mean(ora_h_t.values()), 4),
            "baseline_by_latency": by_latency(test_sents, hs["v0"], spaced),
            "variants": rows_out}, ensure_ascii=False, indent=1), encoding="utf-8")
        save_usage(run_dir / "final")
        log(f"[끝] 비용 ${spent_before + gw.usage.snapshot()['cost']:.2f} "
            f"({gw.usage.snapshot()['calls']} 호출)")
        return 0
    if not a.skip_final and prompt == base_path.read_text(encoding="utf-8"):
        log("[최종] 채택된 개정이 없다 — v0 그대로라 최종 test 를 생략한다")
    elif not a.skip_final:
        t0 = time.perf_counter()
        v0_prompt = base_path.read_text(encoding="utf-8")
        # **프롬프트에 예시로 박힌 문장은 뺀다.** test 는 train 과 겹치지 않아 여태 필요가
        # 없었는데, `--final-split sel` 로 사례 집합을 품은 분할을 재는 순간 전제가 깨진다
        # (judge09 iter 2 실측: dev-A Δ +0.0105 중 +0.006 이 예시 3문장 몫이었다).
        # dev 본채점은 이미 같은 방어를 하고 있다.
        f_excl = example_sentences(prompt, test_sents) | example_sentences(v0_prompt, test_sents)
        if f_excl:
            log(f"[최종] 프롬프트에 예시로 들어간 문장 {len(f_excl)}개를 채점에서 뺀다")

        def keep(h):
            return {k: v for k, v in h.items() if k[0] not in f_excl}

        # 채택본과 v0 는 **서로 독립이라 같이 잰다.** 순차면 각 10분씩 20분인데 분절이 전부 API
        # 대기다 — GPU 채점은 `hset_of` 의 락이 직렬화한다. 채점이 하나뿐이면 병렬 진입을 안 한다.
        need_v0 = v0_prompt != prompt
        if need_v0 and a.score_workers > 1:
            with ThreadPoolExecutor(max_workers=2) as ex:
                f_t = ex.submit(score_avg, prompt, test_sents, lab_test, "최종 test",
                                a.final_draws, "fin")
                f_0 = ex.submit(score_avg, v0_prompt, test_sents, lab_test, "최종 test v0",
                                a.final_draws, "fin")
                rows_t, _sets_t, h_t, m_t = f_t.result()
                _r0, _s0, h0_raw, _m0 = f_0.result()
        else:
            rows_t, _sets_t, h_t, m_t = score_avg(prompt, test_sents, lab_test, "최종 test",
                                                  a.final_draws, "fin")
            h0_raw = None
        ora_t = oracle_sets(lab_test, test_sents, spaced, min_gap, a.min_chunk, a.max_k)
        ora_h_t = hset_of(test_sents, lab_test, ora_t)
        h_t, ora_h_t = keep(h_t), keep(ora_h_t)
        by_bin = {"prompt": by_latency(test_sents, h_t, spaced),
                  "oracle": by_latency(test_sents, ora_h_t, spaced)}
        means = {"prompt": round(st.mean(h_t.values()), 4),
                 "oracle": round(st.mean(ora_h_t.values()), 4)}
        boot_v0 = None
        if need_v0:
            if h0_raw is None:
                _r0, _s0, h0_raw, _m0 = score_avg(v0_prompt, test_sents, lab_test,
                                                  "최종 test v0", a.final_draws, "fin")
            h0 = keep(h0_raw)
            by_bin["v0"] = by_latency(test_sents, h0, spaced)
            means["v0"] = round(st.mean(h0.values()), 4)
            kk = sorted(set(h_t) & set(h0))
            boot_v0 = hset.paired_bootstrap([h_t[k] for k in kk], [h0[k] for k in kk],
                                            clusters=[i for i, _k in kk])
        (run_dir / "curve.json").write_text(
            json.dumps({"split": final_split, "means": means,
                        "by_bin": by_bin, "vs_v0": boot_v0},
                       ensure_ascii=False, indent=1), encoding="utf-8")
        (run_dir / "violations_test.json").write_text(
            json.dumps(violation_summary(rows_t), ensure_ascii=False, indent=1), encoding="utf-8")
        save_usage(run_dir / "final")
        (run_dir / "final_report.md").write_text(
            final_report(a.run_id, means, by_bin, boot_v0, history,
                         spent_before + gw.usage.snapshot()["cost"], m_t["format_pass_rate"]),
            encoding="utf-8")
        log(f"[최종] {final_split} H_set {means['prompt']:.4f} / 오라클 {means['oracle']:.4f}"
            + (f" / v0 {means['v0']:.4f} Δ {boot_v0['mean']:+.4f} "
               f"[{boot_v0['lo']:+.4f}, {boot_v0['hi']:+.4f}]" if boot_v0 else " (v0 그대로)")
            + f" / {by_bin['prompt']} / {time.perf_counter() - t0:.0f}초")

    u = gw.usage.snapshot()
    head = (f"{names[0]} H_set {st.mean(cur_h.values()):.4f} / 오라클 {st.mean(ora_hA.values()):.4f}"
            if cur_h else "score-only")
    log(f"[끝] {head} / 비용 ${spent_before + u['cost']:.2f} "
        f"(이번 실행 ${u['cost']:.2f}, {u['calls']} 호출)")
    gw.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
