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
import copy
from concurrent.futures import ThreadPoolExecutor
import json
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


def build_cases(sents: list, lab: dict, pol: dict, ora: dict, pol_h: dict, ora_h: dict,
                spaced: bool, min_gap: int, n_cases: int, pieces_tr=None,
                exclude_ids: set[str] = frozenset(), alloc: str = "uniform") -> tuple[list[dict], dict]:
    """손해가 난 (문장, k) 를 지연 구간별로 골라(`pick_cases`) 반사실 쌍으로 만든다 —
    (사례, 구간별 손실표). `exclude_ids` 의 문장(앞 이터 사례)은 빼고 뽑는다.
    `alloc="loss"` 면 구간별 사례 수를 손실 몫에 비례시킨다(`case_quota`), 아니면 균등."""
    gaps = [(ora_h[k] - pol_h[k], k) for k in pol if k in ora_h and k in pol_h]
    n_units = {i: len(L.units_of(sents[i].text, spaced)) for i in {k[0] for _g, k in gaps}}
    bin_of = lambda key: latency_bin(n_units[key[0]], key[1])
    lb = loss_by_bin(gaps, bin_of)
    quota = case_quota(lb, n_cases) if alloc == "loss" else None
    excl = {i for i, s in enumerate(sents) if s.id in exclude_ids}
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


def decide(boot: dict, strong: float = 0.005) -> str:
    """CI 하한으로 가른다. 종전 1 se 문턱은 재채점 잡음(+0.023±0.017)이 그냥 넘었다."""
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
                   help="후보 선별에 쓰는 dev-A 앞부분 문장 수 (캐시에 남아 본채점에서 재사용)")
    p.add_argument("--screen-skip", action="store_true",
                   help="선별 최고 후보의 Δ 평균이 0 이하면 본채점 없이 기각한다 (부검은 선별 문장으로)")
    p.add_argument("--candidate-roles", default="",
                   help="후보별 역할을 쉼표로: free / examples_only / single_small / rewrite. 후보 j 는 "
                        "roles[j %% len] 을 받는다. rewrite 는 Writer 가 [Core Principles]·[Examples] "
                        "를 새로 쓴다(--generate-v0 런에서만). 예: free,examples_only,rewrite")
    p.add_argument("--case-alloc", default="uniform", choices=("uniform", "loss"),
                   help="구간별 사례 수 — uniform: 한 바퀴씩 균등 / loss: 구간별 손실 몫에 비례(최소 1). "
                        "test-A 실측은 손실의 61%% 가 ≤3, 26%% 가 ≤5 인데 균등 배분은 그 둘에 6/12 만 준다")
    p.add_argument("--full-score-max", type=int, default=1,
                   help="선별 뒤 본채점할 후보 수 상한 — 선별 평균이 양수인 후보를 Δ 순으로 이만큼. "
                        "1 이면 종전처럼 최선 하나. judge12 는 5이터 전부 후보 CI 가 겹쳐 선별이 고르지 못했다")
    p.add_argument("--confirm-dev-b", action="store_true",
                   help="dev-A 에서 기각됐지만 평균 > 0.005, 하한 > -0.01 인 개정은 dev-B 를 더 재서 "
                        "합산(415문장) CI 하한으로 다시 가른다. 유망한 개정 하나에 약 $5")
    p.add_argument("--labeled-examples", action="store_true",
                   help="사례 문장을 실측 라벨로 채운 Input/Output 예시로 만들어 PE 에 준다. PE 는 "
                        "편집에 labeled_example: <사례 id> 로 그 예시를 그대로 붙일 수 있다")
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
    scheme3 = (src / "data/test_a.json").exists()
    if scheme3:
        case_sents, case_lab = load("train"), load_labels("train")
        devA, labA = load("test_a"), load_labels("test_a")
        devB, labB = load("test_b"), load_labels("test_b")
        dev, train = devA + devB, case_sents
        names = ("test-A", "test-B")
        if (src / "data/test.json").exists():
            # 4분할(run27~): 최종 표는 어느 판정에도 안 쓴 test 에서 잰다. test-B 는 체크포인트·
            # 유망 확인에 쓰여 완전히 안 본 집합이 아니다.
            test_sents, lab_test = load("test"), load_labels("test")
            log(f"[data] 4분할 — train {len(case_sents)} (사례·예시) / test-A {len(devA)} (판정) / "
                f"test-B {len(devB)} (체크포인트·확인) / test {len(test_sents)} (최종 홀드아웃)")
        else:
            test_sents, lab_test = devB, labB
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
        names = ("dev-A", "dev-B")
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
    seg_effort = None if a.seg_reasoning_effort == "none" else a.seg_reasoning_effort
    translators = {t: LocalTranslator(tgt_code=to_lang_code(t),
                                      cache=JsonCache(run_dir / "cache" /
                                                      f"translate_{to_lang_code(t)}.json"))
                   for t in targets}
    scorer = hset.HsetScorer(translators=translators,
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

    def hset_of(sents, lab, sets: dict) -> dict:
        keys = sorted(sets)
        texts = [s.text for s in sents]

        def contra_of(i, j):
            return lab[targets[0]][i]["contra"][j - 1]

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
    if scheme3:
        ora_C = oracle_sets(case_lab, case_sents, spaced, min_gap, a.min_chunk, a.max_k)
        ora_hC = hset_of(case_sents, case_lab, ora_C)
        log(f"[목표] train H_set 오라클 평균 {st.mean(ora_hC.values()):.4f} ({len(ora_hC)} 짝)")
    else:
        ora_C, ora_hC = ora_A, ora_hA
    checkpoint = None

    def segment_rows(pr, sents, lab):
        return evaluate(gw, pr, sents, lab, spaced, min_gap, t_grid, seg_cache,
                        a.workers, a.batch_size, seg_effort, k_samples=a.k_samples)

    def score_prompt(pr, sents, lab, tag, rows_m=None):
        """분절(API) → 절단 집합 → H_set. `rows_m` 을 주면 이미 끝난 분절을 쓴다 — 후보 여럿을
        동시에 분절해 두고 GPU 채점만 차례로 할 때."""
        rows, m = rows_m or segment_rows(pr, sents, lab)
        seg_cache.flush()
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
                + f"Copy this [Output Rules] section verbatim into the prompt:\n\n{out_rules}")

    v0_material = None      # Writer 에게 준 재료 — 후보 역할 "rewrite" 가 다시 쓴다
    v0_path = run_dir / "prompt_v0.txt"
    if a.prompt:
        prompt = Path(a.prompt).read_text(encoding="utf-8")
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
            sel_sents, sel_lab, sel_name = ((case_sents, case_lab, "train") if scheme3
                                            else (devA, labA, names[0]))
            scored = []
            for c, pr in enumerate(cands):
                _r, _s, h, _m = score_prompt(pr, sel_sents, sel_lab, f"v0 후보 {c} ({sel_name})")
                save_usage(run_dir / "iter_00")
                scored.append((st.mean(h.values()), c, pr))
            scored.sort(reverse=True)
            log(f"[v0] 후보 {sel_name} H_set {[round(x[0], 4) for x in scored]} → {scored[0][1]} 채택")
            prompt = scored[0][2]
    else:
        log("[stop] --prompt 또는 --generate-v0 가 필요하다")
        return 2
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

    # 재개 때도 다시 잰다 — 현재 프롬프트의 분절은 캐시에 있어 LLM 호출이 없고 QE 만 돈다.
    cur_rows, cur_sets, cur_h, cur_m = score_prompt(
        prompt, devA, labA, "iter 0" if start == 1 else f"iter {start - 1} 재개")
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
                base: tuple[str, dict] | None = None):
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
        prev = next((h for h in reversed(history) if h.get("diagnosis")), None)
        if prev:        # 직전 개정이 어디서 무너졌는지 Critic 도 본다 — 같은 방향의 반복을 끊는다
            crit_user["last_revision"] = {"iter": prev["iter"], "adopted": prev["adopted"],
                                          "delta": prev.get("gain") or prev.get("delta"),
                                          "edits": prev.get("edits"), **prev["diagnosis"]}
            if aj.is_near_miss(prev):
                crit_user["last_revision"]["near_miss"] = True
                log(f"[iter {it}] 직전 개정은 근소 기각(평균 {crit_user['last_revision']['delta']['mean']:+.4f}) "
                    f"— Critic·PE 에 near_miss 로 알린다")
        crit = gw.chat_json(aj.critic_system(), json.dumps(crit_user, ensure_ascii=False),
                            max_tokens=AGENT_MAX_TOKENS, purpose="critic")
        timing["critic"] = round(time.perf_counter() - t0, 1)
        findings = aj.clean_findings(crit, prompt)
        n_raw = len(aj.clean_findings(crit))
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

        # PE 는 프롬프트를 다시 쓰지 않고 번호 붙은 단위를 편집한다. 적용과 길이는 코드가 하고,
        # 단위마다 출처(들어온 이터·채택 Δ·Critic 지적 횟수)를 붙여 무엇을 갈아끼울지 고르게 한다.
        # 편집 단위(units)가 두 섹션 본문을 이미 담으므로 프롬프트 전체는 안 준다 — 동결 섹션 중
        # 측정 정의만 붙인다. 이력은 방향 반복을 막는 데 필요한 것만 요약한다.
        pe_user = {"fixed_sections": {h: aj.section_of(prompt, h) for h in ("[Role]", "[Scoring Rules]")},
                   "units": aj.units_with_provenance(prompt, prov),
                   "findings": findings, "history": aj.history_brief(history),
                   "size": aj.size_brief(prompt, target)}
        if base_tag:
            pe_user["base"] = base_tag["built_on"] + " — the units you were given already contain that revision's edit; candidates are measured against the current best prompt"
        if loss_bins:
            pe_user["loss_by_bin"] = loss_bins
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

        def resolve(pe_blob, role="free"):
            """`labeled_example` 참조를 실측 예시 본문으로 바꾸고 후보 역할 제약을 건다. 어기거나
            없는 id, 앞 후보가 이미 쓴 id 를 가리킨 편집은 로그에 남기고 뺀다."""
            if not isinstance(pe_blob, dict):
                return pe_blob
            edits, bad = aj.enforce_role(pe_blob.get("edits"), role)
            if examples:
                # 앞 후보가 쓴 예시는 없는 것으로 친다 — judge12 iter 1 은 후보 넷 중 셋이 같은
                # 예시(E4 → en_us_738)를 넣었다. 제약이 강한 역할은 선택지가 그것뿐이었다.
                offered = {k: v for k, v in examples.items() if k not in used_examples}
                edits, bad2 = aj.resolve_labeled_examples(edits, offered)
                bad += bad2
                used_examples.update(str(e["labeled_example"]) for e in edits
                                     if e.get("labeled_example"))
            for b in bad:
                log(f"[iter {it}] PE 편집 무시: edit {b['edit']} {b['reason']}")
            return {**pe_blob, "edits": edits}

        def revise(j: int, extra: dict, role: str = "free"):
            """PE 호출 한 번. 길이만 넘으면 편집별 실측 증감과 초과량을 붙여 한 번 되돌린다 — 콜
            하나가 이터레이션 하나를 통째로 날리는 것보다 싸다. (pe, cand, note, deltas, tries)."""
            t0 = time.perf_counter()
            pe = gw.chat_json(aj.engineer_system(target, len(prompt)),
                              json.dumps({**pe_user, **extra}, ensure_ascii=False),
                              max_tokens=AGENT_MAX_TOKENS, purpose="engineer")
            timing["engineer"] = round(timing.get("engineer", 0) + time.perf_counter() - t0, 1)
            save_usage(idir)
            pe = resolve(pe, role)
            cand, note, draft, deltas, skipped = aj.parse_edits(pe, prompt, budget)
            tries = [record(j, pe, deltas, skipped, draft, cand, note)]
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
                pe = resolve(pe, role)
                cand, note, draft, deltas, skipped = aj.parse_edits(pe, prompt, budget)
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
                for h in aj.FROZEN:
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
                pe2 = resolve(pe2, aj.SHORTEN_ROLE)
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
        for j in range(a.pe_candidates):
            extra = {}
            role = roles[j % len(roles)] if roles else "free"
            if role == "rewrite" and not v0_material:
                log(f"[iter {it}] 후보 {j}: rewrite 역할인데 v0 재료가 없다(--prompt 로 시작) — free 로")
                role = "free"
            if role not in ("free", "rewrite"):
                extra["constraint"] = role
            if findings:
                extra["primary_finding"] = findings[j % len(findings)]["diagnosis"]
            if cands:
                extra["sibling_candidates"] = [
                    {"changelog": c[2], "edits": aj.edit_summary(prompt, (c[0] or {}).get("edits"))}
                    for c in cands if c[1]]
                if pe_user.get("labeled_examples"):
                    extra["labeled_examples"] = [x for x in pe_user["labeled_examples"]
                                                 if x["id"] not in used_examples]
            if role != "free":
                log(f"[iter {it}] 후보 {j} 역할 {role}")
            pe, cand, note, deltas, tries = (rewrite(j) if role == "rewrite"
                                             else revise(j, extra, role))
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

        if len(valid) > 1:
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
            to_full = [sc["candidate"] for sc in sorted(screened, key=lambda x: -x["delta"]["mean"])
                       if sc["delta"]["mean"] > 0][:max(1, a.full_score_max)]
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
        else:
            to_full = [valid[0][0]]

        def full_score(j):
            _j, pe_j, cand_j, note_j, deltas_j = next(v for v in valid if v[0] == j)
            c_rows, c_sets, c_h, c_m = score_prompt(cand_j, devA, labA, f"iter {it} 후보 {j}")
            excl = (example_sentences(best, devA) | example_sentences(prompt, devA)
                    | example_sentences(cand_j, devA))
            keys = sorted(k for k in set(c_h) & set(cur_h) if k[0] not in excl)
            boot = hset.paired_bootstrap([c_h[k] for k in keys], [cur_h[k] for k in keys],
                                         clusters=[i for i, _k in keys])
            log(f"[iter {it}] 후보 {j} 본채점 Δ {boot['mean']:+.4f} [{boot['lo']:+.4f}, "
                f"{boot['hi']:+.4f}] 짝 {boot['n']} / 문장 {boot['n_clusters']}")
            return {"j": j, "pe": pe_j, "cand": cand_j, "note": note_j, "deltas": deltas_j,
                    "rows": c_rows, "sets": c_sets, "h": c_h, "m": c_m, "excl": excl, "boot": boot}

        t0 = time.perf_counter()
        fulls = [full_score(j) for j in to_full]
        timing["score_candidate"] = round(time.perf_counter() - t0, 1)
        chosen = max(fulls, key=lambda f: f["boot"]["lo"])
        for f in fulls:
            if f is not chosen:     # 본채점까지 갔지만 하한이 낮은 후보 — 이력에 남긴다
                history.append({"iter": it, "candidate": f["j"], "adopted": False,
                                "full_scored": True, "delta": f["boot"], "changelog": f["note"],
                                "edits": aj.edit_summary(prompt, (f["pe"] or {}).get("edits")),
                                **base_tag})
        if len(fulls) > 1:
            log(f"[iter {it}] 본채점 {len(fulls)}개 중 하한 최고 후보 {chosen['j']} 선택")
        pe, cand, note, deltas = chosen["pe"], chosen["cand"], chosen["note"], chosen["deltas"]
        c_rows, c_sets, c_h, c_m, excl, boot = (chosen["rows"], chosen["sets"], chosen["h"],
                                                 chosen["m"], chosen["excl"], chosen["boot"])
        (idir / "violations.json").write_text(
            json.dumps({"candidate": violation_summary(c_rows)}, ensure_ascii=False, indent=1),
            encoding="utf-8")
        if excl:
            log(f"[iter {it}] 예시로 들어간 dev-A 문장 {len(excl)}개는 판정에서 뺀다")
        keys = sorted(k for k in set(c_h) & set(cur_h) if k[0] not in excl)
        verdict = decide(boot)
        gain = boot
        log(f"[iter {it}] Δ H_set {boot['mean']:+.4f} [{boot['lo']:+.4f}, {boot['hi']:+.4f}] "
            f"짝 {boot['n']} / 문장 {boot['n_clusters']} → {verdict}")

        if verdict == "confirm":
            log("[iter] 경계선 — 신선한 캐시로 재채점")
            tmp = JsonCache(run_dir / "cache" / f"segment_confirm_{it}.json")
            rows2, _m2 = evaluate(gw, cand, devA, labA, spaced, min_gap, t_grid, tmp,
                                  a.workers, a.batch_size, seg_effort, k_samples=a.k_samples)
            tmp.flush()
            s2 = policy_sets(rows2, devA, spaced, min_gap, a.min_chunk, a.max_k)
            h2 = hset_of(devA, labA, s2)
            k2 = sorted(k for k in set(h2) & set(cur_h) if k[0] not in excl)
            gain = hset.paired_bootstrap([h2[k] for k in k2], [cur_h[k] for k in k2],
                                         clusters=[i for i, _k in k2])
            log(f"[iter {it}] 재채점 Δ {gain['mean']:+.4f} [{gain['lo']:+.4f}, {gain['hi']:+.4f}]")
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

        # 부검 — 어느 (문장, k) 가 어떻게 움직였는지 세고, 왜 그랬는지 한 문단을 받는다.
        edits_now = aj.edit_summary(prompt, (pe or {}).get("edits"))
        diag = revision_diagnosis(devA, cur_h, c_h, cur_sets, c_sets, spaced)
        diag = add_vs_base(diag, c_h, c_sets)
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

    case_state = {"prompt": None, "sets": None, "h": None}
    # 이미 보인 사례 문장 — 채택될 때까지 누적해서 뺀다. 직전 이터만 빼면 프롬프트가 안 바뀌는
    # 동안 두 이터 주기로 같은 12문장이 돌아온다(judge11 iter 1 = iter 3). 채택되면 손해 순위가
    # 새로 생기므로 비운다.
    seen_case_ids: set[str] = set()
    if start > 1:
        # 재개 — 집합은 메모리에만 있으므로 마지막 채택 뒤 이터들의 cases.json 에서 다시 모은다
        # (judge12 iter 2 재개에서 iter 1 과 같은 12문장이 다시 뽑혔다).
        last_adopt = max([h["iter"] for h in history if h.get("adopted")], default=0)
        for k in range(last_adopt + 1, start):
            cp = run_dir / f"iter_{k:02d}" / "cases.json"
            if cp.exists():
                seen_case_ids |= {c["id"] for c in json.loads(cp.read_text(encoding="utf-8"))}
        if seen_case_ids:
            log(f"[resume] 이미 보인 사례 문장 {len(seen_case_ids)}개 복원 (이터 {last_adopt + 1}~{start - 1})")
    for it in range(start, a.iterations + 1):
        snapshot_state(it - 1)
        idir = run_dir / f"iter_{it:02d}"
        idir.mkdir(parents=True, exist_ok=True)
        t_iter = time.perf_counter()

        base = near_miss_base(history, run_dir)
        edit_src = prompt
        if base:
            d = base[1].get("gain") or base[1].get("delta")
            edit_src = base[0]
            log(f"[iter {it}] 근소 기각본(iter {base[1]['iter']}, {d['mean']:+.4f} [{d['lo']:+.4f}, "
                f"{d['hi']:+.4f}]) 을 편집 기반으로 — 판정은 현재 채택본 대비 그대로")
        if scheme3:
            # 사례는 train 에서 뽑는다. 편집 기반이 바뀐 이터에만 다시 잰다(같으면 캐시 적중).
            if case_state["prompt"] != edit_src:
                _cr, case_state["sets"], case_state["h"], _cm = score_prompt(
                    edit_src, case_sents, case_lab, f"iter {it} train")
                case_state["prompt"] = edit_src
            case_sets, case_h = case_state["sets"], case_state["h"]
        else:
            case_sets, case_h = cur_sets, cur_h

        def pieces_tr(i, cut):
            u = L.units_of(case_sents[i].text, spaced)
            out = {}
            for tgt in targets:
                out[tgt] = [scorer.piece(tgt, u, x, y)
                            for x, y in hset.pieces_of(u, cut, spaced)]
            return out

        cases, loss_bins = build_cases(case_sents, case_lab, case_sets, ora_C, case_h, ora_hC,
                                       spaced, min_gap, a.n_cases, pieces_tr,
                                       exclude_ids=seen_case_ids, alloc=a.case_alloc)
        seen_case_ids |= {c["id"] for c in cases}
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
        adopted = propose(it, idir, cases, budget, target, timing, examples, loss_bins, base=base)
        if adopted:
            cand, c_sets, c_h, c_m, gain = adopted
            provenance = aj.adopt_provenance(provenance, cand, it, gain)
            prompt, cur_sets, cur_h, cur_m = cand, c_sets, c_h, c_m
            seen_case_ids = set()
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
                    cur_rows, cur_sets, cur_h, cur_m = score_prompt(prompt, devA, labA,
                                                                    f"iter {it} 롤백 후")
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
    if not a.skip_final and prompt == v0_path.read_text(encoding="utf-8"):
        log("[최종] 채택된 개정이 없다 — v0 그대로라 최종 test 를 생략한다")
    elif not a.skip_final:
        t0 = time.perf_counter()
        rows_t, _sets_t, h_t, m_t = score_prompt(prompt, test_sents, lab_test, "최종 test")
        ora_t = oracle_sets(lab_test, test_sents, spaced, min_gap, a.min_chunk, a.max_k)
        ora_h_t = hset_of(test_sents, lab_test, ora_t)
        by_bin = {"prompt": by_latency(test_sents, h_t, spaced),
                  "oracle": by_latency(test_sents, ora_h_t, spaced)}
        means = {"prompt": round(st.mean(h_t.values()), 4),
                 "oracle": round(st.mean(ora_h_t.values()), 4)}
        v0_prompt = v0_path.read_text(encoding="utf-8")
        boot_v0 = None
        if v0_prompt != prompt:
            _r0, _s0, h0, _m0 = score_prompt(v0_prompt, test_sents, lab_test, "최종 test v0")
            by_bin["v0"] = by_latency(test_sents, h0, spaced)
            means["v0"] = round(st.mean(h0.values()), 4)
            kk = sorted(set(h_t) & set(h0))
            boot_v0 = hset.paired_bootstrap([h_t[k] for k in kk], [h0[k] for k in kk],
                                            clusters=[i for i, _k in kk])
        (run_dir / "curve.json").write_text(
            json.dumps({"split": "test_b" if scheme3 else "test", "means": means,
                        "by_bin": by_bin, "vs_v0": boot_v0},
                       ensure_ascii=False, indent=1), encoding="utf-8")
        (run_dir / "violations_test.json").write_text(
            json.dumps(violation_summary(rows_t), ensure_ascii=False, indent=1), encoding="utf-8")
        save_usage(run_dir / "final")
        (run_dir / "final_report.md").write_text(
            final_report(a.run_id, means, by_bin, boot_v0, history,
                         spent_before + gw.usage.snapshot()["cost"], m_t["format_pass_rate"]),
            encoding="utf-8")
        log(f"[최종] {'test-B' if scheme3 else 'test'} H_set {means['prompt']:.4f} / 오라클 {means['oracle']:.4f}"
            + (f" / v0 {means['v0']:.4f} Δ {boot_v0['mean']:+.4f} "
               f"[{boot_v0['lo']:+.4f}, {boot_v0['hi']:+.4f}]" if boot_v0 else " (v0 그대로)")
            + f" / {by_bin['prompt']} / {time.perf_counter() - t0:.0f}초")

    u = gw.usage.snapshot()
    log(f"[끝] {names[0]} H_set {st.mean(cur_h.values()):.4f} / 오라클 {st.mean(ora_hA.values()):.4f} "
        f"/ 비용 ${spent_before + u['cost']:.2f} (이번 실행 ${u['cost']:.2f}, {u['calls']} 호출)")
    gw.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
