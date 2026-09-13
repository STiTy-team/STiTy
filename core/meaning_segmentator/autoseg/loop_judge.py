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
import json
import statistics as st
import time
from pathlib import Path

from .infra.gateway import Gateway, add_provider_args
from .loop import target_is_spaced
from .loop_distill import evaluate
from .paths import RUNS_DIR
from .runtime import agents_judge as aj
from .runtime import data, hset, labels as L, metrics
from .runtime.pipeline import JsonCache, LocalTranslator, to_lang_code

AGENT_MAX_TOKENS = 24000


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

def error_type(dropped: list[dict], added: list[dict]) -> str:
    """뺐어야 할 자리와 넣었어야 할 자리의 **경계별 H** 를 견줘 오류를 가른다.

    정책이 고른 자리가 전부 목표가 고른 자리보다 경계별로도 낮으면, 그 자리는 혼자 놓고
    봐도 나쁘다 — 원문에서 판단 가능한 실수(boundary).
    정책이 고른 자리 중 목표가 고른 자리만큼(또는 그보다) 높은 것이 있으면, 혼자서는 좋은
    자리인데 조합에서 틀린 것이다(interaction).

    목표가 그리디 오라클이면 interaction 은 드물다 — 그리디는 경계별 H 상위 k개를 고르니
    보통 뺀 자리가 넣은 자리보다 낮다. 다만 `min_gap` 이 간격을 강제해 더 높은 자리를
    건너뛰는 일이 있어 0 은 아니다 (스모크 4사례 중 1건). 이 구분이 제대로 사는 것은
    목표가 집합 탐색 최적일 때다 (`--target-sets`).
    """
    if not dropped or not added:
        return "boundary"
    return "interaction" if max(d["H"] for d in dropped) >= min(d["H"] for d in added) \
        else "boundary"


def build_cases(sents: list, lab: dict, pol: dict, ora: dict, pol_h: dict, ora_h: dict,
                spaced: bool, min_gap: int, n_cases: int, pieces_tr=None) -> list[dict]:
    """손해가 큰 (문장, T) 상위 `n_cases` 를 반사실 쌍으로 만든다."""
    gaps = sorted(((ora_h[k] - pol_h[k], k) for k in pol if k in ora_h and k in pol_h),
                  reverse=True)
    cases = []
    for gap, (i, kk) in gaps[:n_cases]:
        if gap <= 0:
            break
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
        case = {"id": s.id, "cuts": kk, "avg_chunk": round(hset.chunk_len(len(u), kk), 1),
                "gap": round(gap, 4), "error_type": error_type(dropped, added),
                "policy": {"cut": list(pol[(i, kk)]), "H_set": round(pol_h[(i, kk)], 4),
                           "text": _marked(u, pol[(i, kk)], spaced)},
                "target": {"cut": list(ora[(i, kk)]), "H_set": round(ora_h[(i, kk)], 4),
                           "text": _marked(u, ora[(i, kk)], spaced)},
                "diff": {"dropped": dropped, "added": added}}
        if pieces_tr:
            case["policy"]["pieces"] = pieces_tr(i, pol[(i, kk)])
            case["target"]["pieces"] = pieces_tr(i, ora[(i, kk)])
        cases.append(case)
    return cases


def _marked(units: list[str], cut: tuple[int, ...], spaced: bool) -> str:
    out = units[0]
    for j in range(1, len(units)):
        sep = " ‖ " if j in cut else (" " if spaced else "")
        out += sep + units[j]
    return out


def by_latency(sents: list, h: dict, spaced: bool,
               bins=(3, 5, 7, 10, 99)) -> dict[str, float]:
    """(문장, k) 값들을 평균 조각 길이로 묶어 지연축 곡선으로 — 보고용.

    판정은 (문장, k) 짝으로 하고 문장 단위로 재추출한다. 이쪽은 사람이 읽을 표다.
    """
    acc: dict[str, list[float]] = {}
    for (i, k), v in h.items():
        n = len(L.units_of(sents[i].text, spaced))
        c = hset.chunk_len(n, k)
        lab = next((f"≤{b}" for b in bins if c <= b), f">{bins[-2]}")
        acc.setdefault(lab, []).append(v)
    return {k: round(st.mean(v), 4) for k, v in sorted(acc.items())}


# ── 채택 판정 ───────────────────────────────────────────────────────────

def decide(boot: dict, strong: float = 0.005) -> str:
    """CI 하한으로 가른다. 종전 1 se 문턱은 재채점 잡음(+0.023±0.017)이 그냥 넘었다."""
    if boot["lo"] > strong:
        return "accept"
    if boot["lo"] > 0:
        return "confirm"
    return "reject"


# ── 실행 ────────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--from-run", required=True, help="분할·라벨·캐시를 물려받을 런")
    p.add_argument("--run-id", required=True)
    p.add_argument("--prompt", required=True, help="시작 프롬프트 파일")
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
                        "탐색 최적이 된다 — 상호작용 오류는 이때만 보인다")
    p.add_argument("--checkpoint-every", type=int, default=2, help="0 이면 dev-B 를 안 본다")
    p.add_argument("--prompt-budget", type=int, default=9000, help="프롬프트 길이 상한(자)")
    p.add_argument("--k-samples", type=int, default=3)
    p.add_argument("--model", default="gpt-5-mini")
    p.add_argument("--seg-reasoning-effort", default="medium")
    p.add_argument("--agent-reasoning-effort", default="medium")
    p.add_argument("--batch-size", type=int, default=6)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--comet-batch-size", type=int, default=32)
    p.add_argument("--budget", type=float, default=25.0)
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

    dev, train = load("dev"), load("train")
    lab_dev = json.loads((src / "oracle_labels_dev.json").read_text(encoding="utf-8"))
    lab_train = json.loads((src / "oracle_labels_train.json").read_text(encoding="utf-8"))
    devA, devB = dev[:a.dev_a], dev[a.dev_a:] + train
    labA = {t: per[:a.dev_a] for t, per in lab_dev.items()}
    labB = {t: lab_dev[t][a.dev_a:] + lab_train[t] for t in lab_dev}
    log(f"[data] dev-A {len(devA)} / dev-B {len(devB)} / k 1~{a.max_k} "
        f"(평균 조각 ≥ {a.min_chunk}) / min_gap {min_gap}"
        f"{' (런 설정 %d 을 덮어씀)' % cfg['min_gap'] if min_gap != cfg['min_gap'] else ''}"
        f" / 타깃 {targets}")

    gw = Gateway.from_args(a, model=a.model, budget=a.budget,
                           reasoning_effort=a.agent_reasoning_effort)
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

    def hset_of(sents, lab, sets: dict) -> dict:
        keys = sorted(sets)
        texts = [s.text for s in sents]

        def contra_of(i, j):
            return lab[targets[0]][i]["contra"][j - 1]

        vals = scorer.score(texts, [(i, sets[(i, T)]) for i, T in keys], contra_of)
        return dict(zip(keys, vals))

    prompt = Path(a.prompt).read_text(encoding="utf-8")
    history: list[dict] = []
    cur_rows, cur_h = None, None
    if a.target_sets:
        ora_A, ora_hA = search_sets(Path(a.target_sets), t_grid, len(devA))
        kind = f"탐색 최적 ({Path(a.target_sets).name})"
    else:
        ora_A = oracle_sets(labA, devA, spaced, min_gap, a.min_chunk, a.max_k)
        ora_hA = hset_of(devA, labA, ora_A)
        kind = "그리디 오라클 — 상호작용 오류는 드물게만 보인다"
    log(f"[목표] {kind}: dev-A H_set 평균 {st.mean(ora_hA.values()):.4f} ({len(ora_hA)} 짝)")
    checkpoint = None

    def score_prompt(pr, sents, lab, tag):
        rows, m = evaluate(gw, pr, sents, lab, spaced, min_gap, t_grid, seg_cache,
                           a.workers, a.batch_size, seg_effort, k_samples=a.k_samples)
        seg_cache.flush()
        sets = policy_sets(rows, sents, spaced, min_gap, a.min_chunk, a.max_k)
        h = hset_of(sents, lab, sets)
        log(f"[{tag}] H_set {st.mean(h.values()):.4f} {by_latency(sents, h, spaced)} / "
            f"overlap {m['overlap']} / fmt {m['format_pass_rate']} / "
            f"누적 ${gw.usage.snapshot()['cost']:.2f}")
        return rows, sets, h, m

    cur_rows, cur_sets, cur_h, cur_m = score_prompt(prompt, devA, labA, "iter 0")
    (run_dir / "iter_00").mkdir(parents=True, exist_ok=True)
    (run_dir / "prompt_v0.txt").write_text(prompt, encoding="utf-8")

    for it in range(1, a.iterations + 1):
        idir = run_dir / f"iter_{it:02d}"
        idir.mkdir(parents=True, exist_ok=True)

        def pieces_tr(i, cut):
            u = L.units_of(devA[i].text, spaced)
            out = {}
            for tgt in targets:
                out[tgt] = [scorer._tr_cache[tgt][(i, x, y)]
                            for x, y in hset.pieces_of(u, cut, spaced)]
            return out

        cases = build_cases(devA, labA, cur_sets, ora_A, cur_h, ora_hA, spaced, min_gap,
                            a.n_cases, pieces_tr)
        (idir / "cases.json").write_text(json.dumps(cases, ensure_ascii=False, indent=1),
                                         encoding="utf-8")
        kinds = {k: sum(1 for c in cases if c["error_type"] == k)
                 for k in ("boundary", "interaction")}
        log(f"[iter {it}] 사례 {len(cases)} (경계 {kinds['boundary']} / 상호작용 "
            f"{kinds['interaction']}), 손해 상위 {cases[0]['gap'] if cases else 0}")
        if not cases:
            log("[stop] 오라클보다 못한 (문장,T) 가 없다")
            break

        crit = gw.chat_json(aj.critic_system(),
                            json.dumps({"cases": cases, "prompt": prompt}, ensure_ascii=False),
                            max_tokens=AGENT_MAX_TOKENS, purpose="critic")
        findings = aj.clean_findings(crit)
        (idir / "critique.json").write_text(json.dumps({"raw": crit, "findings": findings},
                                                       ensure_ascii=False, indent=1),
                                            encoding="utf-8")
        if not findings:
            log("[iter] Critic 이 쓸 수 있는 finding 을 안 냈다 — 건너뛴다")
            continue

        pe = gw.chat_json(aj.engineer_system(a.prompt_budget, len(prompt)),
                          json.dumps({"current_prompt": prompt, "findings": findings,
                                      "history": history}, ensure_ascii=False),
                          max_tokens=AGENT_MAX_TOKENS, purpose="engineer")
        cand, note = aj.parse_prompt(pe, prompt, a.prompt_budget)
        if cand is None:
            log(f"[iter] PE 출력 반려: {note}")
            history.append({"iter": it, "adopted": False, "reason": note})
            continue

        _, c_sets, c_h, c_m = score_prompt(cand, devA, labA, f"iter {it} 후보")
        keys = sorted(set(c_h) & set(cur_h))
        boot = hset.paired_bootstrap([c_h[k] for k in keys], [cur_h[k] for k in keys],
                                     clusters=[i for i, _k in keys])
        verdict = decide(boot)
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
            k2 = sorted(set(h2) & set(cur_h))
            b2 = hset.paired_bootstrap([h2[k] for k in k2], [cur_h[k] for k in k2],
                                       clusters=[i for i, _k in k2])
            log(f"[iter {it}] 재채점 Δ {b2['mean']:+.4f} [{b2['lo']:+.4f}, {b2['hi']:+.4f}]")
            verdict = "accept" if b2["lo"] > 0 else "reject"

        history.append({"iter": it, "adopted": verdict == "accept", "delta": boot,
                        "changelog": note, "findings": [f["diagnosis"] for f in findings]})
        (idir / "result.json").write_text(json.dumps(history[-1], ensure_ascii=False, indent=1),
                                          encoding="utf-8")
        (idir / "prompt.txt").write_text(cand, encoding="utf-8")
        if verdict == "accept":
            prompt, cur_sets, cur_h, cur_m = cand, c_sets, c_h, c_m
            (run_dir / "best_prompt.txt").write_text(prompt, encoding="utf-8")

        if a.checkpoint_every and it % a.checkpoint_every == 0:
            _, _s, hB, _mB = score_prompt(prompt, devB, labB, f"iter {it} dev-B")
            val = st.mean(hB.values())
            if checkpoint and val < checkpoint["value"] - 1e-6:
                log(f"[체크포인트] dev-B {val:.4f} < 직전 {checkpoint['value']:.4f} — 롤백")
                prompt = checkpoint["prompt"]
                cur_rows, cur_sets, cur_h, cur_m = score_prompt(prompt, devA, labA,
                                                                f"iter {it} 롤백 후")
            else:
                checkpoint = {"iter": it, "value": val, "prompt": prompt}
                log(f"[체크포인트] dev-B {val:.4f} 저장")
            (run_dir / "checkpoint.json").write_text(
                json.dumps({k: v for k, v in (checkpoint or {}).items() if k != "prompt"},
                           ensure_ascii=False, indent=1), encoding="utf-8")

    (run_dir / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=1),
                                          encoding="utf-8")
    (run_dir / "best_prompt.txt").write_text(prompt, encoding="utf-8")
    u = gw.usage.snapshot()
    log(f"[끝] dev-A H_set {st.mean(cur_h.values()):.4f} / 오라클 {st.mean(ora_hA.values()):.4f} "
        f"/ 비용 ${u['cost']:.2f} ({u['calls']} 호출)")
    gw.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
