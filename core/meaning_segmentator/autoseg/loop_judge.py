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


LATENCY_BINS = (3, 5, 7, 10, 99)
CONTRA_KILL = 0.5


def latency_bin(n_units: int, k: int, bins=LATENCY_BINS) -> str:
    c = hset.chunk_len(n_units, k)
    return next((f"≤{b}" for b in bins if c <= b), f">{bins[-2]}")


def pick_cases(gaps: list[tuple[float, tuple[int, int]]], bin_of, n_cases: int) -> list:
    """손해가 난 (문장, k) 를 지연 구간별로 고르게, 문장당 하나씩 고른다.

    손해 상위만 뽑으면 한 구간에 몰린다 — judge05 는 12개가 전부 평균 조각 2~3.5어절에 손해
    0.67~0.80 이었고, 모순 절단 하나로 집합이 0 이 된 경우뿐이었다. Critic 은 그걸 "뒤집힐 수
    있는 자리를 더 피하라" 로 일반화했고, 넣을 때마다 짧은 조각 구간이 떨어졌다(iter 2~5 전부
    기각). 구간마다 손해 큰 순으로 줄 세워 한 바퀴씩 돌며 뽑는다."""
    by_bin: dict[str, list] = {}
    for gap, key in sorted(gaps, reverse=True):
        if gap > 0:
            by_bin.setdefault(bin_of(key), []).append((gap, key))
    queues = [by_bin[b] for b in sorted(by_bin, key=lambda b: float(b.strip("≤>")))]
    picked, seen = [], set()
    while len(picked) < n_cases and any(queues):
        for q in queues:
            while q and q[0][1][0] in seen:
                q.pop(0)
            if q and len(picked) < n_cases:
                gap, key = q.pop(0)
                picked.append((gap, key))
                seen.add(key[0])
    return picked


def build_cases(sents: list, lab: dict, pol: dict, ora: dict, pol_h: dict, ora_h: dict,
                spaced: bool, min_gap: int, n_cases: int, pieces_tr=None) -> list[dict]:
    """손해가 난 (문장, k) 를 지연 구간별로 고르게 골라(`pick_cases`) 반사실 쌍으로 만든다."""
    gaps = [(ora_h[k] - pol_h[k], k) for k in pol if k in ora_h and k in pol_h]
    n_units = {i: len(L.units_of(sents[i].text, spaced)) for i in {k[0] for _g, k in gaps}}
    cases = []
    for gap, (i, kk) in pick_cases(gaps, lambda key: latency_bin(n_units[key[0]], key[1]),
                                   n_cases):
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
                "gap": round(gap, 4), "error_type": error_type(dropped, added),
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
    return cases


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
    for p in run_dir.glob("iter_*/metrics.json"):
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
                        "탐색 최적이 된다 — 상호작용 오류는 이때만 보인다")
    p.add_argument("--checkpoint-every", type=int, default=2, help="0 이면 dev-B 를 안 본다")
    p.add_argument("--growth-per-iter", type=float, default=0.05,
                   help="PE 개정본이 직전 채택본보다 길어질 수 있는 비율. 넘으면 PE 가 근거 약한 "
                        "단위를 갈아끼우거나 압축해야 한다")
    p.add_argument("--growth-ceiling", type=float, default=1.3,
                   help="런 전체 길이 천장 — 시작 프롬프트 길이의 배수. 증가율이 이터마다 쌓이는 "
                        "것을 막는다")
    p.add_argument("--k-samples", type=int, default=3)
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
        kind = "그리디 오라클 — 상호작용 오류는 드물게만 보인다"
    log(f"[목표] {kind}: dev-A H_set 평균 {st.mean(ora_hA.values()):.4f} ({len(ora_hA)} 짝)")
    checkpoint = None

    def score_prompt(pr, sents, lab, tag):
        rows, m = evaluate(gw, pr, sents, lab, spaced, min_gap, t_grid, seg_cache,
                           a.workers, a.batch_size, seg_effort, k_samples=a.k_samples)
        seg_cache.flush()
        sets = policy_sets(rows, sents, spaced, min_gap, a.min_chunk, a.max_k)
        h = hset_of(sents, lab, sets)
        # 번역 캐시는 20건마다만 쓴다. 채점이 끝날 때 비우지 않으면 프로세스가 죽을 때 남은 번역이
        # 사라져 다음 실행이 다시 만든다.
        for tr in translators.values():
            if tr.cache is not None:
                tr.cache.flush()
        log(f"[{tag}] H_set {st.mean(h.values()):.4f} {by_latency(sents, h, spaced)} / "
            f"overlap {m['overlap']} / fmt {m['format_pass_rate']} / "
            f"누적 ${gw.usage.snapshot()['cost']:.2f}")
        return rows, sets, h, m

    v0_path = run_dir / "prompt_v0.txt"
    if a.prompt:
        prompt = Path(a.prompt).read_text(encoding="utf-8")
    elif v0_path.exists():
        prompt = v0_path.read_text(encoding="utf-8")
        log("[v0] 런 산출물 재사용")
    elif a.generate_v0:
        # **프로파일·예시 문장은 어느 분할에도 안 쓰인 문장에서 뽑는다.** dev 에서 뽑으면 그
        # 문장이 프롬프트 안에 그대로 들어간 채로 dev 에서 채점된다 — 라벨 유출은 아니지만
        # 그 문장에서만 유리해진다.
        pool_ids = {x.id for x in dev} | {x.id for x in train} | {
            r["id"] for r in json.loads((src / "data/test.json").read_text(encoding="utf-8"))}
        spare = [x for x in data.load(cfg["dataset"]) if x.id not in pool_ids]
        if len(spare) < 20:
            log(f"[v0] 분할 밖 문장이 {len(spare)}개뿐 — dev-A 에서 뽑는다")
            spare = devA
        log(f"[v0] 분할 밖 문장 {len(spare)}개에서 프로파일 20 / 예시 8")
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
        user = (f"Language profile:\n{json.dumps(profile, ensure_ascii=False, indent=2)}\n\n"
                + (facts + "\n\n" if facts else "")
                + "Sample source sentences to build the examples from:\n"
                + "\n".join(f"{i+1}. {x.text}" for i, x in enumerate(spare[20:28])) + "\n\n"
                + f"Copy this [Output Rules] section verbatim into the prompt:\n\n{out_rules}")
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
            scored = []
            for c, pr in enumerate(cands):
                _r, _s, h, _m = score_prompt(pr, devA, labA, f"v0 후보 {c}")
                save_usage(run_dir / "iter_00")
                scored.append((st.mean(h.values()), c, pr))
            scored.sort(reverse=True)
            log(f"[v0] 후보 H_set {[round(x[0], 4) for x in scored]} → {scored[0][1]} 채택")
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
    save_usage(run_dir / f"iter_{start - 1:02d}")

    def snapshot_state(done: int) -> None:
        save_state(run_dir / STATE_FILE, done, prompt, history, checkpoint, v0_len,
                   spent_before + gw.usage.snapshot()["cost"], provenance)

    def propose(it: int, idir: Path, cases: list, budget: int):
        """Critic → PE → dev-A 채점·판정 한 번. 채택이면 (개정본, 절단집합, H, 지표, 채택 근거 Δ),
        아니면 None. 반려·기각 어느 쪽이든 돌아와야 뒤의 체크포인트가 돈다 — 반려 경로의
        continue 가 체크포인트를 건너뛰어 judge03·04 는 dev-B 를 한 번도 못 쟀다."""
        crit = gw.chat_json(aj.critic_system(),
                            json.dumps({"cases": cases, "prompt": prompt}, ensure_ascii=False),
                            max_tokens=AGENT_MAX_TOKENS, purpose="critic")
        findings = aj.clean_findings(crit)
        aj.count_critic_hits(provenance, findings)
        (idir / "critique.json").write_text(json.dumps({"raw": crit, "findings": findings},
                                                       ensure_ascii=False, indent=1),
                                            encoding="utf-8")
        save_usage(idir)
        if not findings:
            log("[iter] Critic 이 쓸 수 있는 finding 을 안 냈다 — 건너뛴다")
            return None

        # PE 는 프롬프트를 다시 쓰지 않고 번호 붙은 단위를 편집한다. 적용과 길이는 코드가 하고,
        # 단위마다 출처(들어온 이터·채택 Δ·Critic 지적 횟수)를 붙여 무엇을 갈아끼울지 고르게 한다.
        pe_user = {"current_prompt": prompt,
                   "units": aj.units_with_provenance(prompt, provenance),
                   "findings": findings, "history": history,
                   "size": aj.size_brief(prompt, budget)}
        pe = gw.chat_json(aj.engineer_system(budget, len(prompt)),
                          json.dumps(pe_user, ensure_ascii=False),
                          max_tokens=AGENT_MAX_TOKENS, purpose="engineer")
        save_usage(idir)
        cand, note, draft, deltas, skipped = aj.parse_edits(pe, prompt, budget)

        def record(pe_blob, deltas, skipped, draft, cand, note):
            if skipped:
                log(f"[iter {it}] PE 편집 {len(skipped)}개 무시: "
                    + "; ".join(f"edit {s['edit']} {s['reason']}" for s in skipped[:3]))
            return {"edits": (pe_blob or {}).get("edits"), "deltas": deltas, "skipped": skipped,
                    "result_chars": len(draft) if draft else None,
                    "errors": None if cand else note}

        tries = [record(pe, deltas, skipped, draft, cand, note)]
        if cand is None and aj.only_too_long(note):
            # 길이만 넘은 편집은 편집별 실측 증감과 초과량을 붙여 한 번 되돌린다 — 콜 하나가
            # 이터레이션 하나를 통째로 날리는 것보다 싸다.
            fb = aj.edit_feedback(draft, prompt, budget, deltas)
            log(f"[iter {it}] PE 편집 결과 {len(draft)} > {budget} — 편집별 증감과 초과량 "
                f"{fb['over_by']}자를 알려 한 번 더")
            pe = gw.chat_json(aj.engineer_system(budget, len(prompt)),
                              json.dumps({**pe_user, "your_previous_edits": tries[0]["edits"],
                                          "size_feedback": fb}, ensure_ascii=False),
                              max_tokens=AGENT_MAX_TOKENS, purpose="engineer:shorten")
            save_usage(idir)
            cand, note, draft, deltas, skipped = aj.parse_edits(pe, prompt, budget)
            tries.append(record(pe, deltas, skipped, draft, cand, note))
        (idir / "pe_edits.json").write_text(json.dumps(tries, ensure_ascii=False, indent=1),
                                            encoding="utf-8")
        if cand is None:
            log(f"[iter] PE 출력 반려: {note} / 누적 ${gw.usage.snapshot()['cost']:.2f}")
            history.append({"iter": it, "adopted": False, "reason": note,
                            "edits": aj.edit_summary(prompt, (pe or {}).get("edits"))})
            return None
        log(f"[iter {it}] PE 편집 {len(deltas)}개 → {len(cand)}자 / 상한 {budget}자")

        _, c_sets, c_h, c_m = score_prompt(cand, devA, labA, f"iter {it} 후보")
        keys = sorted(set(c_h) & set(cur_h))
        boot = hset.paired_bootstrap([c_h[k] for k in keys], [cur_h[k] for k in keys],
                                     clusters=[i for i, _k in keys])
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
            k2 = sorted(set(h2) & set(cur_h))
            gain = hset.paired_bootstrap([h2[k] for k in k2], [cur_h[k] for k in k2],
                                         clusters=[i for i, _k in k2])
            log(f"[iter {it}] 재채점 Δ {gain['mean']:+.4f} [{gain['lo']:+.4f}, {gain['hi']:+.4f}]")
            verdict = "accept" if gain["lo"] > 0 else "reject"

        history.append({"iter": it, "adopted": verdict == "accept", "delta": boot, "gain": gain,
                        "changelog": note, "findings": [f["diagnosis"] for f in findings],
                        "edits": aj.edit_summary(prompt, (pe or {}).get("edits"))})
        (idir / "result.json").write_text(json.dumps(history[-1], ensure_ascii=False, indent=1),
                                          encoding="utf-8")
        (idir / "prompt.txt").write_text(cand, encoding="utf-8")
        return (cand, c_sets, c_h, c_m, gain) if verdict == "accept" else None

    for it in range(start, a.iterations + 1):
        snapshot_state(it - 1)
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

        budget = length_cap(len(prompt), v0_len, a.growth_per_iter, a.growth_ceiling)
        log(f"[iter {it}] 길이 상한 {budget}자 (직전 채택본 {len(prompt)}자)")
        adopted = propose(it, idir, cases, budget)
        if adopted:
            cand, c_sets, c_h, c_m, gain = adopted
            provenance = aj.adopt_provenance(provenance, cand, it, gain)
            prompt, cur_sets, cur_h, cur_m = cand, c_sets, c_h, c_m
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
        save_usage(idir)
    else:
        snapshot_state(a.iterations)

    (run_dir / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=1),
                                          encoding="utf-8")
    (run_dir / "best_prompt.txt").write_text(prompt, encoding="utf-8")
    u = gw.usage.snapshot()
    log(f"[끝] dev-A H_set {st.mean(cur_h.values()):.4f} / 오라클 {st.mean(ora_hA.values()):.4f} "
        f"/ 비용 ${spent_before + u['cost']:.2f} (이번 실행 ${u['cost']:.2f}, {u['calls']} 호출)")
    gw.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
