"""`H_set` — 절단집합 하나를 통째로 재는 자. 정책·오라클·탐색이 같은 눈금 위에 놓인다.

    H_set(S) = CometKiwi(원문 전체, MT(조각1) ⊕ … ⊕ MT(조각k)) × (1 − max contra(S))

경계별 라벨(`labels.label_value`)은 절단 **하나**를 가정하고 잰다 — 조각이 둘인 그림이다.
실제 절단은 조각 k+1개를 만들고, 각 조각의 양옆은 이웃 절단이 정한다. 그 불일치가 실측으로
크다: test 100문장에서 경계별 상위 k(그리디)를 조합 탐색이 이긴 폭이 T=4 +0.0284,
T=6 +0.0216, gold(COMET-DA)로도 네 T 전부 유의했다 (`AUTOSEG_DETAILS.md`).

**참조 번역이 필요 없다.** 사람 참조 COMET-DA 는 관문에서 빼고 체크포인트 감사로만 쓴다 —
참조를 두면 어순을 단조화한 좋은 분절이 감점되기 때문이다.

조각은 후보 자리 사이의 연속 구간뿐이라 서로 크게 겹친다. 번역은 구간 단위로 한 번만
하고 캐시에 남긴다 (`translate_<code>.json`). 집합마다 드는 것은 QE 1회다.
"""

from __future__ import annotations

import statistics as st
from dataclasses import dataclass, field

from .labels import join_units, units_of
from .pipeline import tag_positions, truncate

SCALE = 10000


def seg_with(units: list[str], scores: dict[int, int], spaced: bool) -> str:
    """`loop_distill._seg_with` 와 같은 표기 — 위치 j 앞에 `<SEG:점수>` 를 박는다."""
    out = units[0]
    for j in range(1, len(units)):
        out = (f"{out} <SEG:{scores[j]}> {units[j]}" if j in scores
               else out + (" " if spaced else "") + units[j])
    return out


def cut_set(units: list[str], scores: dict[int, float], T: int, spaced: bool,
            min_gap: int) -> tuple[int, ...]:
    """점수에서 예산 T 의 절단집합을 뽑는다 — 상위 k개, `min_gap` 은 `truncate` 가 지킨다."""
    if not scores:
        return ()
    seg = seg_with(units, {j: int(round(v * SCALE)) for j, v in scores.items()}, spaced)
    cut, _ = truncate(seg, T, spaced, min_gap)
    pos, _ = tag_positions(cut, spaced)
    return tuple(sorted(pos))


def top_k_cuts(units: list[str], scores: dict[int, float], k: int,
               min_gap: int) -> tuple[int, ...]:
    """점수 상위 k개 자리. 동점은 앞쪽이 이긴다 (`truncate` 와 같은 규칙).

    T 대신 k 로 부르는 이유: T 는 문장 길이에 따라 k 를 정하는 함수라 격자를 몇 점 고르는
    순간 **그 깊이의 순위만** 채점된다. k 를 1부터 훑으면 순위 전체가 채점되고, LLM 비용은
    안 는다 — 점수 벡터는 문장당 한 번만 뽑기 때문이다.
    """
    if k <= 0 or not scores:
        return ()
    picked: list[int] = []
    for j in sorted(scores, key=lambda x: (-scores[x], x)):
        if all(abs(j - q) >= min_gap for q in picked) and \
           j >= min_gap and len(units) - j >= min_gap:
            picked.append(j)
            if len(picked) == k:
                break
    return tuple(sorted(picked))


def k_range(n_units: int, min_chunk: int = 2, max_k: int = 10) -> list[int]:
    """이 문장에서 재볼 k 들 — 평균 조각이 `min_chunk` 어절 아래로 가면 멈춘다.

    1어절 조각 구간은 어떤 분절이든 `H_set` 이 바닥이라 잡음만 는다.
    """
    kmax = min(max_k, n_units // min_chunk - 1)
    return list(range(1, kmax + 1))


def chunk_len(n_units: int, k: int) -> float:
    """k 개를 자를 때의 평균 조각 길이 — 보고용 지연축."""
    return n_units / (k + 1)


def pieces_of(units: list[str], cut: tuple[int, ...], spaced: bool) -> list[tuple[int, int]]:
    """절단집합 → 조각의 (시작, 끝) 인덱스. 절단이 없으면 문장 하나."""
    out, prev = [], 0
    for j in list(cut) + [len(units)]:
        out.append((prev, j))
        prev = j
    return out


@dataclass
class HsetScorer:
    """조각 번역과 QE 를 묶어 `H_set` 을 낸다.

    `translators` 는 타깃 이름 → `full(list[str]) -> list[str]` 을 가진 번역기,
    `qe` 는 `score(srcs, hyps) -> list[float]`. 둘 다 캐시·배치를 자기가 책임진다.
    """

    translators: dict
    qe: object
    spaced: bool
    target_spaced: dict          # 타깃 이름 → 조각 번역을 공백으로 이을지
    _tr_cache: dict = field(default_factory=dict)

    def translate_spans(self, texts: list[str], spans: set[tuple[int, int, int]]) -> None:
        """`(문장 index, 시작, 끝)` 구간을 타깃마다 번역해 둔다. 이미 있는 것은 건너뛴다."""
        todo = sorted(s for s in spans if (self.translators and s not in self._tr_cache.get(
            next(iter(self.translators)), {})))
        if not todo:
            return
        units = {i: units_of(texts[i], self.spaced) for i in {s[0] for s in todo}}
        srcs = [join_units(units[i][a:b], self.spaced) for i, a, b in todo]
        for tgt, tr in self.translators.items():
            got = tr.full(srcs)
            self._tr_cache.setdefault(tgt, {}).update(zip(todo, got))

    def score(self, texts: list[str], jobs: list[tuple[int, tuple[int, ...]]],
              contra_of) -> list[float]:
        """`jobs` = [(문장 index, 절단집합)] → 각각의 `H_set`.

        `contra_of(i, j)` 는 위치 j 의 소스 contra. 절단이 없으면 contra 항은 1 이다.
        """
        units = {i: units_of(texts[i], self.spaced) for i, _ in jobs}
        spans = {(i, a, b) for i, c in jobs for a, b in pieces_of(units[i], c, self.spaced)}
        self.translate_spans(texts, spans)
        srcs, hyps, owner = [], [], []
        for n, (i, c) in enumerate(jobs):
            for tgt in self.translators:
                join = " " if self.target_spaced[tgt] else ""
                parts = [self._tr_cache[tgt][(i, a, b)]
                         for a, b in pieces_of(units[i], c, self.spaced)]
                srcs.append(texts[i]); hyps.append(join.join(parts)); owner.append((n, tgt))
        got = self.qe.score(srcs, hyps)
        per: dict[int, list[float]] = {}
        for (n, _tgt), v in zip(owner, got):
            per.setdefault(n, []).append(v)
        out = []
        for n, (i, c) in enumerate(jobs):
            q = st.mean(per.get(n, [0.0]))
            worst = max((contra_of(i, j) for j in c), default=0.0)
            out.append(q * (1 - worst))
        return out


def paired_bootstrap(a: list[float], b: list[float], clusters: list | None = None,
                     iters: int = 2000, seed: int = 1) -> dict:
    """짝지어진 차이의 평균과 95% 신뢰구간.

    종전 관문은 `Δ > 1 se` 였는데, **같은 프롬프트를 다시 채점만 해도** Δ 가
    +0.023 ± 0.017 로 그 문턱을 넘었다 (run22 실측). CI 하한을 쓰면 그 잡음이 걸러진다.

    `clusters` 를 주면 **그 단위로 재추출한다** (기본은 짝 단위). 짝이 (문장, k) 일 때 같은
    문장의 k 들은 같은 점수 벡터에서 나오므로 독립이 아니다 — 짝 단위로 재추출하면 CI 가
    실제보다 좁아져 아무것도 아닌 개정을 통과시킨다. 문장 id 를 클러스터로 주면 그 문장의
    k 들이 통째로 뽑히거나 통째로 빠진다.
    """
    import random
    if len(a) != len(b):
        raise ValueError(f"짝이 안 맞는다: {len(a)} vs {len(b)}")
    d = [x - y for x, y in zip(a, b)]
    if len(d) < 2:
        return {"mean": 0.0, "lo": 0.0, "hi": 0.0, "n": len(d), "n_clusters": 0}
    rng = random.Random(seed)
    if clusters is None:
        groups = [[x] for x in d]
    else:
        by: dict = {}
        for c, x in zip(clusters, d):
            by.setdefault(c, []).append(x)
        groups = list(by.values())
    g = len(groups)
    means = []
    for _ in range(iters):
        picked = [groups[rng.randrange(g)] for _ in range(g)]
        flat = [x for grp in picked for x in grp]
        means.append(sum(flat) / len(flat))
    means.sort()
    return {"mean": round(st.mean(d), 5), "lo": round(means[int(.025 * iters)], 5),
            "hi": round(means[int(.975 * iters) - 1], 5), "n": len(d), "n_clusters": g}
