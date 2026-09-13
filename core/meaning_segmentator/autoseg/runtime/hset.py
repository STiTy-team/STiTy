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


def paired_bootstrap(a: list[float], b: list[float], iters: int = 2000,
                     seed: int = 1) -> dict:
    """짝지어진 차이의 평균과 95% 신뢰구간. 짝은 (문장, T) 단위다.

    종전 관문은 `Δ > 1 se` 였는데, **같은 프롬프트를 다시 채점만 해도** Δ 가
    +0.023 ± 0.017 로 그 문턱을 넘었다 (run22 실측). CI 하한을 쓰면 그 잡음이 걸러진다.
    """
    import random
    if len(a) != len(b):
        raise ValueError(f"짝이 안 맞는다: {len(a)} vs {len(b)}")
    d = [x - y for x, y in zip(a, b)]
    if len(d) < 2:
        return {"mean": 0.0, "lo": 0.0, "hi": 0.0, "n": len(d)}
    rng = random.Random(seed)
    n = len(d)
    means = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(iters))
    return {"mean": round(st.mean(d), 5), "lo": round(means[int(.025 * iters)], 5),
            "hi": round(means[int(.975 * iters) - 1], 5), "n": n}
