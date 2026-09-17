"""Zhang et al. 2020 (EMNLP) Algorithm 1 — 접두사 매칭 Meaning Unit.

    prefix x≤t 의 번역 M(x≤t) 가 전체 문장 번역 ỹ 의 **접두사이면** t 를 MU 경계로 삼는다.
    확정된 MU 의 번역은 고정이므로 다음 판정은 그것을 tgt_force 로 깔고 이어 디코딩한다.
    엄격한 접두사 조건은 논문대로 ỹ 를 beam top-N(=10) 후보 집합으로 넓혀 완화한다.

원논문은 이 산출을 BERT 분류기 **학습 데이터**로 쓰지만, Table 1a 는 오프라인
"라벨 출처" 비교이므로 Algorithm 1 자체를 라벨러로 쓴다 — 분류기 학습이 필요 없다.

**구현 범위: basic method 만.** 논문의 MU++(refined)는 prefix-attention 으로 단조 NMT 를
수백만 문장에 파인튜닝해야 해서 범위 밖이다. 논문 스스로 basic 은 재배열이 심한 쌍에서
MU 가 문장 전체 하나로 붕괴한다고 밝힌다(Fig. 2) — en→de(동사후치)·en→ja(SOV)가 정확히
그 경우다. 붕괴하면 그 자체가 결과다. 표에 올릴 때 "basic, top-10 완화" 를 명시할 것.
"""

from __future__ import annotations

import re

_WS = re.compile(r"\s+")


def _norm(s: str) -> str:
    return _WS.sub(" ", s).strip()


def is_prefix_of_any(hyp: str, cands: list[str], spaced: bool) -> bool:
    h = _norm(hyp)
    if not h:
        return False
    for c in cands:
        c = _norm(c)
        if not c.startswith(h):
            continue
        # 띄어쓰기 언어는 어절 중간에서 끊긴 매칭을 배제한다 ("I go" vs "I going").
        if spaced and len(c) > len(h) and not c[len(h)].isspace():
            continue
        return True
    return False


def segment(nmt, text: str, tgt_spaced: bool, n_cands: int = 10,
            trace: list | None = None) -> list[str]:
    toks = _WS.split(text.strip())
    if len(toks) < 2:
        return [text.strip()]
    cands = nmt.full_candidates(text, n=n_cands)

    pieces: list[str] = []
    forced: list[int] = []
    k = 0
    for t in range(1, len(toks) + 1):
        prefix_src = " ".join(toks[:t])
        hyp, ids = nmt.translate_prefix(prefix_src, forced=forced or None)
        if trace is not None:
            trace.append({"t": t, "hyp": hyp})
        if t < len(toks) and is_prefix_of_any(hyp, cands, tgt_spaced):
            pieces.append(" ".join(toks[k:t]))
            k = t
            forced = ids
    if k < len(toks):                       # 남은 꼬리는 마지막 MU
        pieces.append(" ".join(toks[k:]))
    return pieces or [text.strip()]


class _St:
    """문장 하나의 진행 상태. t = 지금까지 읽은 어절 수, k = 마지막으로 끊은 자리."""
    __slots__ = ("i", "text", "toks", "n", "t", "k", "forced", "pieces", "cands")

    def __init__(self, i: int, text: str, toks: list[str], cands: list[str]):
        self.i, self.text, self.toks, self.n = i, text, toks, len(toks)
        self.cands = cands
        self.t, self.k, self.forced, self.pieces = 1, 0, [], []


def segment_batch(nmt, texts: list[str], tgt_spaced: bool, n_cands: int = 10,
                  pool: int = 512, max_batch: int = 64, max_beams: int = 128,
                  on_done=None) -> list[list[str]]:
    """여러 문장을 한 파장으로 밀어 `segment` 와 같은 결과를 배치로 낸다.

    구조는 `alignatt.segment_batch` 와 같다 — 문장 안에서는 `forced` 때문에 순차지만
    문장끼리는 독립이므로 같은 회차를 묶는다. 후보 집합 `cands` 는 `t` 와 무관하므로
    풀 단위로 미리 배치 계산한다. 빔이 배치 안에서 곱해져 `n_cands` 가 클수록 그 배치가
    작아지므로 `max_beams` 로 따로 잡는다.

    `t = len(toks)` 회차는 돌지 않는다. 원래 구현도 그 회차의 번역 결과를 쓰지 않는다
    (`t < len(toks)` 조건에서 걸러진다) — 결과는 같고 호출만 1/n 줄어든다.
    """
    out: list[list[str] | None] = [None] * len(texts)
    todo: list[tuple[int, str, list[str]]] = []
    for i, text in enumerate(texts):
        toks = _WS.split(text.strip())
        if len(toks) < 2:
            out[i] = [text.strip()]
            if on_done:
                on_done(i, out[i])
        else:
            todo.append((i, text, toks))

    for s in range(0, len(todo), pool):
        chunk = todo[s:s + pool]
        cands = nmt.full_candidates_batch([c[1] for c in chunk], n=n_cands,
                                          max_beams=max_beams)
        active = [_St(i, text, toks, cd) for (i, text, toks), cd in zip(chunk, cands)]
        while active:
            items = [(" ".join(st.toks[:st.t]), st.forced or None) for st in active]
            res = nmt.translate_prefix_batch(items, max_batch=max_batch)
            nxt: list[_St] = []
            for st, (hyp, ids) in zip(active, res):
                if is_prefix_of_any(hyp, st.cands, tgt_spaced):
                    st.pieces.append(" ".join(st.toks[st.k:st.t]))
                    st.k = st.t
                    st.forced = ids
                st.t += 1
                if st.t < st.n:
                    nxt.append(st)
                    continue
                if st.k < st.n:                 # 남은 꼬리는 마지막 MU
                    st.pieces.append(" ".join(st.toks[st.k:]))
                out[st.i] = st.pieces or [st.text.strip()]
                if on_done:
                    on_done(st.i, out[st.i])
            active = nxt
    return out
