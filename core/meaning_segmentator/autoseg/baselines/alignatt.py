"""AlignAtt (Papi et al., Interspeech 2023) → 소스 경계.

원 정책: 지금 내보내려는 타깃 토큰의 **교차어텐션 argmax 가 최근 `f` 프레임 안에 있으면
아직 정보가 부족한 것이므로 내보내지 않고 더 읽는다.** 노브는 `f` 하나다.

텍스트 적응 두 가지를 명시한다:
  1. 프레임 대신 **소스 어절**을 센다 (오디오가 없다).
  2. 원 논문은 방출 스케줄만 정하지 소스 분절을 내놓지 않는다. `causal_align` 과 같은
     방식으로, 접두사 길이 `t` 에서 새 타깃 토큰이 하나라도 나가면 거기에 경계를 찍는다.

마지막 층은 `</s>` 로 쏠려(attention sink 실측 확인) 쓸 수 없다. `attn_layer` 기본 6 은
NLLB-600M 에서 정렬 단조성이 가장 높았던 층이다(0.92) — 원 논문도 층을 고른다.
"""

from __future__ import annotations

import re

_WS = re.compile(r"\s+")


def segment(nmt, text: str, f: int = 2) -> list[str]:
    words = _WS.split(text.strip())
    n = len(words)
    if n < 2:
        return [text.strip()]

    pieces: list[str] = []
    forced: list[int] = []
    k = 0
    for t in range(1, n):                     # t = 지금까지 읽은 어절 수
        emitted = nmt.emit_with_alignment(" ".join(words[:t]), forced or None)
        commit: list[int] = []
        for tok_id, w in emitted:
            if w >= t - f:                    # 최근 f 어절에 붙어 있다 → 아직 못 낸다
                break
            commit.append(tok_id)
        if commit:
            forced = forced + commit
            pieces.append(" ".join(words[k:t]))
            k = t
    if k < n:                                 # 소스를 다 읽은 뒤 남은 꼬리
        pieces.append(" ".join(words[k:]))
    return pieces or [text.strip()]


class _St:
    """문장 하나의 진행 상태. t = 지금까지 읽은 어절 수, k = 마지막으로 끊은 자리."""
    __slots__ = ("i", "text", "words", "n", "t", "k", "forced", "pieces")

    def __init__(self, i: int, text: str, words: list[str]):
        self.i, self.text, self.words, self.n = i, text, words, len(words)
        self.t, self.k, self.forced, self.pieces = 1, 0, [], []


def segment_batch(nmt, texts: list[str], f: int = 2, pool: int = 512,
                  max_batch: int = 32, on_done=None) -> list[list[str]]:
    """여러 문장을 한 파장으로 밀어 `segment` 와 같은 결과를 배치로 낸다.

    한 문장 안에서 `t` 는 직전 회차가 정한 `forced` 에 의존하므로 순차다. 하지만 **문장끼리는
    독립**이라 여러 문장의 같은 회차를 한 배치로 묶을 수 있다. 회차마다 활성 문장 전부가
    요청 하나씩을 내고, `emit_with_alignment_batch` 가 그것을 `forced` 길이별로 묶어 돌린다.

    `pool` 은 동시에 진행하는 문장 수다. 이게 커야 `forced` 길이별 묶음이 커진다 — 회차
    하나에 길이가 20가지쯤 섞이므로, 배치를 `max_batch` 까지 채우려면 pool 이 그 몇 배여야
    한다. 상태는 문장당 리스트 몇 개라 pool 을 키워도 메모리는 거의 안 든다.

    `on_done(i, pieces)` 를 주면 문장이 끝나는 즉시 부른다 (진행분 기록용). 끝나는 순서는
    입력 순서가 아니다 — 짧은 문장이 먼저 끝난다. 반환 리스트는 입력 순서다.
    """
    out: list[list[str] | None] = [None] * len(texts)
    queue: list[_St] = []
    for i, text in enumerate(texts):
        words = _WS.split(text.strip())
        if len(words) < 2:
            out[i] = [text.strip()]
            if on_done:
                on_done(i, out[i])
        else:
            queue.append(_St(i, text, words))
    queue.reverse()                        # pop() 이 앞에서부터 꺼내도록

    active: list[_St] = []
    while queue or active:
        while queue and len(active) < pool:
            active.append(queue.pop())
        items = [(" ".join(s.words[:s.t]), s.forced or None) for s in active]
        res = nmt.emit_with_alignment_batch(items, max_batch=max_batch)
        nxt: list[_St] = []
        for s, emitted in zip(active, res):
            commit: list[int] = []
            for tok_id, w in emitted:
                if w >= s.t - f:           # 최근 f 어절에 붙어 있다 → 아직 못 낸다
                    break
                commit.append(tok_id)
            if commit:
                s.forced = s.forced + commit
                s.pieces.append(" ".join(s.words[s.k:s.t]))
                s.k = s.t
            s.t += 1
            if s.t < s.n:
                nxt.append(s)
                continue
            if s.k < s.n:                  # 소스를 다 읽은 뒤 남은 꼬리
                s.pieces.append(" ".join(s.words[s.k:]))
            out[s.i] = s.pieces or [s.text.strip()]
            if on_done:
                on_done(s.i, out[s.i])
        active = nxt
    return out
