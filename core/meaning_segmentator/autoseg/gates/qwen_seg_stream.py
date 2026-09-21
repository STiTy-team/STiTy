"""텍스트 스트리밍 `<SEG>` 데모 — 단어를 하나씩 디코더에 이어 붙이며 경계 확률을 로그로 찍는다.

Qwen3-ASR 파인튜닝 모델의 디코더만 쓴다 (오디오 인코더는 안 부른다). 단어가 들어올 때마다
그 토큰만 KV cache 에 붙여 forward 하고, 다음 토큰이 `<SEG>` 일 확률을 읽는다. 앞만 보므로
`qwen_seg_prob.py` 의 `p0` 와 같은 값이다 (`--feed-seg` 를 끄면 수치까지 같다).

P(<SEG>) ≥ `--threshold` 이고 마지막 절단 뒤 `--min-words` 어절 이상이면 자른다. 기본 0.2 는
run27 test 기준 평균 조각 약 5어절 (0.3 이면 약 8어절).

    # 문장 하나
    PYTHONPATH=. .venv/bin/python -m core.meaning_segmentator.autoseg.gates.qwen_seg_stream \
        --text "At the time, nearly 100 residents were evacuated from the area."
    # 파일 전체를 이어진 한 스트림으로 (줄바꿈도 그냥 공백)
    ... --file some.txt --delay 0.2
    # 직접 타이핑 — 줄마다 그 단어들을 흘려 넣는다. 빈 줄이면 남은 조각을 내보낸다
    ... (인자 없이)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

from .qwen_seg_prob import DEFAULT_MODEL, load_thinker, prompt_ids


class StreamSegmenter:
    def __init__(self, tok, thinker, language: str, threshold: float, min_words: int,
                 feed_seg: bool, max_words: int):
        import torch
        self.torch, self.tok, self.m = torch, tok, thinker
        self.seg_id = tok.convert_tokens_to_ids("<SEG>")
        self.sp_id = tok.convert_tokens_to_ids("Ġ")
        self.prompt = prompt_ids(tok, language)
        self.threshold, self.min_words = threshold, min_words
        self.feed_seg, self.max_words = feed_seg, max_words
        self.reset([])

    def reset(self, carry: list[str]) -> None:
        """문맥을 프롬프트부터 다시 만든다. `carry` 는 아직 안 자른 어절 (문맥 길이 제한용)."""
        from transformers import DynamicCache
        self.cache = DynamicCache()
        self.n = 0                      # cache 에 들어간 토큰 수
        self.pending: list[str] = []    # 마지막 절단 뒤 어절
        self.first = True
        self._forward(self.prompt)
        for w in carry:
            self._push_word(w)

    def _forward(self, ids: list[int]):
        torch = self.torch
        x = torch.tensor([ids], device="cuda")
        att = torch.ones((1, self.n + len(ids)), dtype=torch.long, device="cuda")
        pos = torch.arange(self.n, self.n + len(ids), device="cuda")
        if self.n == 0:
            self.m.rope_deltas = None
        out = self.m(input_ids=x, attention_mask=att, past_key_values=self.cache,
                     cache_position=pos, use_cache=True)
        self.n += len(ids)
        return torch.log_softmax(out.logits[0, -1].float(), dim=-1)

    def _push_word(self, w: str):
        ids = self.tok(w if self.first else " " + w, add_special_tokens=False)["input_ids"]
        self.first = False
        self.pending.append(w)
        return self._forward(ids)

    def p_seg(self, logp) -> float:
        """공백 없는 `<SEG>` 와 `' ' → <SEG>` 두 경로의 합. 뒤 경로는 공백 토큰을 잠깐 넣었다 뺀다."""
        direct = float(logp[self.seg_id])
        lp_sp = float(logp[self.sp_id])
        mark = self.n
        lp2 = float(self._forward([self.sp_id])[self.seg_id])
        self.cache.crop(mark)
        self.n = mark
        return math.exp(direct) + math.exp(lp_sp + lp2)

    def feed(self, w: str) -> dict:
        """어절 하나를 넣고 `{word, p, cut, chunk, ms}` 를 돌려준다."""
        t0 = time.perf_counter()
        logp = self._push_word(w)
        p = self.p_seg(logp)
        cut = p >= self.threshold and len(self.pending) >= self.min_words
        chunk = None
        if cut:
            chunk = " ".join(self.pending)
            self.pending = []
            if self.feed_seg:
                self._forward([self.sp_id, self.seg_id])
            if self.n > 0 and self._words_in_context() > self.max_words:
                self.reset([])
        elif self._words_in_context() > self.max_words:
            self.reset(self.pending[:])
        return {"word": w, "p": p, "cut": cut, "chunk": chunk,
                "ms": (time.perf_counter() - t0) * 1000}

    def _words_in_context(self) -> int:
        return (self.n - len(self.prompt)) // 1.3        # 대략의 어절 수 (영어 토큰/어절 ≈ 1.3)

    def flush(self) -> str | None:
        chunk = " ".join(self.pending) if self.pending else None
        self.pending = []
        return chunk


def batch_stream_cuts(tok, thinker, texts: list[str], language: str, threshold: float,
                      min_words: int, batch_size: int = 64, log=None) -> list[list[int]]:
    """`StreamSegmenter(feed_seg=True)` 와 같은 절단을 여러 문장에 걸쳐 배치로 낸다.

    단어를 하나씩 흘리면 문장마다 어절 수만큼 forward 가 필요하다 (CoVoST2 15,530문장이면
    임계값 하나에 한 시간). 대신 문장별로 두 종류의 질의를 번갈아 배치에 싣는다.

      scan   문맥(지금까지의 어절과 넣은 <SEG>) + 남은 어절 전부를 한 번에 forward.
             causal 이라 각 경계의 값은 그 앞만 본 것이다 — 아직 안 자른 구간에서는 단어를
             하나씩 넣은 것과 같다. 경계마다 P(<SEG>) + P(' ') 를 상한으로 적어 둔다.
      check  상한이 θ 를 넘는 첫 경계에서 `' '` 를 붙여 P(<SEG>|' ') 를 읽는다.
             P = P(<SEG>) + P(' ')·P(<SEG>|' ') ≥ θ 면 자르고 ' <SEG>' 를 문맥에 넣은 뒤 다시
             scan, 아니면 같은 scan 의 다음 후보로 간다 (앞에서 안 잘랐으니 여전히 유효).
    """
    import torch
    seg_id = tok.convert_tokens_to_ids("<SEG>")
    sp_id = tok.convert_tokens_to_ids("Ġ")
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    P = prompt_ids(tok, language)
    st_ = []
    for t in texts:
        u = t.split()
        wt = [tok(w if i == 0 else " " + w, add_special_tokens=False)["input_ids"] for i, w in enumerate(u)]
        st_.append({"wt": wt, "n": len(u), "ctx": list(P), "cur": 0, "last": 0, "cuts": [],
                    "cands": None, "q": "scan"})

    def scan_seq(s):
        return s["ctx"] + [x for w in s["wt"][s["cur"]:] for x in w]

    def check_seq(s, j):
        return s["ctx"] + [x for w in s["wt"][s["cur"]:j] for x in w] + [sp_id]

    active = [i for i, s in enumerate(st_) if s["n"] > 1]
    t0, rounds = time.time(), 0
    while active:
        rounds += 1
        jobs = []
        for i in active:
            s = st_[i]
            jobs.append((i, scan_seq(s) if s["q"] == "scan" else check_seq(s, s["cands"][0][0])))
        jobs.sort(key=lambda x: len(x[1]))
        results = {}
        with torch.inference_mode():
            for b in range(0, len(jobs), batch_size):
                chunk = jobs[b:b + batch_size]
                L = max(len(x) for _, x in chunk)
                ids = torch.full((len(chunk), L), pad_id, dtype=torch.long)
                att = torch.zeros((len(chunk), L), dtype=torch.long)
                for r, (_i, x) in enumerate(chunk):
                    ids[r, :len(x)] = torch.tensor(x)
                    att[r, :len(x)] = 1
                thinker.rope_deltas = None
                logits = thinker(input_ids=ids.cuda(), attention_mask=att.cuda(), use_cache=False).logits
                for r, (i, x) in enumerate(chunk):
                    s = st_[i]
                    if s["q"] == "scan":
                        # 경계 j(cur < j < n) = 어절 j−1 의 마지막 토큰 자리
                        pos, ends = len(s["ctx"]) - 1, []
                        for w in s["wt"][s["cur"]:]:
                            pos += len(w)
                            ends.append(pos)
                        js = list(range(s["cur"] + 1, s["n"]))
                        lp = torch.log_softmax(logits[r, [ends[j - s["cur"] - 1] for j in js]].float(), -1)
                        results[i] = [(j, float(lp[k, seg_id].exp()), float(lp[k, sp_id].exp()))
                                      for k, j in enumerate(js)]
                    else:
                        lp = torch.log_softmax(logits[r, len(x) - 1].float(), -1)
                        results[i] = float(lp[seg_id].exp())
        nxt = []
        for i in active:
            s, res = st_[i], results[i]
            if s["q"] == "scan":
                s["cands"] = [c for c in res if c[0] - s["last"] >= min_words and c[1] + c[2] >= threshold]
            else:
                j, direct, psp = s["cands"][0]
                if direct + psp * res >= threshold:
                    s["ctx"] = s["ctx"] + [x for w in s["wt"][s["cur"]:j] for x in w] + [sp_id, seg_id]
                    s["cur"] = s["last"] = j
                    s["cuts"].append(j)
                    s["cands"] = None
                else:
                    s["cands"] = s["cands"][1:]
            s["q"] = "check" if s["cands"] else "scan"
            if s["cands"] == [] or (s["cands"] is None and s["cur"] >= s["n"] - 1):
                continue                     # 남은 후보 없음 = 이 문장은 끝
            nxt.append(i)
        active = nxt
        if log and rounds % 5 == 0:
            log(f"[batch θ={threshold}] 라운드 {rounds} / 남은 문장 {len(active)} / {time.time() - t0:.0f}s")
    return [s["cuts"] for s in st_]


def bar(p: float, width: int = 20) -> str:
    k = min(width, round(p * width))
    return "█" * k + "·" * (width - k)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--text", default=None)
    ap.add_argument("--file", default=None)
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--language", default="English")
    ap.add_argument("--threshold", type=float, default=0.2)
    ap.add_argument("--min-words", type=int, default=2, help="조각 최소 어절 수")
    ap.add_argument("--delay", type=float, default=0.0, help="어절 사이 대기(초) — 실시간처럼 보려면 0.2 정도")
    ap.add_argument("--feed-seg", action=argparse.BooleanOptionalAction, default=False,
                    help="자른 자리에 ' <SEG>' 를 문맥에 넣는다 (ASR 서버처럼). 끄면 오프라인 p0 와 같은 값")
    ap.add_argument("--max-words", type=int, default=120, help="문맥이 이 어절을 넘으면 다시 짠다")
    ap.add_argument("--jsonl", default=None, help="어절별 기록을 이 파일에 덧붙인다")
    a = ap.parse_args()

    print(f"[load] {Path(a.model).name}", file=sys.stderr, flush=True)
    tok, thinker = load_thinker(Path(a.model))
    seg = StreamSegmenter(tok, thinker, a.language, a.threshold, a.min_words, a.feed_seg, a.max_words)
    sink = open(a.jsonl, "a", encoding="utf-8") if a.jsonl else None
    print(f"[ready] threshold {a.threshold} / min_words {a.min_words} / feed_seg {a.feed_seg}",
          file=sys.stderr, flush=True)

    def emit_word(r: dict) -> None:
        mark = "  ◀ SEG" if r["cut"] else ""
        print(f"{r['ms']:6.1f}ms  {r['word']:<18} p={r['p']:.3f} {bar(r['p'])}{mark}", flush=True)
        if r["cut"]:
            print(f"          ┗━ 커밋: {r['chunk']}", flush=True)
        if sink:
            sink.write(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()},
                                  ensure_ascii=False) + "\n")
            sink.flush()

    def run_words(words):
        for w in words:
            emit_word(seg.feed(w))
            if a.delay:
                time.sleep(a.delay)

    def flush():
        c = seg.flush()
        if c:
            print(f"          ┗━ 커밋(끝): {c}", flush=True)

    if a.text or a.file:
        src = a.text if a.text else Path(a.file).read_text(encoding="utf-8")
        run_words(src.split())
        flush()
    else:
        print("[입력] 단어를 치고 Enter. 빈 줄 = 남은 조각 내보내기, Ctrl-D = 끝", file=sys.stderr, flush=True)
        for line in sys.stdin:
            if not line.strip():
                flush()
                continue
            run_words(line.split())
        flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
