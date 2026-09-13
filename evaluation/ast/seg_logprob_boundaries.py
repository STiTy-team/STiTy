#!/usr/bin/env python3
"""gold 문장 경계마다 `<SEG>` 로그확률이 얼마였는지 — `SEG_LOGPROB_DUMP` 기록을 읽는다.

    python evaluation/ast/seg_logprob_boundaries.py \\
        --dump full_*.jsonl --manifest evaluation/ast/manifests/acl6060_t268_en-de.jsonl \\
        [--out 결과.json]

무엇을 재나
-----------
서버는 청크마다 마지막 `<SEG>` 뒤를 다시 생성한다. 그 생성의 각 토큰 위치에서 `<SEG>` 가
top-K 에 들었다면 그 로그확률이 기록돼 있다(`qwen3_asr.py`, `SEG_LOGPROB_DUMP`).

단어 사이 틈 하나에 대해 "그 틈에서 `<SEG>` 를 냈을 확률"은 앞 단어의 마지막 글자 다음부터
뒤 단어의 첫 글자까지 걸친 생성 토큰 위치들의 `<SEG>` logprob 중 최댓값으로 둔다. 마침표
토큰도 그 틈에 들어간다(`set.` 의 `.` 자리에서 `<SEG>` 를 내면 `set <SEG>` 가 된다).

**버퍼 끝도 틈이다.** 모델은 `<SEG>` 를 생성의 맨 끝(뒤따르는 단어 없이 `<|im_end|>` 앞)에
내는 일이 많다. 그 청크에는 다음 문장의 첫 단어가 아직 없으므로, 마지막 단어 뒤의 토큰들
(`<SEG>`, `<|im_end|>` 자리)을 "마지막 단어와 그다음 gold 단어 사이 틈" 으로 셈한다. 이걸
빼면 휴지가 긴 경계(발화가 끊겨 버퍼 끝에서 `<SEG>` 를 내는 자리)가 통째로 누락된다.

`<SEG>` 를 **실제로 냈는지**는 logprob 이 아니라 텍스트로 판정한다. 앞 청크에서 낸 `<SEG>` 는
다음 청크의 prefix 로 들어가 logprob 이 없지만 텍스트에는 남는다.

가설 단어를 gold 단어에 정렬(difflib)해 틈마다 "gold 문장 경계인가, 문장 안인가" 를 붙이고,
같은 틈을 여러 청크가 다시 생성하므로 **청크 전체의 최댓값**을 그 틈의 값으로 쓴다.
`<SEG>` 가 top-K 밖이면 기록이 없다 — 그때는 그 위치 top-K 최저값보다 낮다는 것만 안다.

창 매니페스트(`offset` 이 0 이 아닌 행)는 스트림 시각에 offset 을 더해 발표 시각으로 옮긴다.
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

WORD = re.compile(r"[\w'’]+")
SEG = "<SEG>"


def norm(w: str) -> str:
    return w.lower().replace("’", "'")


def _gap(toks, spans, text, lo: int, hi: int):
    """글자 구간 [lo, hi) 에 걸친 생성 토큰들의 (SEG logp 최대, top-K 최저, 생성 토큰이 있었나)."""
    seg_max = floor_min = None
    touched = False
    for i, (ts, te) in enumerate(spans):
        if te <= lo or ts >= hi:
            continue
        touched = True
        tok, lp_ch, lp_seg, lp_floor = toks[i]
        if tok.strip() == SEG:
            lp_seg = lp_ch
        if lp_seg is not None:
            seg_max = lp_seg if seg_max is None else max(seg_max, lp_seg)
        if lp_floor is not None:
            floor_min = lp_floor if floor_min is None else min(floor_min, lp_floor)
    return seg_max, floor_min, touched


def chunk_gaps(rec: dict):
    """청크 하나 → (단어 목록, 끝 틈).

    단어 목록: [(단어, 앞 틈 SEG logp 최대, 앞 틈 top-K 최저, 앞 틈에 <SEG> 텍스트, 앞 틈이 생성 구간)]
    끝 틈   : 마지막 단어 뒤 (SEG logp 최대, top-K 최저, <SEG> 텍스트, 생성 구간)
    """
    prefix = rec["prefix"] or ""
    toks = rec["tok"]
    text = prefix
    spans = []
    for t in toks:
        spans.append((len(text), len(text) + len(t[0])))
        text += t[0]
    body = text.find("<asr_text>")
    body = body + len("<asr_text>") if body >= 0 else 0
    # 단어는 특수 토큰을 같은 길이 공백으로 가린 사본에서 찾는다. 안 가리면 `<|im_end|>` 의
    # `im_end` 와 `<SEG>` 의 `SEG` 가 단어로 잡혀, 버퍼 끝에서 낸 `<SEG>` 가 전부 "그 가짜 단어
    # 앞 틈" 으로 셈해지고 마지막 단어가 gold 에 정렬되지 않는다.
    masked = re.sub(r"<\|[^|>]*\|>|<SEG>|<asr_text>", lambda m: " " * len(m.group()), text)
    words = [(m.start(), m.end(), m.group()) for m in WORD.finditer(masked, body)]
    out = []
    prev_end = body
    for ws, we, w in words:
        seg_max, floor_min, touched = _gap(toks, spans, text, prev_end, ws + 1)
        out.append((w, seg_max, floor_min, SEG in text[prev_end:ws], touched))
        prev_end = we
    end = None
    if words:
        seg_max, floor_min, touched = _gap(toks, spans, text, prev_end, len(text))
        end = (seg_max, floor_min, SEG in text[prev_end:], touched)
    return out, end


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True, nargs="+")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--margin-sec", type=float, default=3.0,
                    help="청크가 덮는 오디오 앞뒤로 gold 단어를 이만큼 더 넣고 정렬한다")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    man = {}
    for line in open(a.manifest, encoding="utf-8"):
        if line.strip():
            e = json.loads(line)
            man[e["utt_id"]] = e

    # gold 단어열: (정규화 단어, 문장 id, 문장 안 순번, 문장 시작, 문장 끝, 앞 문장과의 휴지)
    gold = {}
    for uid, e in man.items():
        ws = []
        sents = e["sentences"]
        for k, s in enumerate(sents):
            gap = None
            if k > 0:
                p = sents[k - 1]
                gap = max(0.0, s["offset"] - (p["offset"] + p["duration"]))
            for j, w in enumerate(WORD.findall(s["src"])):
                ws.append((norm(w), s["seg_id"], j, s["offset"], s["offset"] + s["duration"], gap))
        gold[uid] = ws

    obs = defaultdict(lambda: {"seg": None, "floor": None, "chosen": False,
                               "chosen_at_end": False, "n_gen": 0, "n_end": 0})

    def put(key, seg_max, floor_min, chosen, touched, at_end):
        o = obs[key]
        if touched:
            o["n_gen"] += 1
            if at_end:
                o["n_end"] += 1
            if seg_max is not None:
                o["seg"] = seg_max if o["seg"] is None else max(o["seg"], seg_max)
            if floor_min is not None:
                o["floor"] = floor_min if o["floor"] is None else min(o["floor"], floor_min)
        if chosen:
            o["chosen"] = True
            o["chosen_at_end"] = o["chosen_at_end"] or at_end

    n_rec = 0
    for path in a.dump:
        for line in open(path, encoding="utf-8"):
            rec = json.loads(line)
            uid = rec.get("tag")
            if uid not in man or rec.get("now") is None:
                continue
            n_rec += 1
            off = float(man[uid].get("offset") or 0.0)
            t1 = off + float(rec["now"])
            t0 = t1 - float(rec["accum_sec"])
            G = gold[uid]
            gi = [i for i, g in enumerate(G) if g[4] >= t0 - a.margin_sec and g[3] <= t1 + a.margin_sec]
            if not gi:
                continue
            hyp, end = chunk_gaps(rec)
            if not hyp:
                continue
            sm = difflib.SequenceMatcher(None, [norm(h[0]) for h in hyp], [G[i][0] for i in gi],
                                         autojunk=False)
            last_aligned = None
            for blk in sm.get_matching_blocks():
                for d in range(blk.size):
                    h = hyp[blk.a + d]
                    put((uid, gi[blk.b + d]), h[1], h[2], h[3], h[4], False)
                    if blk.a + d == len(hyp) - 1:
                        last_aligned = gi[blk.b + d]
            # 끝 틈 — 마지막 가설 단어가 gold 에 정렬됐을 때만 다음 gold 단어 앞 틈으로 셈한다
            if end is not None and last_aligned is not None and last_aligned + 1 < len(G):
                put((uid, last_aligned + 1), end[0], end[1], end[2], end[3], True)

    rows = []
    for (uid, i), o in obs.items():
        g = gold[uid][i]
        rows.append({"utt": uid, "gold_idx": i, "word": g[0],
                     "boundary": g[2] == 0 and g[5] is not None,
                     "sent_start": round(g[3], 2), "gap_sec": None if g[5] is None else round(g[5], 3),
                     "seg_lp": o["seg"], "floor": o["floor"], "chosen": o["chosen"],
                     "chosen_at_end": o["chosen_at_end"], "n_gen": o["n_gen"], "n_end": o["n_end"]})

    def bucket(r):
        if r["chosen"]:
            return "채택"
        v = r["seg_lp"]
        if v is None:
            return "top-K 밖"
        for lo, name in ((-1, "≥−1"), (-3, "−3~−1"), (-6, "−6~−3")):
            if v >= lo:
                return name
        return "< −6"

    names = ["채택", "≥−1", "−3~−1", "−6~−3", "< −6", "top-K 밖"]
    print(f"청크 기록 {n_rec}줄, 관측된 틈 {len(rows)}개")
    for label, sel in (("gold 문장 경계", [r for r in rows if r["boundary"]]),
                       ("문장 안 단어 틈", [r for r in rows if not r["boundary"]])):
        c = defaultdict(int)
        for r in sel:
            c[bucket(r)] += 1
        tot = max(1, len(sel))
        at_end = sum(1 for r in sel if r["chosen_at_end"])
        print(f"  {label:12s} n={len(sel):5d} | " +
              " ".join(f"{k} {c[k] / tot * 100:5.1f}%" for k in names) +
              f" | 채택 중 버퍼 끝 {at_end}")
    bd = [r for r in rows if r["boundary"]]
    for lo, hi, name in ((0, 0.1, "0~0.1"), (0.1, 0.3, "0.1~0.3"), (0.3, 0.6, "0.3~0.6"), (0.6, 99, "0.6~")):
        sel = [r for r in bd if lo <= r["gap_sec"] < hi]
        if not sel:
            continue
        c = defaultdict(int)
        for r in sel:
            c[bucket(r)] += 1
        print(f"  경계 휴지 {name:7s}초 n={len(sel):3d} | " + " ".join(f"{k} {c[k]}" for k in names) +
              f" | 채택 중 버퍼 끝 {sum(1 for r in sel if r['chosen_at_end'])}")

    if a.out:
        Path(a.out).write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"저장: {a.out}")


if __name__ == "__main__":
    main()
