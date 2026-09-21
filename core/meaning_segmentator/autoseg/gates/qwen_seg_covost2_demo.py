"""CoVoST2 문장을 단어 단위로 흘려 `<SEG>` 로 자르고, 자른 조각을 바로 번역하는 데모 로그.

분절은 `qwen_seg_stream.StreamSegmenter` (Qwen3-ASR 디코더, 텍스트만), 번역은 로컬 madlad
(CoVoST2 실험과 같은 모델, 비용 0). 문장마다 컨텍스트를 새로 짜고, 문장 끝에서 남은 조각을
내보낸다 (발화 끝 = VAD 커밋에 해당).

마지막에 코퍼스 BLEU 를 셋 비교한다: 조각 번역을 이어붙인 것 / 문장 통째 번역 / 참조.

    PYTHONPATH=. .venv/bin/python -m core.meaning_segmentator.autoseg.gates.qwen_seg_covost2_demo \
        --n 30 --out <로그 경로>
"""
from __future__ import annotations

import argparse
import json
import random
import statistics as st
import time
from pathlib import Path

from .qwen_seg_prob import DEFAULT_MODEL, REPO, load_thinker
from .qwen_seg_stream import StreamSegmenter, bar

MANIFEST = REPO / ("core/meaning_segmentator/experiment/artifacts/en2x/covost2/full/labels/"
                   "covost2_full_judge13_iter3.jsonl")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=str(MANIFEST), help="src_text / tgt_text 가 있는 jsonl")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--min-len", type=int, default=8, help="이 어절 수 미만 문장은 뽑지 않는다 (짧으면 자를 자리가 없다)")
    ap.add_argument("--seed", type=int, default=20260919)
    ap.add_argument("--tgt", default="de")
    ap.add_argument("--threshold", type=float, default=0.2)
    ap.add_argument("--min-words", type=int, default=2)
    ap.add_argument("--feed-seg", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--out", required=True, help="사람이 읽는 로그")
    a = ap.parse_args()

    import sacrebleu
    from ..runtime.pipeline import LocalTranslator

    rows = [json.loads(l) for l in open(a.manifest, encoding="utf-8")]
    pool = [r for r in rows if len(r["src_text"].split()) >= a.min_len and r["tgt_lang"] == a.tgt]
    picked = random.Random(a.seed).sample(pool, a.n)

    out = open(a.out, "w", encoding="utf-8")

    def say(s: str = "") -> None:
        print(s, flush=True)
        out.write(s + "\n")
        out.flush()

    tok, thinker = load_thinker(Path(a.model))
    seg = StreamSegmenter(tok, thinker, "English", a.threshold, a.min_words, a.feed_seg, 500)
    tr = LocalTranslator(tgt_code=a.tgt, cache=None)
    tr.full(["Warm-up."])                         # 모델 적재를 첫 조각 시간에 넣지 않는다
    say(f"# CoVoST2 en→{a.tgt} 텍스트 스트리밍 분절 + 조각 번역")
    say(f"# {Path(a.manifest).name} 에서 {a.min_len}어절 이상 {len(pool)}문장 중 {a.n}개 (seed {a.seed})")
    say(f"# 분절 {Path(a.model).name} 디코더만 / threshold {a.threshold} / min_words {a.min_words} "
        f"/ feed_seg {a.feed_seg} / 번역 {tr.model_id}")

    hyp_chunk, hyp_full, refs = [], [], []
    word_ms, mt_ms, n_chunks, chunk_len, first_at = [], [], [], [], []
    for k, r in enumerate(picked, 1):
        words = r["src_text"].split()
        say()
        say(f"━━ [{k}/{a.n}] {r['utt_id']} ({len(words)}어절)")
        say(f"   원문 : {r['src_text']}")
        say(f"   참조 : {r['tgt_text']}")
        seg.reset([])
        chunks = []

        def commit(chunk: str, at: int, end: bool) -> None:
            t0 = time.perf_counter()
            t = tr.full([chunk])[0]
            ms = (time.perf_counter() - t0) * 1000
            mt_ms.append(ms)
            chunks.append(t)
            chunk_len.append(len(chunk.split()))
            if len(chunks) == 1:
                first_at.append(at / len(words))
            say(f"          ┗━ 커밋 #{len(chunks)}{'(끝)' if end else ''} @{at}/{len(words)}어절  "
                f"\"{chunk}\"")
            say(f"             → {t}   (번역 {ms:.0f}ms)")

        for i, w in enumerate(words, 1):
            res = seg.feed(w)
            word_ms.append(res["ms"])
            say(f"   {res['ms']:5.1f}ms  {w:<18} p={res['p']:.3f} {bar(res['p'], 16)}"
                + ("  ◀ SEG" if res["cut"] else ""))
            if res["cut"]:
                commit(res["chunk"], i, False)
        rest = seg.flush()
        if rest:
            commit(rest, len(words), True)
        full = tr.full([r["src_text"]])[0]
        joined = " ".join(chunks)
        hyp_chunk.append(joined); hyp_full.append(full); refs.append(r["tgt_text"])
        n_chunks.append(len(chunks))
        b_c = sacrebleu.sentence_bleu(joined, [r["tgt_text"]]).score
        b_f = sacrebleu.sentence_bleu(full, [r["tgt_text"]]).score
        say(f"   조각 번역 이어붙임 : {joined}   (문장 BLEU {b_c:.1f})")
        say(f"   문장 통째 번역     : {full}   (문장 BLEU {b_f:.1f})")

    tok_arg = "ja-mecab" if a.tgt == "ja" else ("zh" if a.tgt == "zh" else "13a")
    bc = sacrebleu.corpus_bleu(hyp_chunk, [refs], tokenize=tok_arg).score
    bf = sacrebleu.corpus_bleu(hyp_full, [refs], tokenize=tok_arg).score
    say()
    say("━━ 요약")
    say(f"   문장 {a.n} / 조각 {sum(n_chunks)} (문장당 {st.mean(n_chunks):.2f}, 평균 {st.mean(chunk_len):.1f}어절)")
    say(f"   첫 조각이 나온 시점 : 문장의 평균 {st.mean(first_at) * 100:.0f}% 지점 (통째 번역이면 100%)")
    say(f"   분절 {st.mean(word_ms):.1f}ms/어절 (중앙 {st.median(word_ms):.1f}) / 번역 {st.mean(mt_ms):.0f}ms/조각")
    say(f"   코퍼스 BLEU  조각 이어붙임 {bc:.2f}  /  문장 통째 {bf:.2f}  (차 {bc - bf:+.2f})")
    out.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
