"""2.1 입력 준비: Gehry 강연 문장을 Qwen3-ASR <SEG> 디코더로 조각낸다 → data/talk231_seg.jsonl

오디오 없이 텍스트만 디코더에 넣고, 단어 경계마다 P(<SEG>) 를 읽어 θ 이상이면 자른다(스트리밍, 자른 자리에
' <SEG>' 를 문맥에 넣는다 — ASR 서버와 같은 조건). 모델은 models/Qwen3-ASR-1.7B-en-covost2-dailytalk-mix-c200-merged.

분절 코드는 이 워크트리에 없고 autoseg-judge 브랜치(d5d6a297)의 gates/qwen_seg_stream.py 에 있다.
그 트리를 풀어 둔 곳을 --autoseg-root 로 준다(그 아래에 Qwen3-ASR/, models/ 가 있어야 한다):

    git archive autoseg-judge core/meaning_segmentator/autoseg | tar -x -C <root>
    touch <root>/core/__init__.py <root>/core/meaning_segmentator/__init__.py
    ln -s <repo>/Qwen3-ASR <root>/Qwen3-ASR; ln -s <repo>/models <root>/models
    $METRICS_PY -u evaluation/TranslatorPrompt/scripts/build_seg.py --autoseg-root <root>   # transformers 4.57 환경

한 줄: {"idx", "en", "ko", "theta", "cuts"(조각이 끝나는 어절 번호), "pieces"}
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
TALK = HERE.parent / "LongContextMT" / "data" / "talk231.jsonl"
MODEL = "models/Qwen3-ASR-1.7B-en-covost2-dailytalk-mix-c200-merged"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--autoseg-root", required=True)
    ap.add_argument("--thetas", type=float, nargs="+", default=[0.2, 0.3])
    ap.add_argument("--min-words", type=int, default=2)
    args = ap.parse_args()

    root = Path(args.autoseg_root).resolve()
    sys.path.insert(0, str(root))
    from core.meaning_segmentator.autoseg.gates.qwen_seg_prob import load_thinker
    from core.meaning_segmentator.autoseg.gates.qwen_seg_stream import batch_stream_cuts

    talk = [json.loads(line) for line in TALK.read_text(encoding="utf-8").splitlines() if line.strip()]
    tok, thinker = load_thinker(root / MODEL)
    for theta in args.thetas:
        cuts = batch_stream_cuts(tok, thinker, [r["en"] for r in talk], "English", theta,
                                 args.min_words, log=lambda m: print(m, flush=True))
        out = HERE / "data" / f"talk231_seg_t{int(theta * 100):02d}.jsonl"
        n_pieces = []
        with open(out, "w", encoding="utf-8") as f:
            for r, c in zip(talk, cuts):
                words = r["en"].split()
                ends = sorted(set(c) | {len(words)})
                pieces, start = [], 0
                for e in ends:
                    if e > start:
                        pieces.append(" ".join(words[start:e]))
                        start = e
                n_pieces.append(len(pieces))
                f.write(json.dumps({"idx": r["idx"], "en": r["en"], "ko": r["ko"], "theta": theta,
                                    "cuts": c, "pieces": pieces}, ensure_ascii=False) + "\n")
        words_total = sum(len(r["en"].split()) for r in talk)
        print(f"θ={theta}: {out.name} — {sum(n_pieces)} pieces, {words_total / sum(n_pieces):.1f} words/piece, "
              f"{sum(1 for n in n_pieces if n == 1)} unsplit sentences", flush=True)


if __name__ == "__main__":
    main()
