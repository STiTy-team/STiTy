"""Northstar run scorers that read a finished bench run directory (items.jsonl / events.jsonl), no re-decoding.

counts      dual-stream hallucinations, primary-stream english emissions, special-token leaks. The english-emission
            rule strips e-mail addresses and URLs before counting letters - measured on the guard run, both remaining
            "emissions" were e-mail addresses read out correctly ("Emma.Witfield@gmail.com이에요."). --show prints the
            emission events instead of only counting.
dual-wer    per-utterance WER/CER of a dual-decode run. The dual run decodes two headers per step: the run target
            (pipeline stream, `translated` records) and the other language (event log only, `dual` records). An
            utterance's transcription is whichever stream matches its gold language, so WER/CER become measurable.
hangul-cer  hangul-only CER of the ko passthrough - the transcription-content yardstick. Strips everything but hangul
            characters before scoring, so Korean spacing conventions, language-label noise and latin/digit variants
            drop out. A cascade run (--cascade) is scored on ALL its transcribed events, which is generous to it
            (code-switched segments included).

dual-wer and hangul-cer bin emissions to reference sentences by decision time (decision_audio_sec - lag landing inside
the sentence, open-ended until the next sentence starts).

usage: python3 core/research/zprobe/eval/ns_score.py counts [--runs-root R] [--show] <cfg> [<cfg> ...]
       (bench venv) python core/research/zprobe/eval/ns_score.py dual-wer --run-dir <R>/adapter.prefix4-segreset-dual --target ko
       (bench venv) python core/research/zprobe/eval/ns_score.py hangul-cer --cascade asr.qwen-seg-koen+mt.qwen3.5-4b <cfg> [<cfg> ...]

The secondary-stream COMET (ns_secondary_score.py) stays separate: it imports bench and runs from STiTy-bench."""

import argparse, json, os, re
from collections import defaultdict

RUNS_ROOT = "/home/mobility/STiTy-bench/bench/runs/northstar_ko+en-ko@v2"
HALLU = re.compile(
    r"The (two men|1990s|1980s|government|first|woman)|arrested|criticised|How are you|Good morning"
)
EMAIL_URL = re.compile(
    r"[\w.+-]+@[\w-]+\.[\w.-]+|https?://\S+|www\.\S+|\b[\w-]+\.(?:com|io|net|org|co\.kr|kr)\b"
)


def items(run_dir):
    for l in open(os.path.join(run_dir, "items.jsonl"), encoding="utf-8"):
        yield json.loads(l)


def events(run_dir):
    for l in open(os.path.join(run_dir, "events.jsonl"), encoding="utf-8"):
        yield json.loads(l)


def sentences(item):
    """Full reference sentences in time order, each with the [lo, hi) window its emissions are binned into."""
    sents = sorted(
        (s for s in item["reference_segmentation"] if not s.get("partial")),
        key=lambda s: s["offset"],
    )
    for i, s in enumerate(sents):
        yield s, s["offset"], sents[i + 1]["offset"] if i + 1 < len(sents) else float("inf")


def binned(records, lo, hi, lag):
    return [
        r["original"]
        for r in records
        if lo <= r["decision_audio_sec"] - lag < hi and r.get("original")
    ]


def counts(a):
    for name in a.cfgs:
        run_dir = os.path.join(a.runs_root, name)
        if not os.path.exists(os.path.join(run_dir, "events.jsonl")):
            print(f"{name}: no events.jsonl")
            continue
        h = w = f = 0
        for d in events(run_dir):
            if d.get("type") == "dual" and HALLU.search(d.get("original", "")):
                h += 1
            if d.get("type") == "translated":
                t = d.get("translation", "")
                if "<|" in t:
                    f += 1
                letters = [c for c in EMAIL_URL.sub("", t) if c.isalpha()]
                if len(letters) >= 4 and sum(c.isascii() for c in letters) / len(letters) > 0.7:
                    w += 1
                    if a.show:
                        print(f"  {d.get('item')} @{d.get('audio')}: {t[:120]}")
        print(f"{name}: dual 환각류 {h}, 주(ko)스트림 영어방출 {w}, 특수토큰 누출 {f}")


def dual_wer(a):
    import jiwer
    from whisper_normalizer.basic import BasicTextNormalizer
    from whisper_normalizer.english import EnglishTextNormalizer

    norm = {"en": EnglishTextNormalizer()}
    basic = BasicTextNormalizer(preserve_marks=True)
    dual = defaultdict(list)
    for d in events(a.run_dir):
        if d.get("type") == "dual":
            dual[d["item"]].append(d)
    pairs = defaultdict(list)
    for item in items(a.run_dir):
        if item.get("status") != "ok":
            continue
        prim = [r for r in item.get("records", []) if r["type"] == "translated"]
        for s, lo, hi in sentences(item):
            pool = prim if s["src_lang"] == a.target else dual.get(item["id"], [])
            pairs[s["src_lang"]].append((s["transcript"], " ".join(binned(pool, lo, hi, a.lag))))

    print("| lang | n | WER | CER | stream |")
    print("|---|---|---|---|---|")
    for lang in sorted(pairs):
        n = norm.get(lang, basic)
        refs, hyps = zip(*[(n(r).strip(), n(h).strip()) for r, h in pairs[lang] if n(r).strip()])
        wer = jiwer.wer(list(refs), list(hyps))
        cer = jiwer.cer(["".join(r.split()) for r in refs], ["".join(h.split()) for h in hyps])
        which = "primary (passthrough)" if lang == a.target else "dual"
        print(f"| {lang} | {len(refs)} | {wer:.3f} | {cer:.3f} | {which} |")


def hangul_cer(a):
    import jiwer

    hangul = lambda t: "".join(re.findall(r"[가-힣]", t))

    def adapter_hyp(run_dir):
        out = {}
        for it in items(run_dir):
            prim = [r for r in it.get("records", []) if r["type"] == "translated"]
            out[it["id"]] = " ".join(
                x
                for s, lo, hi in sentences(it)
                if s["src_lang"] == "ko"
                for x in binned(prim, lo, hi, a.lag)
            )
        return out

    def cascade_hyp(run_dir):
        out = defaultdict(list)
        for d in events(run_dir):
            if d.get("type") == "transcribed":
                out[d["item"]].append(d.get("original", ""))
        return {k: " ".join(v) for k, v in out.items()}

    first = os.path.join(a.runs_root, a.cascade or a.cfgs[0])
    refs = {
        it["id"]: " ".join(s["transcript"] for s, _, _ in sentences(it) if s["src_lang"] == "ko")
        for it in items(first)
    }
    print("| run | hangul CER |")
    print("|---|---|")
    for run, get in ([(a.cascade, cascade_hyp)] if a.cascade else []) + [
        (c, adapter_hyp) for c in a.cfgs
    ]:
        try:
            h = get(os.path.join(a.runs_root, run))
        except FileNotFoundError:
            print(f"| {run} | no run |")
            continue
        R = [hangul(refs[k]) for k in sorted(refs)]
        H = [hangul(h.get(k, "")) for k in sorted(refs)]
        print(f"| {run} | {jiwer.cer(R, H):.3f} |")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("counts")
    p.add_argument("cfgs", nargs="+")
    p.add_argument("--runs-root", default=RUNS_ROOT)
    p.add_argument("--show", action="store_true")
    p = sub.add_parser("dual-wer")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--target", required=True)
    p.add_argument(
        "--lag",
        type=float,
        default=0.5,
        help="decode lag subtracted from emission time before binning",
    )
    p = sub.add_parser("hangul-cer")
    p.add_argument("cfgs", nargs="*")
    p.add_argument("--runs-root", default=RUNS_ROOT)
    p.add_argument("--cascade")
    p.add_argument("--lag", type=float, default=0.5)
    a = ap.parse_args()
    {"counts": counts, "dual-wer": dual_wer, "hangul-cer": hangul_cer}[a.mode](a)
