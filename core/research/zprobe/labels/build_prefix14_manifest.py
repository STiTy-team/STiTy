"""Build the prefix14 sentence manifest: one line per unique source sentence.

Inputs (all optional except --aligned):
  --aligned          prefix13 aligned.jsonl (FLEURS + CoVoST recordings; only en->ko and ko->en kept)
  --fleurs-root      FLEURS data dir with {en_us,ko_kr}/train.tsv (col0 = sentence id shared across
                     languages, col2 = raw text); used to look up the reference translation
  --aihub-manifest / --aihub-en-manifest
                     AI Hub 71693 Korean / English talk manifests (one sentence per line); whole talks are
                     picked round-robin over the four domains (EA/MA/NA/SA), biggest talks first
  --kosp2e-manifest  pre-joined kosp2e manifest (ko transcript + en translation per line)

Output (--out): sentences.jsonl, one line per unique (lang, whitespace-collapsed src):
  {"sid","lang","tgt","src","ref","source","recs":[{"audio": path or path#start=..#end=..}, ...]}
  sid = f"{source}_{lang}_{sha1(src)[:8]}"; when several sources share a sentence the first source's
  name is kept and the recordings are merged. Prints per (lang->tgt, source) sentence/recording counts.
"""

import argparse
import csv
import hashlib
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict

DIRECTIONS = {("en", "ko"), ("ko", "en")}
DOMAINS = ("EA", "MA", "NA", "SA")
_TSV = {"en": "en_us", "ko": "ko_kr"}


def norm(s: str) -> str:
    return " ".join(s.split())


def make_sid(source: str, lang: str, src: str) -> str:
    return f"{source}_{lang}_{hashlib.sha1(norm(src).encode('utf-8')).hexdigest()[:8]}"


def read_jsonl(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def dedup(rows: list[dict]) -> list[dict]:
    """Merge recording rows ({lang,tgt,src,ref,source,audio}) sharing (lang, src) into sentence lines."""
    out: dict[tuple, dict] = {}
    for r in rows:
        src = norm(r["src"])
        key = (r["lang"], src)
        if key not in out:
            out[key] = {"sid": make_sid(r["source"], r["lang"], src), "lang": r["lang"], "tgt": r["tgt"],
                        "src": src, "ref": r.get("ref"), "source": r["source"], "recs": []}
        e = out[key]
        if not e["ref"] and r.get("ref"):
            e["ref"] = r["ref"]
        rec = {"audio": r["audio"]}
        if rec not in e["recs"]:
            e["recs"].append(rec)
    return list(out.values())


def select_aihub(manifest_rows: list[dict], n_sentences: int, domains=DOMAINS, stats: dict | None = None) -> list[dict]:
    """Pick whole talks round-robin over domains (biggest talk first within a domain).

    Stops as soon as the sentence count reaches n_sentences (checked after each talk), so domains
    stay balanced to within one talk. A talk id must look like `X_<DOMAIN>_...` (ValueError otherwise);
    talks of a domain not in `domains` are skipped and counted in stats["skipped_talks"].
    """
    talks: dict[str, list[dict]] = defaultdict(list)
    for r in manifest_rows:
        talks[r["group"]].append(r)
    by_dom: dict[str, list[str]] = {d: [] for d in domains}
    skipped = 0
    for g in talks:
        parts = g.split("_")
        if len(parts) < 3 or not parts[1]:
            raise ValueError(f"malformed AI Hub talk id {g!r}: expected '<X>_<DOMAIN>_<n>'")
        if parts[1] in by_dom:
            by_dom[parts[1]].append(g)
        else:
            skipped += 1
    if stats is not None:
        stats["skipped_talks"] = stats.get("skipped_talks", 0) + skipped
    for d in domains:
        by_dom[d].sort(key=lambda g: (-len(talks[g]), g))
    picked: list[dict] = []
    i = 0
    while any(i < len(v) for v in by_dom.values()):
        for d in domains:
            if i < len(by_dom[d]):
                picked += talks[by_dom[d][i]]
                if len(picked) >= n_sentences:
                    return picked
        i += 1
    return picked


def aihub_rows(manifest_rows, manifest_path, lang: str, tgt: str, n_sentences: int, stats: dict | None = None) -> list[dict]:
    base = os.path.dirname(os.path.abspath(manifest_path))
    out = []
    for r in select_aihub(manifest_rows, n_sentences, stats=stats):
        ref = r["reference"]
        src, tr = norm(ref.get("transcript") or ""), norm((ref.get("translations") or {}).get(tgt) or "")
        if not src or not tr:
            continue
        a = os.path.join(base, r["audio"])
        off = float(r.get("offset", 0.0))
        out.append({"lang": lang, "tgt": tgt, "src": src, "ref": tr, "source": "aihub",
                    "audio": f"{a}#start={off:.3f}#end={off + float(r['duration']):.3f}"})
    return out


def kosp2e_rows(manifest_rows, n_sentences: int, seed: int = 14) -> list[dict]:
    """First n usable rows after a seeded shuffle (the manifest is in corpus order, not random)."""
    manifest_rows = list(manifest_rows)
    random.Random(seed).shuffle(manifest_rows)
    out = []
    for r in manifest_rows:
        src, en = norm(r.get("transcript") or ""), norm(r.get("en") or "")
        if src and en:
            out.append({"lang": "ko", "tgt": "en", "src": src, "ref": en, "source": "kosp2e", "audio": r["audio"]})
        if len(out) >= n_sentences:
            break
    return out


def _read_tsv(path) -> list[list[str]]:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE))


def aligned_rows(aligned_path, fleurs_root) -> list[dict]:
    """Recordings from aligned.jsonl; FLEURS refs come from the other language's tsv (same sentence id)."""
    fn2id: dict[str, dict[str, str]] = {}
    id2text: dict[str, dict[str, str]] = {}
    for lang, loc in _TSV.items():
        p = os.path.join(fleurs_root, loc, "train.tsv")
        fn2id[lang], id2text[lang] = {}, {}
        if not os.path.exists(p):
            raise FileNotFoundError(f"FLEURS tsv missing: {p} (check --fleurs-root)")
        for row in _read_tsv(p):
            if len(row) >= 3:
                fn2id[lang][row[1]] = row[0]
                id2text[lang][row[0]] = row[2]
    other = {"en": "ko", "ko": "en"}
    out = []
    for r in read_jsonl(aligned_path):
        if (r["lang"], r["tgt"]) not in DIRECTIONS:
            continue
        source = "covost" if "covost" in r["audio"] else "fleurs"
        ref = None
        if source == "fleurs":
            sid = fn2id[r["lang"]].get(os.path.basename(r["audio"]))
            if sid is not None:
                ref = id2text[other[r["lang"]]].get(sid)
        out.append({"lang": r["lang"], "tgt": r["tgt"], "src": r["src"], "ref": ref, "source": source,
                    "audio": r["audio"]})
    return out


def _clip_dur(audio):
    m = re.search(r"#start=([0-9.]+)#end=([0-9.]+)", audio)
    return float(m.group(2)) - float(m.group(1)) if m else None


def length_report(sents) -> list[str]:
    """Per source: sentences, source word (eojeol) count mean/p99/max, longest clip (clipped recordings only)."""
    by = defaultdict(list)
    for s in sents:
        by[s["source"]].append(s)
    lines = []
    for src in sorted(by):
        ss = by[src]
        n = sorted(len(s["src"].split()) for s in ss)
        p99 = n[min(len(n) - 1, int(0.99 * (len(n) - 1) + 0.5))]
        durs = [d for s in ss for r in s["recs"] if (d := _clip_dur(r["audio"])) is not None]
        dmax = f"{max(durs):.1f}s" if durs else "-"
        lines.append(f"LENGTH {src:8s} sentences={len(ss):6d} words mean={sum(n) / len(n):.1f} p99={p99} "
                     f"max={n[-1]} clip_dur_max={dmax}")
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--aligned", default="models/zprobe/data/prefix13/aligned.jsonl")
    ap.add_argument("--fleurs-root", default=os.path.expanduser("~/datasets/fleurs/data"))
    ap.add_argument("--aihub-manifest", default=None)
    ap.add_argument("--aihub-sentences", type=int, default=5000)
    ap.add_argument("--aihub-en-manifest", default=None)
    ap.add_argument("--aihub-en-sentences", type=int, default=2000)
    ap.add_argument("--kosp2e-manifest", default=None)
    ap.add_argument("--kosp2e-sentences", type=int, default=3000)
    ap.add_argument("--out", default="models/zprobe/data/prefix14/sentences.jsonl")
    ap.add_argument("--seed", type=int, default=14, help="kosp2e sample order")
    args = ap.parse_args()

    stats: dict = {}
    rows = aligned_rows(args.aligned, args.fleurs_root)
    if args.aihub_manifest:
        rows += aihub_rows(read_jsonl(args.aihub_manifest), args.aihub_manifest, "ko", "en", args.aihub_sentences, stats)
    if args.aihub_en_manifest:
        rows += aihub_rows(read_jsonl(args.aihub_en_manifest), args.aihub_en_manifest, "en", "ko",
                           args.aihub_en_sentences, stats)
    if args.kosp2e_manifest:
        rows += kosp2e_rows(read_jsonl(args.kosp2e_manifest), args.kosp2e_sentences, args.seed)
    sents = dedup(rows)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    tmp = args.out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for s in sents:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    os.replace(tmp, args.out)  # the chain skips this stage when the file exists: never leave a partial one

    ns, nr, tot = Counter(), Counter(), Counter()
    for s in sents:
        k = (f"{s['lang']}->{s['tgt']}", s["source"])
        ns[k] += 1
        nr[k] += len(s["recs"])
        tot[k[0]] += 1
    for k in sorted(ns):
        print(f"{k[0]} {k[1]:8s} sentences={ns[k]:6d} recordings={nr[k]:6d}")
    for d in sorted(tot):
        print(f"TOTAL {d} sentences={tot[d]}")
    noref = sum(1 for r in rows if r["source"] == "fleurs" and not r["ref"])
    print(f"FLEURS recordings with ref=None: {noref}")
    print(f"AI Hub talks skipped (domain not in {DOMAINS}): {stats.get('skipped_talks', 0)}")
    for line in length_report(sents):
        print(line)
    print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
