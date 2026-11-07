"""aligned_plus2: the x->en side rebuilt on the luna relabel, at full FLEURS scale.

Three moves on top of build_aligned_plus.py: (1) every x->en recording whose sentence has a luna
relabel (relabel_adds_en.py) gets its adds swapped - the en-side twin of the prefix7 swap; (2) the
first-pass failures the relabel rescued become usable, and (3) recordings of leak-ok sentences that
the original align pass never covered get aligned now. Ends depend only on the chunk split, which no
relabel touches, so existing alignments are reused and only new sids go through the aligner.

prep  : writes manifest_missing / pieces_missing for the align stage (sids usable now but absent
        from connective_en/aligned.jsonl).
merge : prefix7 aligned en->ko rows + the full x->en pool (old aligned + aligned_extra) with
        luna adds (old pieces where the relabel failed and the first pass was leak-ok).

usage: .venv/bin/python core/research/zprobe/labels/build_aligned_plus2.py prep
       (build_connective_en.py align --manifest ... --pieces ... --out aligned_extra.jsonl)
       .venv/bin/python core/research/zprobe/labels/build_aligned_plus2.py merge"""
import argparse
import json

ap = argparse.ArgumentParser()
ap.add_argument("stage", choices=["prep", "merge"])
ap.add_argument("--relabel", default="models/zprobe/data/relabel/full_xen.jsonl")
ap.add_argument("--pieces", default="models/zprobe/data/connective_en/pieces1.jsonl")
ap.add_argument("--manifest", default="models/zprobe/data/connective_en/manifest.jsonl")
ap.add_argument("--aligned", default="models/zprobe/data/connective_en/aligned.jsonl")
ap.add_argument("--aligned-extra", default="models/zprobe/data/connective_en/aligned_extra.jsonl")
ap.add_argument("--base", default="models/zprobe/data/prefix7/aligned.jsonl")
ap.add_argument("--out", default="models/zprobe/data/prefix7/aligned_plus2.jsonl")
a = ap.parse_args()

luna = {}
for l in open(a.relabel, encoding="utf-8"):
    r = json.loads(l)
    luna[r["sid"]] = r["adds"]

pieces = {}
for l in open(a.pieces, encoding="utf-8"):
    r = json.loads(l)
    pieces[r["sid"]] = r

usable = {s for s, r in pieces.items()
          if (s in luna and len(luna[s]) == len(r["chunks"]))
          or (not r.get("fail") and r.get("pieces") and len(r["pieces"]) == len(r["chunks"]))}

if a.stage == "prep":
    aligned_sids = {json.loads(l)["sid"] for l in open(a.aligned, encoding="utf-8")}
    missing = usable - aligned_sids
    dirn = a.pieces.rsplit("/", 1)[0]
    n = 0
    with open(f"{dirn}/manifest_missing.jsonl", "w", encoding="utf-8") as f:
        for l in open(a.manifest, encoding="utf-8"):
            if json.loads(l)["sid"] in missing:
                f.write(l)
                n += 1
    with open(f"{dirn}/pieces_missing.jsonl", "w", encoding="utf-8") as f:
        for s in sorted(missing):
            r = dict(pieces[s])
            if s in luna:
                r["pieces"] = luna[s]
            r["fail"] = ""
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"usable sids {len(usable)}, already aligned {len(usable & aligned_sids)}, "
          f"to align: {len(missing)} sids / {n} recordings", flush=True)
    print("PREP_DONE", flush=True)
else:
    out = open(a.out, "w", encoding="utf-8")
    n_enko = 0
    for l in open(a.base, encoding="utf-8"):
        r = json.loads(l)
        if r["lang"] == "en" and r["tgt"] == "ko":
            out.write(l)
            n_enko += 1
    added = {"ko": 0, "ja": 0, "zh": 0}
    swapped = kept_old = 0
    seen = set()
    for path in (a.aligned, a.aligned_extra):
        try:
            fh = open(path, encoding="utf-8")
        except FileNotFoundError:
            continue
        for l in fh:
            r = json.loads(l)
            sid = r["sid"]
            if sid not in usable or (sid, r["audio"]) in seen:
                continue
            seen.add((sid, r["audio"]))
            adds = r.get("pieces") or r.get("adds")
            if sid in luna and len(luna[sid]) == len(r["ends"]):
                adds = luna[sid]
                swapped += 1
            elif pieces[sid].get("fail"):
                continue
            else:
                kept_old += 1
            if not adds or len(adds) != len(r["ends"]):
                continue
            out.write(json.dumps({"id": r["id"], "sid": sid, "audio": r["audio"],
                                  "src": r["src"], "lang": r["lang"], "tgt": "en",
                                  "dur": float(r["dur"]), "adds": adds, "ends": r["ends"]},
                                 ensure_ascii=False) + "\n")
            added[r["lang"]] += 1
    out.close()
    print(f"en->ko kept {n_enko}, x->en added {added} (luna {swapped}, old pieces {kept_old})", flush=True)
    print("MERGE_DONE", flush=True)
