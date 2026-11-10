"""prefix13 stage 1: the ko->en expansion folded into the aligned file.

Recordings already in prefix12/aligned.jsonl pass through unchanged. Every other recording of
ko_all_manifest gets a row when (a) its sentence has fine chunks - phrases1 (Qwen3-8B, ok rows) or
pieces_ko_fail (luna --fine rechunk of the 692 Qwen3-8B failures) - and (b) word-level alignment
exists in word_times.jsonl: chunk ends are derived by mapping cumulative alnum chunk boundaries
onto word end times. adds come from the terra relabels (full_fine_terra, fine_ok285_terra,
fine_fail692_terra - later files win), falling back to the chunking model's own pieces.

usage: .venv/bin/python core/research/zprobe/labels/build_prefix13_aligned.py
"""
import argparse
import json
import os

ap = argparse.ArgumentParser()
ap.add_argument("--base-aligned", default="models/zprobe/data/prefix12/aligned.jsonl")
ap.add_argument("--manifest", default="models/zprobe/data/prefix13/ko_all_manifest.jsonl")
ap.add_argument("--word-times", default="models/zprobe/data/prefix12/word_times.jsonl")
ap.add_argument("--phrases1", default="models/zprobe/data/prefix3/phrases1.jsonl")
ap.add_argument("--luna-pieces", default="models/zprobe/data/prefix13/pieces_ko_fail.jsonl")
ap.add_argument("--relabels", nargs="+", default=[
    "models/zprobe/data/relabel/full_fine_terra.jsonl",
    "models/zprobe/data/relabel/fine_ok285_terra.jsonl",
    "models/zprobe/data/relabel/fine_fail692_terra.jsonl"])
ap.add_argument("--out", default="models/zprobe/data/prefix13/aligned.jsonl")
a = ap.parse_args()


def alnum(s):
    return sum(c.isalnum() for c in s)


chunks_by_sid = {}
firstpass_by_sid = {}
for l in open(a.phrases1, encoding="utf-8"):
    r = json.loads(l)
    if r["lang"] == "ko" and not r.get("fail"):
        chunks_by_sid[r["sid"]] = r["phrases"]
        firstpass_by_sid[r["sid"]] = r["adds"]
if os.path.exists(a.luna_pieces):
    for l in open(a.luna_pieces, encoding="utf-8"):
        r = json.loads(l)
        if not r.get("fail") and r.get("chunks"):
            chunks_by_sid[r["sid"]] = r["chunks"]
            firstpass_by_sid[r["sid"]] = r["pieces"]

adds_by_sid = {}
for path in a.relabels:
    if not os.path.exists(path):
        continue
    for l in open(path, encoding="utf-8"):
        r = json.loads(l)
        adds_by_sid[r["sid"]] = r["adds"]

words_by_path = {}
for l in open(a.word_times, encoding="utf-8"):
    r = json.loads(l)
    words_by_path[os.path.abspath(r["audio"].split("#")[0])] = r["words"]

have = set()
os.makedirs(os.path.dirname(a.out), exist_ok=True)
out = open(a.out, "w", encoding="utf-8")
base = 0
for l in open(a.base_aligned, encoding="utf-8"):
    out.write(l)
    base += 1
    r = json.loads(l)
    have.add(r["id"])

added = no_chunks = no_words = map_fail = terra = fallback = 0
for l in open(a.manifest, encoding="utf-8"):
    r = json.loads(l)
    if r["id"] in have:
        continue
    chunks = chunks_by_sid.get(r["sid"])
    if not chunks or len(chunks) < 2:
        no_chunks += 1
        continue
    ws = words_by_path.get(os.path.abspath(r["audio"]))
    if not ws:
        no_words += 1
        continue
    counts = [sum(c.isalnum() for c in w["w"]) for w in ws]
    total_w = sum(counts)
    if total_w != alnum(r["src"]):
        map_fail += 1
        continue
    marks = [alnum("".join(chunks[: k + 1])) for k in range(len(chunks) - 1)]
    ends, cum = [], 0
    wi = 0
    for mark in marks:
        while wi < len(ws) and cum < mark:
            cum += counts[wi]
            wi += 1
        if cum != mark:
            break
        ends.append(ws[wi - 1]["end"])
    dur = ws[-1]["end"]
    if len(ends) != len(marks) or any(b <= x for x, b in zip(ends, ends[1:])) \
            or (ends and ends[-1] > dur - 0.2):
        map_fail += 1
        continue
    adds = adds_by_sid.get(r["sid"])
    if adds and len(adds) == len(chunks):
        terra += 1
    else:
        adds = firstpass_by_sid.get(r["sid"])
        if not adds or len(adds) != len(chunks):
            map_fail += 1
            continue
        fallback += 1
    out.write(json.dumps({"id": r["id"], "sid": r["sid"], "audio": r["audio"],
                          "src": r["src"], "lang": "ko", "tgt": "en",
                          "dur": round(dur + 0.2, 2), "adds": adds,
                          "ends": [round(e, 2) for e in ends] + [round(dur + 0.2, 2)]},
                         ensure_ascii=False) + "\n")
    added += 1
print(f"base {base} | ko->en 신규 {added} (terra {terra}, 1차 대체 {fallback}) | "
      f"제외: 분할없음 {no_chunks}, 정렬없음 {no_words}, 매핑실패 {map_fail}")
