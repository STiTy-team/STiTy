"""IWSLT17 en-ko train 에서 강연 한 편을 뽑아 data/talk<id>.jsonl 로 쓴다.

기본은 talkid 231 (Frank Gehry, 398문장) — 256문장 이상 29편 중 가장 길다.
한 줄: {"idx", "en", "ko"}. idx 는 강연 안 순번(0부터).
"""
import argparse
import json
import re
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
REPO_FILE = "data/2017-01-trnted/texts/en/ko/en-ko.zip"


def talks(lines):
    out, cur = {}, None
    for line in lines:
        s = line.strip()
        if s.startswith("<doc"):
            cur = {"meta": {}, "segs": []}
            continue
        m = re.match(r"<(\w+)>(.*)</\1>", s)
        if m:
            cur["meta"][m.group(1)] = m.group(2)
            if m.group(1) == "talkid":
                out[m.group(2)] = cur
        elif s.startswith("<") or cur is None:
            continue
        else:
            cur["segs"].append(s)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--talkid", default="231")
    args = ap.parse_args()

    from huggingface_hub import hf_hub_download

    zpath = hf_hub_download("IWSLT/iwslt2017", REPO_FILE, repo_type="dataset")
    with zipfile.ZipFile(zpath) as z:
        en = talks(z.read("en-ko/train.tags.en-ko.en").decode("utf-8").splitlines())
        ko = talks(z.read("en-ko/train.tags.en-ko.ko").decode("utf-8").splitlines())
    e, k = en[args.talkid], ko[args.talkid]
    if len(e["segs"]) != len(k["segs"]):
        raise SystemExit(f"line count mismatch: en {len(e['segs'])} ko {len(k['segs'])}")
    if any(not a or not b for a, b in zip(e["segs"], k["segs"])):
        raise SystemExit("empty line in talk")

    out = HERE / "data" / f"talk{args.talkid}.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for i, (a, b) in enumerate(zip(e["segs"], k["segs"])):
            f.write(json.dumps({"idx": i, "en": a, "ko": b}, ensure_ascii=False) + "\n")
    meta = {k2: e["meta"].get(k2) for k2 in ("talkid", "title", "speaker", "url", "keywords")}
    meta.update(source=f"IWSLT/iwslt2017:{REPO_FILE} (train.tags.en-ko)", sentences=len(e["segs"]))
    (HERE / "data" / f"talk{args.talkid}.meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out} ({len(e['segs'])} sentences): {meta['title']}")


if __name__ == "__main__":
    main()
