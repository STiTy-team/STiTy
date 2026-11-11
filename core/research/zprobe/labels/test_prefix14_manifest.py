import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_prefix14_manifest as m  # noqa: E402


def _w(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def test_dedup_merges_recordings_with_same_sid():
    rows = [
        {"lang": "en", "tgt": "ko", "src": "Hello  world", "ref": None, "source": "fleurs", "audio": "/a.wav"},
        {"lang": "en", "tgt": "ko", "src": "Hello world", "ref": "안녕", "source": "covost", "audio": "/b.wav"},
        {"lang": "ko", "tgt": "en", "src": "Hello world", "ref": None, "source": "fleurs", "audio": "/c.wav"},
    ]
    out = m.dedup(rows)
    assert len(out) == 2
    en = [r for r in out if r["lang"] == "en"][0]
    assert [r["audio"] for r in en["recs"]] == ["/a.wav", "/b.wav"]
    assert en["source"] == "fleurs" and en["ref"] == "안녕"
    assert en["sid"] == m.make_sid("fleurs", "en", "Hello world")
    assert en["sid"].startswith("fleurs_en_") and len(en["sid"]) == len("fleurs_en_") + 8


def _talk_rows(group, n, dur=1.0):
    return [
        {"id": f"{group}_{i}", "audio": f"audio/{group}.wav", "duration": dur, "group": group,
         "src_lang": "ko", "offset": float(i), "reference": {"transcript": f"s {group} {i}", "translations": {"en": f"t {group} {i}"}}}
        for i in range(n)
    ]


def test_select_aihub_balances_domains_and_returns_sentences():
    rows = []
    for dom in ("EA", "MA", "NA", "SA"):
        for k in range(3):
            rows += _talk_rows(f"K_{dom}_{k}", 5 + k)
    sel = m.select_aihub(rows, 29)
    assert all("reference" in r for r in sel)
    groups = {r["group"] for r in sel}
    per_dom = {}
    for g in groups:
        per_dom[g.split("_")[1]] = per_dom.get(g.split("_")[1], 0) + 1
    assert per_dom == {"EA": 2, "MA": 2, "NA": 2, "SA": 2}
    # largest talks first within a domain
    assert "K_EA_2" in groups and "K_EA_1" in groups and "K_EA_0" not in groups


def test_aihub_audio_gets_start_end(tmp_path):
    d = tmp_path / "ko"
    d.mkdir()
    rows = _talk_rows("K_EA_1", 2, dur=1.5)
    rows[1]["reference"]["translations"]["en"] = ""  # skipped: empty ref
    mp = d / "manifest.jsonl"
    _w(mp, rows)
    out = m.aihub_rows(m.read_jsonl(mp), mp, "ko", "en", 10)
    assert len(out) == 1
    assert out[0]["audio"] == f"{d}/audio/K_EA_1.wav#start=0.000#end=1.500"
    assert out[0]["source"] == "aihub" and out[0]["ref"] == "t K_EA_1 0"


def test_fleurs_ref_lookup_by_sentence_id(tmp_path):
    root = tmp_path / "fleurs"
    for loc, line in (("en_us", "7\ta.wav\tHello there.\thello there\tx\t1\tM"),
                      ("ko_kr", "7\tb.wav\t안녕하세요.\t안녕하세요\tx\t1\tF\n8\tc.wav\t혼자.\t혼자\tx\t1\tF")):
        (root / loc).mkdir(parents=True)
        (root / loc / "train.tsv").write_text(line + "\n", encoding="utf-8")
    aligned = tmp_path / "aligned.jsonl"
    _w(aligned, [
        {"audio": "/x/fleurs/en_us/train/a.wav", "src": "Hello there.", "lang": "en", "tgt": "ko"},
        {"audio": "/x/fleurs/ko_kr/train/c.wav", "src": "혼자.", "lang": "ko", "tgt": "en"},
        {"audio": "/x/covost/q.wav", "src": "Cov.", "lang": "en", "tgt": "ko"},
        {"audio": "/x/covost/z.wav", "src": "ja", "lang": "ja", "tgt": "ko"},
    ])
    out = m.aligned_rows(aligned, root)
    assert len(out) == 3
    by = {r["src"]: r for r in out}
    assert by["Hello there."]["ref"] == "안녕하세요." and by["Hello there."]["source"] == "fleurs"
    assert by["혼자."]["ref"] is None
    assert by["Cov."]["source"] == "covost" and by["Cov."]["ref"] is None


def test_kosp2e_rows_are_ko_to_en(tmp_path):
    mp = tmp_path / "k.jsonl"
    _w(mp, [{"id": "z1", "audio": "/z/1.wav", "lang": "ko", "transcript": "안녕", "en": "hi", "duration": 1.0},
            {"id": "z2", "audio": "/z/2.wav", "lang": "ko", "transcript": "빈", "en": "", "duration": 1.0}])
    out = m.kosp2e_rows(m.read_jsonl(mp), 10)
    assert len(out) == 1
    assert (out[0]["lang"], out[0]["tgt"], out[0]["ref"], out[0]["source"]) == ("ko", "en", "hi", "kosp2e")
