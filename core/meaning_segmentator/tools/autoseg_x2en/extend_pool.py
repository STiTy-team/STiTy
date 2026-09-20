"""x2en 소스 언어의 판정용 매니페스트를 **뒤에 붙여** 늘린다 (loop905 → 전체 풀).

왜 붙이기만 하나. `data.stratified_order` 는 층마다 섞으므로 풀에 문장이 하나만 늘어도
순열 전체가 달라진다 — 기존 905문장의 라벨(문장 단위로 이미 계산해 둔 것)을 쓰려면
**기존 행이 그 자리에 그대로 있어야** 한다. 그래서 앞 905행은 건드리지 않고 남은 문장만
뒤에 붙인다. 붙는 순서는 FLEURS 숫자 id 오름차순 — 결정적이고 재현된다.

풀의 정의는 영어 쪽 `pool1580` 과 같다: **소스 train+dev ∩ en train+dev, 25자 이상.**
FLEURS `test` 스플릿은 넣지 않는다(STT·AST 평가용 봉인).

    python -m core.meaning_segmentator.tools.autoseg_x2en.extend_pool --lang de [--dry-run]
"""
import argparse, csv, json, pathlib, sys

FL = pathlib.Path.home() / "datasets/fleurs/data"
M = pathlib.Path("evaluation/ast/manifests")
CODE = {"de": "de_de", "ja": "ja_jp", "zh": "cmn_hans_cn"}
MIN_CHARS = 25


def fleurs(lang_code: str, splits=("train", "dev")) -> dict[str, tuple[str, str]]:
    """{id: (문장, 스플릿)}. **`csv.QUOTE_NONE` 이 필수다** — FLEURS TSV 에 따옴표가 그대로
    들어 있어 기본 파서는 여러 행을 한 필드로 삼킨다(실측으로 2,916어절짜리 잔해가 나왔다)."""
    out: dict[str, tuple[str, str]] = {}
    for s in splits:
        p = FL / lang_code / f"{s}.tsv"
        if not p.exists():
            continue
        with open(p, encoding="utf-8") as f:
            for r in csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
                if len(r) >= 4 and r[0] not in out:
                    out[r[0]] = (r[3].strip(), s)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lang", choices=sorted(CODE), required=True)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()

    base = M / f"fleurs_nway_{a.lang}-en_multi2en_loop905.jsonl"
    rows = [json.loads(l) for l in open(base, encoding="utf-8")]
    have = {r["utt_id"] for r in rows}

    src, en = fleurs(CODE[a.lang]), fleurs("en_us")
    pool = {k: v for k, v in src.items() if k in en and len(v[0]) >= MIN_CHARS}
    new_ids = sorted((k for k in pool if f"{CODE[a.lang]}_{k}" not in have and k not in have),
                     key=lambda x: int(x) if x.isdigit() else x)

    # 기존 행의 utt_id 가 "de_de_1135" 꼴이므로 같은 꼴로 맞춘다.
    prefix = rows[0]["utt_id"].rsplit("_", 1)[0]
    added = [{"utt_id": f"{prefix}_{i}", "src_lang": rows[0]["src_lang"], "tgt_lang": "en",
              "src_text": pool[i][0], "tgt_text": en[i][0], "talk_id": int(i) if i.isdigit() else i,
              "fleurs_split": pool[i][1]} for i in new_ids]
    added = [r for r in added if r["utt_id"] not in have]
    total = len(rows) + len(added)
    out = M / f"fleurs_nway_{a.lang}-en_multi2en_loop{total}.jsonl"
    print(f"{a.lang}: 기존 {len(rows)} + 새로 {len(added)} = {total}  -> {out.name}")
    if added:
        n = [len(r["src_text"]) for r in added]
        print(f"  붙는 문장 길이 {min(n)}~{max(n)}자 (평균 {sum(n)//len(n)})")
    if a.dry_run:
        return 0
    with open(out, "w", encoding="utf-8") as f:
        for r in rows + added:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    assert [json.loads(l)["utt_id"] for l in open(out, encoding="utf-8")][:len(rows)] == [r["utt_id"] for r in rows], \
        "앞 905행이 움직였다"
    print(f"  앞 {len(rows)}행 그대로 확인")
    return 0


if __name__ == "__main__":
    sys.exit(main())
