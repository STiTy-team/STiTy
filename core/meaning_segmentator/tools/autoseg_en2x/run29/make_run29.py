"""run29 — run28 의 dev 600 을 판정 300 / 확인 300 으로 가른다.

judge21 이 dev 600 하나로 후보를 12번 재고 최고를 골라 채택했는데 최종 홀드아웃에서
Δ −0.0089 로 뒤집혔다. 선택 편향이다: 12번 재면 효과가 0 인 개정도 하나쯤 하한이 0 을 넘는다.
고르는 분할(test_a)과 확인하는 분할(test_b)을 나누면 **확인 단계에는 고르기가 없으므로**
그 편향이 없다.

파일 이름이 test_a/test_b 인 것은 `loop_judge` 의 3분할 경로(`scheme3`)가 그 이름을 본다.
문장·라벨은 run28 것을 그대로 쓴다 — 캐시가 전부 적중해 iter 0 채점이 0원이다.

    .venv-autoseg/bin/python core/meaning_segmentator/tools/autoseg_en2x/run29/make_run29.py
"""
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
SRC = ROOT / "core/meaning_segmentator/experiment/artifacts/en2x/en-multi/run28"
DST = ROOT / "core/meaning_segmentator/experiment/artifacts/en2x/en-multi/run29"
HALF = 300


def read(p):
    return json.loads(p.read_text(encoding="utf-8"))


def write(p, obj):
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")


def main():
    (DST / "data").mkdir(parents=True, exist_ok=True)
    dev = read(SRC / "data/dev.json")
    lab = read(SRC / "oracle_labels_dev.json")
    assert len(dev) == 2 * HALF, f"dev 가 {len(dev)} 개다 — {2 * HALF} 를 기대했다"

    # **문장 순서를 그대로 가른다.** 섞으면 run28 의 dev 인덱스와 어긋나 캐시 키(원문 기준)는
    # 맞아도 사람이 두 런을 비교할 때 헷갈린다. dev 자체가 이미 세 출처를 섞어 만든 것이다.
    halves = {"test_a": dev[:HALF], "test_b": dev[HALF:]}
    ids = {k: {s["id"] for s in v} for k, v in halves.items()}
    assert not (ids["test_a"] & ids["test_b"]), "두 절반이 겹친다"

    for name, sents in halves.items():
        write(DST / f"data/{name}.json", sents)
        sub = {t: [r for r in rows if r["id"] in ids[name]] for t, rows in lab.items()}
        for t, rows in sub.items():
            assert len(rows) == HALF, f"{name}/{t} 라벨이 {len(rows)} 개다"
        write(DST / f"oracle_labels_{name}.json", sub)
        print(f"{name}: 문장 {len(sents)} / 타깃 {len(sub)}")

    for name in ("train", "test"):
        shutil.copy2(SRC / f"data/{name}.json", DST / f"data/{name}.json")
        shutil.copy2(SRC / f"oracle_labels_{name}.json", DST / f"oracle_labels_{name}.json")
        print(f"{name}: run28 에서 그대로 복사")

    for f in ("config.json", "language_profile.json", "measured_profile.json",
              "pseudoref_test.json"):
        if (SRC / f).exists():
            shutil.copy2(SRC / f, DST / f)
    for f in SRC.glob("contra_floor_*.json"):
        shutil.copy2(f, DST / f.name)

    # 캐시는 **심링크**다. 문장이 run28 과 같으니 분절·번역이 전부 적중한다. 런을 두 개 동시에
    # 돌리면 서로의 캐시를 덮으므로 judge 런은 하나씩만 띄운다.
    link = DST / "cache"
    if not link.exists():
        link.symlink_to("../run28/cache")
    print("cache -> ../run28/cache (심링크)")


if __name__ == "__main__":
    main()
