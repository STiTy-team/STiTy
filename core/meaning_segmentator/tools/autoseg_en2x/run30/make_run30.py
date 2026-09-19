"""run30 — train 을 키우고 dev 를 단일로 되돌린다.

judge23 이 남긴 진단: 확인 분할(test-B)이 **고르기 편향**은 걸러냈지만(이터 1 −0.0014 탈락)
개정 자체의 **과적합**은 못 막았다(이터 3 은 두 단계를 다 통과하고 최종 홀드아웃에서 −0.0047).
규칙 한 줄이 사례 45개에서 귀납되는데 그 표본이 좁은 것이 원인이다. 그러면 손댈 곳은 확인
관문이 아니라 **귀납 기반**이다 — train 을 300 → 500 으로 키워 사례를 100 까지 쓴다.

dev 를 500 단일로 되돌리는 것도 같은 판단이다. judge23 은 dev 를 300/300 으로 쪼개 구간 반폭이
0.0102 → 0.0147 로 벌어졌고, 효과 크기(+0.01~0.017)와 같은 자릿수가 됐다. 500 단일이면 약
0.0112 로 돌아온다.

배분은 **prefix 를 보존한다** — dev·test 의 앞부분을 그대로 두고 뒤에서 덜어 train 에 넣는다.
분절·번역 캐시가 원문 기준이라 대부분 그대로 적중한다.

    train 500 = run28 train 300 + dev[500:] 100 + test[560:] 100
    dev   500 = run28 dev[:500]
    test  560 = run28 test[:560]

홀드아웃에서 train 으로 내리는 것은 오염이 아니다 — 판정에 쓴 문장을 홀드아웃에 올리는
반대 방향만 문제다.

    .venv-autoseg/bin/python core/meaning_segmentator/tools/autoseg_en2x/run30/make_run30.py
"""
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
SRC = ROOT / "core/meaning_segmentator/experiment/artifacts/en2x/en-multi/run28"
DST = ROOT / "core/meaning_segmentator/experiment/artifacts/en2x/en-multi/run30"
N_DEV, N_TEST = 500, 560


def read(p):
    return json.loads(p.read_text(encoding="utf-8"))


def write(p, obj):
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")


def main():
    (DST / "data").mkdir(parents=True, exist_ok=True)
    src = {n: read(SRC / f"data/{n}.json") for n in ("train", "dev", "test")}
    lab = {n: read(SRC / f"oracle_labels_{n}.json") for n in ("train", "dev", "test")}
    targets = list(lab["train"])

    new = {"dev": src["dev"][:N_DEV], "test": src["test"][:N_TEST]}
    new["train"] = src["train"] + src["dev"][N_DEV:] + src["test"][N_TEST:]

    # 라벨은 **어느 분할에서 왔든 그 문장의 것을 그대로** 따라간다 — id 로 모아 두고 꺼낸다.
    by_id = {t: {} for t in targets}
    for split, rows in lab.items():
        for t in targets:
            for r in rows[t]:
                by_id[t][r["id"]] = r

    for name, sents in new.items():
        write(DST / f"data/{name}.json", sents)
        ids = [s["id"] for s in sents]
        assert len(set(ids)) == len(ids), f"{name} 에 중복 id 가 있다"
        out = {}
        for t in targets:
            rows = [by_id[t][i] for i in ids if i in by_id[t]]
            assert len(rows) == len(ids), f"{name}/{t} 라벨 {len(rows)} != 문장 {len(ids)}"
            out[t] = rows
        write(DST / f"oracle_labels_{name}.json", out)
        print(f"{name:6s} 문장 {len(sents):4d} / 라벨 타깃 {len(out)}")

    total = sum(len(v) for v in new.values())
    assert total == sum(len(v) for v in src.values()), f"총량이 바뀌었다: {total}"
    allid = [s["id"] for v in new.values() for s in v]
    assert len(set(allid)) == len(allid), "분할 간 문장이 겹친다"
    print(f"총량 {total} (run28 과 같다) / 겹침 0")

    for f in ("config.json", "language_profile.json", "measured_profile.json"):
        if (SRC / f).exists():
            shutil.copy2(SRC / f, DST / f)
    for f in SRC.glob("contra_floor_*.json"):
        shutil.copy2(f, DST / f.name)

    link = DST / "cache"
    if not link.exists():
        link.symlink_to("../run28/cache")
    print("cache -> ../run28/cache (심링크) — 문장이 같아 분절·번역이 적중한다")


if __name__ == "__main__":
    main()
