"""조건별 `H_set` — 루프의 목적함수로 비교군과 우리 정책을 같은 자로 잰다.

    H_set(S) = (1/|L|) Σ_{t∈L} QE_t(x, ŷ_t(S)) · (1 − max_{j∈S} c_j)
    c_j      = P_NLI(contradiction | premise = x, hypothesis = x_{<j})

**왜 재나.** 루프는 이 값을 최대화하도록 프롬프트를 고쳤고 dev 에서 +0.0187 을 얻었는데,
같은 분절을 참조 기반 COMET 으로 재면 채택본과 초기본 차이가 ±0.005 다. 목적함수가
참조 기반 지표와 같은 순서를 내는지가 확인된 적이 없다 — 비교군까지 같은 자로 놓으면
그 질문에 답이 된다.

**두 항의 성격이 다르다.**

- `c_j` 는 **소스만** 본다 (`labels.apply_source_contra` 와 같은 정의). 타깃과 무관하고
  문장·위치마다 한 번만 재면 되므로, 후보 위치 전체를 미리 재 두고 조건마다 max 만 취한다.
- `QE_t` 는 조각 번역을 이어 붙인 것을 원문 전체와 대 본다 (= `cohesion`). 조건마다 다르고
  타깃마다 다르므로 여기가 비용의 전부다. 같은 (원문, 가설) 쌍은 백엔드가 memo 로 건너뛴다.

**비교군 둘은 타깃마다 분절이 다르다** (`causal_align` 은 정렬 대상이, `mu_prefix` 는 NMT 가
타깃별이다). 그 둘은 타깃별 S 로 각각 재서 평균한다 — 루프가 타깃 하나의 S 를 공유해 쓰는
것과 다르므로 표에 올릴 때 밝힐 것.

비용: **API $0**, GPU 만. 번역은 `bleu_eval` 이 남긴 것을 그대로 읽는다.

    python3 <이 파일> --run full_j44v0 --run full_j44best
"""
import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

from core.meaning_segmentator.autoseg.runtime import metrics            # noqa: E402
from core.meaning_segmentator.autoseg.runtime.labels import units_of, join_units  # noqa: E402
from core.meaning_segmentator.autoseg.scoring import bleu_eval as BE    # noqa: E402

A = REPO / "core/meaning_segmentator/experiment/artifacts/en2x/covost2"
POLICIES = ["punct", "alignatt", "mu_prefix", "causal_align", "syntax"]
TGT_LANG = {"zh": "Chinese", "de": "German", "ja": "Japanese"}

p = argparse.ArgumentParser()
p.add_argument("--run", action="append", required=True)
p.add_argument("--targets", nargs="+", default=["zh", "de", "ja"])
p.add_argument("--t-grid", type=float, nargs="+",
               default=[2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 7, 8, 10])
p.add_argument("--score-grid", type=int, nargs="+",
               default=[0, 2, 5, 10, 20, 40, 60, 80, 90, 95, 99, 100])
p.add_argument("--nli-batch", type=int, default=64)
p.add_argument("--qe-batch", type=int, default=64)
a = p.parse_args()

SPACED = True          # 소스가 영어다
LABEL = {"full_j44v0": "auto_j44v0", "full_j44best": "auto_j44best"}


def log(*x):
    print(f"[{time.strftime('%T')}]", *x, flush=True)


# ── 1. 소스 NLI — 후보 위치 전체를 한 번에 ──────────────────────────────────
rows = json.loads((A / "full" / "prompt_eval" /
                   f"{LABEL[a.run[0]]}_test.json").read_text())["rows"]
texts = [r["text"] for r in rows]
units = [units_of(t, SPACED) for t in texts]
log(f"문장 {len(rows)} / 후보 위치 {sum(len(u) - 1 for u in units)}")

contra_path = A / "full_judge44" / "source_contra_test.json"
if contra_path.exists():
    contra = {int(k): v for k, v in json.loads(contra_path.read_text()).items()}
    log(f"소스 NLI 재사용 — {contra_path.name}")
else:
    nli = metrics.make_contradiction_backend(batch_size=a.nli_batch)
    prem, hyp, owner = [], [], []
    for i, u in enumerate(units):
        for j in range(1, len(u)):
            prem.append(texts[i]); hyp.append(join_units(u[:j], SPACED)); owner.append(i)
    t0 = time.time()
    vals, _ = nli.score_dual(prem, hyp)
    contra = {}
    for k, i in enumerate(owner):
        contra.setdefault(i, []).append(round(vals[k], 5))
    contra_path.write_text(json.dumps({str(k): v for k, v in contra.items()}),
                           encoding="utf-8")
    log(f"소스 NLI {len(prem)}쌍 {time.time() - t0:.0f}s -> {contra_path.name}")
    del nli


def cuts_of(pieces: list[str]) -> list[int]:
    """조각 목록 → 절단 위치(소스 단위 인덱스). 마지막 조각 뒤는 경계가 아니다."""
    out, acc = [], 0
    for x in pieces[:-1]:
        acc += len(units_of(x, SPACED))
        out.append(acc)
    return out


qe = metrics.make_adequacy_backend("cometkiwi", batch_size=a.qe_batch)
out_path = A / "full_judge44" / "hset_conditions.json"
result = json.loads(out_path.read_text()) if out_path.exists() else {}

for run in a.run:
    D = A / run
    blob = {t: json.loads((D / "bleu" / f"{t}.json").read_text())["conditions"]
            for t in a.targets}
    ev = json.loads((A / "full" / "prompt_eval" /
                     f"{LABEL[run]}_test.json").read_text())["rows"]
    # 조건 정의를 번역 없이 다시 만든다 — 절단 위치가 산출물에 안 남아 있다.
    conds = {t: {**BE.build_conditions(ev, a.t_grid, SPACED, 8, True, True,
                                       a.score_grid, True),
                 **BE.load_baseline_conditions(D, POLICIES, t, "test", ev,
                                               a.t_grid, SPACED)}
             for t in a.targets}
    names = sorted(set.intersection(*(set(conds[t]) for t in a.targets)))
    log(f"== {run}: 조건 {len(names)}")
    for n in names:
        # **런에 따라 달라지는 것은 `auto_S*` 뿐이다.** 비교군·무분절·기계분절은 정책과
        # 번역이 같아 두 런의 가설이 바이트까지 같다(확인함). 런마다 다시 재면 조건
        # 124개가 되는데 공유하면 74개다.
        key = f"{run}/{n}" if n.startswith("auto_") else n
        if key in result:
            continue
        per_t = []
        for t in a.targets:
            cell = blob[t].get(n)
            if cell is None or n not in conds[t]:
                break
            hyps = cell["hyps"]
            q = qe.score(texts, hyps)
            pen = []
            for i, row in enumerate(conds[t][n]):
                cj = [contra[i][j - 1] for j in cuts_of(row["pieces"])
                      if 1 <= j <= len(contra.get(i, []))]
                # 절단이 없으면 반박당할 방출도 없다 — 벌점 항이 1 이다.
                pen.append(1.0 - max(cj) if cj else 1.0)
            per_t.append([qi * pi for qi, pi in zip(q, pen)])
        if len(per_t) != len(a.targets):
            continue
        # 문장마다 타깃 평균을 먼저 내고(식의 1/|L|), 그 다음 코퍼스 평균.
        hset = sum(sum(col) / len(col) for col in zip(*per_t)) / len(rows)
        result[key] = {"hset": round(hset, 5),
                       "laal_ms": blob[a.targets[0]][n]["laal_ms"],
                       "comet": {t: blob[t][n]["comet"] for t in a.targets},
                       "k": blob[a.targets[0]][n]["k"]}
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=1),
                            encoding="utf-8")
        log(f"   {n:20s} H_set {hset:.4f}  laal {result[key]['laal_ms']:.0f}ms")
log(f"-> {out_path}")
