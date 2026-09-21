#!/bin/bash
# 논문용 등지연 표 + 곡선 — 상한(무분절) / AlignAtt / 디코더 <SEG> 임계값 셋만.
#
# 스트리밍 정책은 `qwenseg_th*` 다: Qwen3-ASR 파인튜닝 디코더에 텍스트만 흘려
# P(<SEG>) >= θ 인 자리에서 바로 자른다. 문장 끝을 안 보므로 실시간에 그대로 쓸 수 있고,
# 노브는 임계값 θ 하나다 (오프라인 `qwenp0`/`qwenlrall` 은 T 로 자르므로 여기 안 쓴다).
#
# AlignAtt 은 **원논문 노브 `f`** 로 쓴다 (`alignatt_mad_f{2,4,6,8}`) — 최근 f 어절에
# 정렬된 토큰은 안 내보낸다. 우리 `T` 를 얹은 `alignatt_T` 판은 안 쓴다: 노브가 두 겹이
# 되어 무슨 축을 쓸었는지 알 수 없다. `_mad_` 는 판정용 내부 NMT 를 평가 번역기와 같은
# madlad 로 맞춘 라벨이라는 뜻이고, 옛 NLLB 산출과 한 곡선에 섞으면 안 된다.
#
# 두 런을 합쳐야 한다 — θ 곡선은 full_qwenseg_lowlat 에, AlignAtt 조건은 거기와
# full_judge13_cmp 양쪽에 있다. 같은 15,530문장·같은 madlad 번역기라 무분절 상한이
# 소수 셋째 자리까지 같다 (아래 assert 가 그걸 검사한다).
#
#   bash core/meaning_segmentator/tools/autoseg_en2x/qwenseg/covost2_paper_table.sh
set -eu
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
export PYTHONPATH=.
PY=.venv/bin/python
A=core/meaning_segmentator/experiment/artifacts/en2x/covost2
RUN=en2x/covost2/full_qwenseg_paper

mkdir -p $A/full_qwenseg_paper/bleu
$PY - <<'PYEOF'
import json, pathlib
A = pathlib.Path("core/meaning_segmentator/experiment/artifacts/en2x/covost2")
for t in ("de", "ja", "zh"):
    lo = json.loads((A / "full_qwenseg_lowlat/bleu" / f"{t}.json").read_text(encoding="utf-8"))
    cmp_ = json.loads((A / "full_judge13_cmp/bleu" / f"{t}.json").read_text(encoding="utf-8"))
    assert lo["n"] == cmp_["n"] and lo["translator"] == cmp_["translator"]
    a, b = lo["conditions"]["unsegmented"], cmp_["conditions"]["unsegmented"]
    assert abs(a["comet"] - b["comet"]) < 2e-4 and abs(a["laal_ms"] - b["laal_ms"]) < 1.0, t
    added = []
    for k, v in cmp_["conditions"].items():
        if k.startswith("alignatt") and k not in lo["conditions"]:
            lo["conditions"][k] = v
            added.append(k)
    (A / "full_qwenseg_paper/bleu" / f"{t}.json").write_text(
        json.dumps(lo, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[{t}] n={lo['n']} AlignAtt 조건 {len(added)}개 옮김")
PYEOF

$PY core/meaning_segmentator/tools/covost2_chain/en2x_table.py \
  --run-id $RUN \
  --row 'alignatt_mad_f:AlignAtt~\cite{papi-2023}' \
  --row 'qwenseg_th:Decoder \texttt{SEG} threshold (Ours; stream)' \
  --outside unsegmented --outside-first

# 지연 구간은 AlignAtt 의 네 점 전부가 들어가게 잡고(`--xlim-cover alignatt_mad`), θ 곡선은
# 그 구간을 덮는 점까지만 그린다(`--xlim-snap`). 양 끝은 점 단위로 끊긴다 — 측정점이
# 없는 자리에서 선이 잘리면 마커 없는 끝이 생겨 그 자리를 측정점으로 읽게 된다.
# 노브 값은 안 적는다 — 맞대는 축은 지연이고 θ·T 는 각 정책의 내부 눈금이다.
$PY -m core.meaning_segmentator.autoseg.scoring.plot_tradeoff \
  --run-id $RUN --targets zh de ja --metric comet --t-grid 2 3 4 6 \
  --drop auto causal_align syntax mu_prefix punct alignatt alignatt_native \
         mu_prefix_mad judge13 qwenp0 qwenlrall \
  --xlim-cover alignatt_mad --xlim-snap \
  --point-labels none --ceiling-in-ylim --solid --wspace 0.34 --font-scale 1.35 \
  --out tradeoff_stream_comet
