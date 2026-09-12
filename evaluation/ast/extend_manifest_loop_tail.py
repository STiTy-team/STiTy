"""loop405 매니페스트에 층화 정렬의 미사용 꼬리를 붙인다 — train 근거를 늘리려고.

autoseg 루프가 Critic 에게 주는 실패 사례는 train 배치에서만 나온다. 배치를 키우려면
문장이 더 필요한데 `loop405` 는 405문장뿐이고, test 100 + dev 215 를 빼면 train 풀이
90 밖에 안 남는다 (배치 30 + 홀드아웃 60).

**그냥 늘리면 안 된다.** `data.stratified_order` 는 층마다 `rng.shuffle` 을 하므로 풀에
문장이 하나만 늘어도 그 층의 순열 전체가 달라진다 — 실측으로 70문장을 붙이면 test 는
100개 중 22개, dev 는 215개 중 98개만 남는다. 그래서 루프 쪽에 `--split-from` (평가
분할 고정)을 함께 넣었고, **이 스크립트로 만든 매니페스트는 그 플래그와 같이 써야
한다.** 안 쓰면 이전 런과 비교가 끊긴다.

재료는 이미 있다. `fleurs_nway_en_clean500_order.json` 이 영어 1405문장의 층화 정렬을
얼려 두었고, `clean500` 이 0~499, `loop405` 가 500~904 를 썼다. **905~1404 의 500문장은
아무도 안 썼다.** 그 구간을 순서대로 붙인다.

붙는 행은 **소스 텍스트만** 있다 (`tgt_text` 없음). 로컬에 FLEURS de/ja/zh TSV 가 없어
n-way 를 못 만드는 것도 이유지만, 없어도 되는 것이기도 하다 — 루프는 `_load_ast_manifest`
가 읽는 `src_text`/`utt_id` 만 쓰고, 정답 번역은 `--target-aware` 의 타깃 프로파일에만
들어간다 (`target_texts` 가 빈 값을 건너뛴다). 붙인 문장은 `--split-from` 하에서 전부
train 으로만 가므로 평가에는 닿지 않는다.

강제정렬(`*_unittimes.json`)도 원본 것을 새 이름으로 복사한다. 붙인 문장에는 항목이
없지만 `data.units_per_sec` 가 없는 문장을 건너뛰므로, 발화 속도는 **원래와 같은 부분
집합에서** 계산돼 `min_gap` 이 움직이지 않는다.

    python evaluation/ast/extend_manifest_loop_tail.py --take 500
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

MANIFESTS = Path(__file__).resolve().parent / "manifests"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", default="fleurs_nway_en-de_multi_loop405",
                   help="늘릴 매니페스트 (확장자 없이)")
    p.add_argument("--order", default="fleurs_nway_en_clean500_order.json",
                   help="얼려 둔 층화 정렬. 여기 뒤쪽에서 미사용 문장을 가져온다")
    p.add_argument("--take", type=int, default=500, help="붙일 문장 수 (미사용분 한도)")
    p.add_argument("--tag", default=None, help="새 태그. 기본은 loop<총문장수>")
    args = p.parse_args()

    base = MANIFESTS / f"{args.base}.jsonl"
    rows = [json.loads(l) for l in base.read_text(encoding="utf-8").splitlines() if l.strip()]
    have = {str(r["utt_id"]) for r in rows}
    src_lang = rows[0].get("src_lang", "en")

    order = json.loads((MANIFESTS / args.order).read_text(encoding="utf-8"))["order"]
    tail = [o for o in order if str(o["id"]) not in have]
    if len(tail) < args.take:
        print(f"미사용 문장이 {len(tail)}개뿐 — --take {args.take} 를 줄여라", file=sys.stderr)
        return 1

    added = [{"utt_id": str(o["id"]), "src_lang": src_lang, "src_text": o["text"],
              # 정답 번역·talk_id·원 스플릿은 안 싣는다. 위 docstring 참조 — 루프가
              # 안 읽고, 빈 값으로 두면 `target_texts` 가 알아서 건너뛴다.
              "source": f"{args.order}[tail]"}
             for o in tail[:args.take]]

    tag = args.tag or f"loop{len(rows) + len(added)}"
    out = MANIFESTS / f"{args.base.rsplit('_loop', 1)[0]}_{tag}.jsonl"
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n"
                           for r in rows + added), encoding="utf-8")

    ut_src = MANIFESTS / f"{args.base}_unittimes.json"
    ut_out = out.with_name(out.stem + "_unittimes.json")
    if ut_src.exists():
        shutil.copyfile(ut_src, ut_out)

    print(f"{out.name}: 원본 {len(rows)} + 추가 {len(added)} = {len(rows) + len(added)}")
    print(f"  추가분은 소스 텍스트만 (tgt_text 없음) — train 전용, `--split-from` 필수")
    if ut_src.exists():
        print(f"  강제정렬 복사: {ut_out.name} (추가분은 항목 없음 → 발화속도 불변)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
