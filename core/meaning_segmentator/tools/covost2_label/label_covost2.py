"""CoVoST2 에 autoseg 프롬프트로 <SEG:n> 라벨을 단다. **라벨링 전용** — 번역·채점 안 한다.

루프의 분절 경로(`pipeline.segment_batch`)를 그대로 쓴다. 정규화·검증·재시도가 런과
동일해야 라벨이 같은 자로 나온다.

**산출물은 레포 안에 쓴다.** 종전에 `/tmp` 스크래치패드에 두었다가 머신 재부팅으로
3모델 비교 결과를 통째로 잃었다 (2026-08-31). 중간 결과를 볼 수 있도록 캐시 저장
임계치도 낮춘다.
"""
from __future__ import annotations

import argparse, json, re, sys, threading, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from core.meaning_segmentator.autoseg.infra import gateway
from core.meaning_segmentator.autoseg.infra.gateway import Gateway
from core.meaning_segmentator.autoseg.runtime import agents_distill as ad
from core.meaning_segmentator.autoseg.runtime import labels as L
from core.meaning_segmentator.autoseg.runtime import pipeline as P
from core.meaning_segmentator.autoseg.loop import target_is_spaced

SEG = re.compile(r"<SEG(?::\d+)?>")

# 프롬프트 문면에 박힌 숫자. `agents._COVERAGE_RULE` / `_GAP_RULE` 이 찍어 넣는 형식이다.
_PROMPT_COVERAGE = re.compile(r"AT LEAST one boundary per (\d+) (words|characters)")
_PROMPT_GAP = re.compile(r"Leave AT LEAST (\d+) (words|characters) between")


def prompt_numbers(prompt: str) -> tuple[int | None, int | None]:
    """프롬프트가 모델에게 지시한 `(t_floor, min_gap)`. 없으면 None.

    **이 숫자는 프롬프트를 만들 때 문면에 굳는다** (`agents.output_rules`). 라벨링은
    프롬프트 파일을 그대로 읽어 보내므로 `--min-gap`/`--t-floor` 를 바꿔도 문면은
    안 바뀐다 — 두 경로가 분리돼 있다.
    """
    c = _PROMPT_COVERAGE.search(prompt)
    g = _PROMPT_GAP.search(prompt)
    return (int(c.group(1)) if c else None), (int(g.group(1)) if g else None)


def strip_seg(s: str, spaced: bool = True) -> str:
    """태그를 뗀 본문. **띄어쓰기 없는 언어는 공백으로 치환하면 안 된다.**

    종전에는 `SEG.sub(" ", s)` 로 태그 자리에 공백을 넣었다. 영어에서는 태그 양옆이
    이미 공백이라 무해하지만, 일본어·중국어에서는 없던 공백이 생겨 원문과 달라진다.
    그래서 `text_preserved` 가 실제로는 멀쩡한 결과를 실패로 셌다 (실측: zh 8/12,
    ja 10/12 가 거짓 실패 — 같은 결과를 `spaced=False` 로 재면 12/12 일치).
    검증기(`P.validate`)는 `spaced` 를 알고 있어서 위반 0 을 냈으므로, 지표만
    혼자 틀린 상태였다.
    """
    return P.strip_tags(s, spaced=spaced)


def unwrap(t: str) -> str:
    """문장 전체를 감싼 큰따옴표를 벗긴다.

    CoVoST2 원문 300건 중 **79건(26%)이 양끝 따옴표로 감싸여** 있는데, 분절기가 그것을
    떼고 출력해 검증기가 전부 `text_modified` 로 잡았다 (실측: 원문보존 26~31/60 이
    따옴표를 무시하면 44~46/60 으로 회복). 모델 품질이 아니라 표기 아티팩트이고, 실제
    ASR 출력에는 이 따옴표가 없으므로 도메인상으로도 벗기는 쪽이 맞다.

    **감싼 경우만 벗긴다** — 실측 79건 전부 양끝이 따옴표였고 내부에만 있는 문장은 0건.
    """
    t = t.strip()
    if len(t) > 1 and t[0] == '"' and t[-1] == '"' and t.count('"') == 2:
        return t[1:-1].strip()
    return t


class LiveCache(P.JsonCache):
    """`JsonCache` 는 20건마다 저장한다. 긴 런의 중간 결과를 못 봐서 임계치를 낮춘다.

    사고 모델은 콜당 9분이 걸려 60문장에 1시간이 넘는데, 그동안 산출물이 하나도 안
    보이면 진행 상황을 판단할 수 없다 (실측: 46분 경과 시점에 캐시 파일이 아직 없었다).
    """

    def __init__(self, path: Path, every: int = 1):
        super().__init__(path)
        self._every = max(1, every)

    def put(self, k: str, v) -> None:
        with self._lock:
            self._data[k] = v
            self._dirty += 1
            if self._dirty >= self._every:
                self._flush_locked()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--manifest",
                    default="evaluation/ast/manifests/covost2_en-de_full.jsonl")
    ap.add_argument("--out", required=True, help="라벨 jsonl (레포 안 경로 권장)")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--model", default="gpt-5-mini")
    ap.add_argument("--allow-prompt-mismatch", action="store_true",
                    help="프롬프트 문면의 숫자와 --min-gap/--t-floor 가 달라도 강행")
    ap.add_argument("--spaced", default=None, choices=["yes", "no"],
                    help="소스가 띄어쓰기 언어인가. 기본은 매니페스트의 src_lang 으로 판정")
    ap.add_argument("--min-gap", type=int, default=3)
    ap.add_argument("--t-floor", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=6)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--budget", type=float, default=100000.0)
    ap.add_argument("--seg-reasoning-effort", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--cache-every", type=int, default=1,
                    help="캐시 저장 주기(건). 1이면 매 건 저장 — 중간 결과를 즉시 본다")
    # **분절 출력 상한.** 기본 SEG_MAX_TOKENS 는 32768 인데 그 값은 thinking 모델의
    # 사고 토큰까지 감당하려고 잡힌 것이다. 비사고 모델에는 과하고(폭주한 콜 하나가
    # 3.1 tok/s 로 2.9시간을 먹는다), 사고 모델에는 이 값이 필요하다.
    ap.add_argument("--max-tokens", type=int, default=2048)
    # **사고 모델은 타임아웃이 예산과 함께 움직여야 한다.** Gateway 기본 900초인데
    # 16,000 토큰을 10.8 tok/s 로 쓰면 콜 하나가 최대 24분이다. 실측에서 15분에 잘리고
    # 재시도 5회가 전부 같은 이유로 잘려 런이 통째로 죽었다 (2026-08-31, 30문장 런).
    ap.add_argument("--timeout", type=float, default=900.0)
    gateway.add_provider_args(ap)
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.manifest, encoding="utf-8") if l.strip()]

    # **띄어쓰기 여부는 소스 언어가 정한다** — 종전에는 `SPACED = True` 로 박혀 있었다.
    # 영어 소스(en→X)만 돌리던 시절에는 맞았지만, X→en 을 돌리면 조용히 틀린다:
    # 일본어·중국어는 공백이 거의 없어 `coverage_need` 가 문장 전체를 1단위로 세고,
    # 요구 경계 수가 0 이 되어 **아무 경계도 안 찍힌 결과가 전부 통과**한다.
    #
    # `target_is_spaced` 는 `zh-CN` 같은 지역 접미사를 모르므로(`zh-cn` 이 목록에
    # 없어 True 를 돌려준다) 앞부분만 떼어 넘긴다.
    if a.spaced is None:
        codes = {(r.get("src_lang") or "").split("-")[0].lower() for r in rows}
        if len(codes) != 1:
            print(f"매니페스트에 소스 언어가 섞여 있습니다: {sorted(codes)}. "
                  f"--spaced 로 직접 지정하세요", file=sys.stderr)
            return 2
        SPACED = target_is_spaced(codes.pop())
    else:
        SPACED = a.spaced == "yes"
    if a.limit:
        rows = rows[: a.limit]
    src_codes = sorted({(r.get("src_lang") or "?") for r in rows})
    print(f"[label] 소스 {src_codes} / spaced={SPACED} / min_gap={a.min_gap} "
          f"t_floor={a.t_floor} / n={len(rows)}", flush=True)
    for r in rows:
        r["src_text_raw"] = r["src_text"]
        r["src_text"] = unwrap(r["src_text"])
    # **후보 경계를 코드가 박아 준다.** 판정 프롬프트의 `[Output Rules]` 첫 줄이 "입력에 이미
    # 모든 후보 자리에 `<SEG:?>` 가 있다. 모든 마커를 유지하고 ? 를 0~100 정수로 바꿔라" 다.
    # 그런데 여기서는 맨 문장을 넣고 있었다 — 모델이 자리까지 발명해야 했고, 실측으로 문장
    # 넷 중 하나가 후보를 빼먹었다(covost2 15,530 중 24.2%). 1차 통과도 0.84/0.57 로 낮아
    # 재시도가 호출의 45% 를 먹었다. 루프(`loop_judge` → `loop_distill.evaluate`)와 **같은
    # 함수**로 박는다: 단위 나누기 → 후보 자리 → 마킹.
    units = [L.units_of(t, SPACED) for t in [r["src_text"] for r in rows]]
    cands = [ad.candidate_positions(len(u), a.min_gap) for u in units]
    skip = [i for i, c in enumerate(cands) if not c]
    if skip:
        print(f"[label] 후보 자리가 없는 문장 {len(skip)}개는 분절하지 않는다 "
              f"(단위 {min(len(units[i]) for i in skip)}~{max(len(units[i]) for i in skip)}개)",
              flush=True)
    keep = [i for i, c in enumerate(cands) if c]
    rows = [rows[i] for i in keep]
    texts = [ad.mark_candidates(units[i], cands[i]) for i in keep]
    prompt = Path(a.prompt).read_text(encoding="utf-8")

    # **문면과 인자가 어긋나면 시작하지 않는다.** 어긋난 채로 돌면 모델은 문면대로
    # 성기게 찍는데 코드는 더 빡빡하게 요구해, 전 문장이 too_few_tags 로 재시도되고
    # 모델이 개수를 맞추려다 안전하지 않은 자리까지 찍는다.
    #
    # 실측(CoVoST2 zh, 문면 8/6 vs 인자 7/5): too_few_tags 95건, 콜 900회(예상 500),
    # 비용 $13.17(예상 $10.07), 그리고 **단어 내부 절단** — 검증기가 글자 단위라
    # 못 잡는 종류의 손상이다. 경고로는 부족해서 기본을 중단으로 둔다.
    p_floor, p_gap = prompt_numbers(prompt)
    bad = []
    if p_floor is not None and p_floor != a.t_floor:
        bad.append(f"커버리지: 문면 {p_floor} vs --t-floor {a.t_floor}")
    if p_gap is not None and p_gap != a.min_gap:
        bad.append(f"간격: 문면 {p_gap} vs --min-gap {a.min_gap}")
    if bad:
        msg = ("프롬프트 문면의 숫자와 인자가 다릅니다 — " + " / ".join(bad)
               + f"\n  프롬프트: {a.prompt}"
               + "\n  프롬프트의 [Output Rules] 를 인자에 맞춰 다시 만들거나 "
                 "(core.meaning_segmentator.autoseg.runtime.agents.output_rules), "
                 "인자를 문면에 맞추세요."
                 "\n  의도한 불일치라면 --allow-prompt-mismatch 를 주세요.")
        if not a.allow_prompt_mismatch:
            print(msg, file=sys.stderr)
            return 2
        print(f"[label] 경고(무시함): {msg}", flush=True)

    # 검증·정규화도 **점수 형식**의 것으로 바꾼다. `P.validate` 는 모델이 절단 자리를 고르는
    # 형식(`<SEG>` 몇 개)을 검사하고 `coverage_need` 로 최소 개수를 요구하는데, 점수 형식에서는
    # 자리가 이미 정해져 있어 그 요구가 뜻이 없다 — 봐야 할 것은 "마커를 하나도 잃지 않았고
    # 전부 정수로 채웠는가" 다. 그게 `ad.validate_scored` 이고 루프가 쓰는 것과 같다.
    need_fn = lambda t: None
    validate_fn = lambda t, out: ad.validate_scored("", t, out, SPACED)
    normalize_fn = ad.normalize_scored

    def cost_ticker(gw, every: float = 120.0):
        """누적 비용을 주기적으로 로그에 찍는다.

        요약 JSON 은 **끝날 때 한 번**만 나온다. 4시간짜리 런이 3시간째 죽으면 그때까지
        쓴 돈이 로그에서 통째로 사라진다 (.claude/rules/cost-watch.md). 캐시로 역산은
        되지만 로그가 1차 기록이어야 한다.
        """
        stop = threading.Event()

        def loop():
            while not stop.wait(every):
                u = gw.usage.snapshot()
                per = gateway.by_key_line(u, gw.key_names)
                print(f"[label] 진행 calls={u['calls']} "
                      f"completion={u['completion_tokens']} "
                      f"누적 비용 ${u['cost']:.4f}"
                      + (f" | 429 {gw.rate_limited}회" if gw.rate_limited else "")
                      + (f" | 키별 {per}" if per else ""), flush=True)

        threading.Thread(target=loop, daemon=True).start()
        return stop

    P.SEG_MAX_TOKENS = a.max_tokens
    gw = Gateway.from_args(a, model=a.model, budget=a.budget, timeout=a.timeout,
                           max_connections=max(16, a.workers))
    if len(gw._keys) > 1:
        print(f"[gateway] 키 {len(gw._keys)}개 라운드로빈 / 동시 연결 {max(16, a.workers)}",
              flush=True)
    cache = LiveCache(Path(a.cache), every=a.cache_every)
    first: list = []
    t0 = time.time()
    ticker = cost_ticker(gw)
    outs, ok = P.segment_batch(
        gw, prompt, texts, cache=cache, workers=a.workers,
        validate_fn=validate_fn, normalize_fn=normalize_fn,
        # **태그만 옮겨 고칠 수 있으면 다시 부르지 않는다.** 위반이 `text_modified` 하나면
        # 태그를 원문 어절 경계로 옮겨 본다 — 모델이 따옴표·철자를 바꾼 것뿐인데 LLM 을 다시
        # 부를 이유가 없다. v0 실측에서 재시도가 호출의 45%였다. 안전장치는 `realign_tags`
        # 안에 있다: 바뀐 어절이 max(2, 20%) 를 넘거나 태그가 바뀐 구간에 떨어지면 포기하고
        # 종전대로 재시도한다(그 경로에서 오탐 0 / 36건 전부 검증 통과).
        realign_fn=lambda t, o: ad.realign_tags(t, o, SPACED),
        reasoning_effort=a.seg_reasoning_effort, batch_size=a.batch_size,
        first_pass_sink=first, need_fn=need_fn)
    ticker.set()
    dt = time.time() - t0

    n_ok = n_pres = 0
    out_rows = []
    # **검증에는 마커가 박힌 입력을 준다.** `validate_scored` 는 "입력 마커를 하나도 잃지
    # 않았는가" 를 보므로 맨 문장을 주면 기준이 없다. 원문 보존은 그대로 맨 문장과 견준다.
    for r, mk, o, first_ok in zip(rows, texts, outs, ok):
        viol = validate_fn(mk, o)
        pres = strip_seg(o, SPACED) == strip_seg(r["src_text"], SPACED)
        n_pres += pres
        n_ok += not viol
        out_rows.append({**r, "seg_text": o, "marked": mk,
                         "n_boundaries": len(SEG.findall(o)),
                         # 점수 형식에서 "요구" 는 **코드가 박은 마커 수**다. 종전의
                         # `coverage_need`(최소 몇 개를 잘라야 한다)는 모델이 자리를 고르는
                         # 형식의 값이라 여기서는 뜻이 없다. 이 값과 `n_boundaries` 가 같아야
                         # 후보를 하나도 안 잃은 것이다.
                         "required": mk.count("<SEG:?>"),
                         "first_pass_ok": bool(first_ok),
                         "text_preserved": pres,
                         "violations": [getattr(v, "rule", str(v)) for v in viol]})
    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # **1차 위반을 파일로 남긴다.** 종전에는 `first_pass` 비율만 요약에 찍혀서 "재시도가 비용의
    # 45%" 는 알아도 그중 태그만 옮겨 고칠 수 있는 몫이 얼마인지 셀 수 없었다. 규칙별 분포와
    # 재정렬 성공 여부가 여기 남으면 다음 런에서 바로 계산된다.
    (out_path.parent / f"{out_path.stem}.first_pass.json").write_text(
        json.dumps({"n": len(rows), "n_violations": len(first),
                    "realigned": sum(1 for v in first if v.get("realigned")),
                    "by_rule": {r: sum(1 for v in first if v["rule"] == r)
                                for r in sorted({v["rule"] for v in first})},
                    "samples": first[:20]}, ensure_ascii=False, indent=1), encoding="utf-8")
    with out_path.open("w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    u = gw.usage.snapshot()
    per = gateway.by_key_line(u, gw.key_names)
    if per:
        print(f"[label] 키별 최종 지출 {per}", flush=True)
    nb = [r["n_boundaries"] for r in out_rows]
    rq = [r["required"] for r in out_rows]
    n = len(out_rows)
    print(json.dumps({
        "model": a.model, "prompt": a.prompt, "n": n, "wall_sec": round(dt, 1),
        "format_pass": round(n_ok / n, 3),
        "first_pass": round(sum(ok) / len(ok), 3),
        "text_preserved": round(n_pres / n, 3),
        "mean_boundaries": round(sum(nb) / n, 2),
        "mean_required": round(sum(rq) / n, 2),
        "coverage_met": round(sum(1 for r in out_rows
                                  if r["n_boundaries"] >= r["required"]) / n, 3),
        "calls": u["calls"], "completion_tokens": u["completion_tokens"],
        "cost": round(u["cost"], 4),
    }, ensure_ascii=False, indent=1))
    print("LABELDONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
