#!/usr/bin/env python3
"""ACL 60/60 장문 — LongYAAL(OmniSTEval, SoftSegmenter) 채점. StreamLAAL 과 나란히 낸다.

    .venv-streamlaal/bin/python evaluation/ast/longyaal_acl6060.py \\
        --tag voxmix_t91c320_gate_20260921_173311 --split eval

왜 따로 재나
------------
StreamLAAL 은 문장 하나의 LAAL 을 낼 때 소스 끝(문장 끝)을 넘긴 단어를 **첫 하나만 세고
멈춘다**(`if d >= source_length: break`). 문장을 통째로 문장 끝에서 커밋하는 정책은 단어
평균이 아니라 `delays[0]` 하나, 곧 문장 길이 전체가 값이 된다. 고정 청크 정책은 청크 안
단어가 0~N 초를 고르게 기다려 N/2 가 나온다. 두 정책이 다른 규칙으로 재지는 셈이다.

LongYAAL(arXiv 2509.17349) 은 문장 경계 너머 단어를 전부 넣고 **녹음 전체 끝** 이후만 뺀다.
재분절도 mwerSegmenter 대신 SoftSegmenter(문자 집합 유사도 정렬 + 미래 문장 금지)다.

지표 계산은 **OmniSTEval 에 위임**한다(`pip install OmniSTEval`, `.venv-streamlaal` 에 있다).
이 파일이 하는 일은 우리 커밋 스트림을 그쪽 입력으로 옮기는 것뿐이다:

    hyp.jsonl      {"source": <wav 이름>, "prediction": 이어붙인 번역,
                    "delays": 단위별 ms, "elapsed": 단위별 ms, "source_length": 녹음 길이 ms}
    segments.yaml  참조 문장마다 {wav, offset, duration}  (gold 타임스탬프)
    refs.txt       참조 번역, 한 줄 한 문장
    srcs.txt       gold 영어 전사, 한 줄 한 문장

단위 규칙은 StreamLAAL 쪽과 같게 맞춘다. 커밋을 이어붙이고 지연을 펼치는 건
`streamlaal_adapter.build_output_with_delays` 를 그대로 쓴다. OmniSTEval 은 word 면
`prediction.split()`, char 면 `list(prediction)` 로 단위를 세므로, word 언어는 커밋 텍스트의
공백을 먼저 정규화해 두 쪽의 단위 수가 어긋나지 않게 한다.

delays 는 녹음 시작 기준 절대 시각이다. 우리 talk 은 offset 0 이므로 `--offset_delays` 없이
segments.yaml 의 gold offset 과 바로 맞는다.

산출물: `results/ACL6060/longyaal_{split}_{tag}/{axis}-{lang}/` (OmniSTEval 원본 출력)
       `results/ACL6060/longyaal_{split}_{tag}.json` (요약 + StreamLAAL 병기)
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from score_acl6060 import (  # noqa: E402
    DELAY_FIELDS, LANG_UNIT, BLEU_TOK, commits_from_row, load_manifest)
from streamlaal_adapter import Commit, Diagnostics, build_output_with_delays  # noqa: E402

ORDER = ["static-c3", "static-c4", "static-c5", "static-c6", "seg-c0.5", "seg-c1",
         "static-c10", "static-c12", "punct-c1", "punct"]


def write_inputs(run_dir: Path, manifest: Dict[str, dict], lang: str,
                 delay_field: str, out: Path) -> dict:
    unit = LANG_UNIT[lang]
    metric = json.loads((run_dir / "metric.json").read_text(encoding="utf-8"))
    out.mkdir(parents=True, exist_ok=True)

    hyp_lines: List[str] = []
    seg_entries: List[str] = []
    refs: List[str] = []
    srcs: List[str] = []
    diag = Diagnostics()
    n_ref = 0
    # 발표 순서는 manifest 를 따른다. OmniSTEval 은 segmentation 의 wav 순서로 가설을 찾는다.
    rows = {r["utt_id"]: r for r in metric["rows"]}
    for utt_id, item in manifest.items():
        row = rows[utt_id]
        wav = os.path.basename(item["wav"])
        commits = commits_from_row(row, delay_field)
        if unit == "word":
            # OmniSTEval 은 split() 으로 센다. 탭·개행·연속 공백을 한 칸으로 접어 둔다.
            commits = [Commit(text=" ".join(c.text.split()), ideal_delay=c.ideal_delay,
                              ca_delay=c.ca_delay) for c in commits]
        d = Diagnostics()
        owd = build_output_with_delays(commits, unit, d)
        diag.n_commits += d.n_commits
        diag.n_units += d.n_units
        diag.n_empty_commits_dropped += d.n_empty_commits_dropped
        units = list(owd.final_text) if unit == "char" else owd.final_text.split()
        if len(units) != len(owd.ideal_delays):
            raise AssertionError(
                f"{utt_id}: OmniSTEval 단위 수 {len(units)} vs 지연 {len(owd.ideal_delays)}")
        hyp_lines.append(json.dumps({
            "source": wav,
            "prediction": owd.final_text,
            "delays": [round(x * 1000.0, 1) for x in owd.ideal_delays],
            "elapsed": [round(x * 1000.0, 1) for x in owd.computational_aware_delays],
            "source_length": round(float(item["duration"]) * 1000.0, 1),
        }, ensure_ascii=False))
        for s in item["sentences"]:
            seg_entries.append(json.dumps({"wav": wav, "offset": float(s["offset"]),
                                           "duration": float(s["duration"])}))
            refs.append(" ".join(s["tgt"].split()))
            srcs.append(" ".join(s["src"].split()))
            n_ref += 1

    (out / "hyp.jsonl").write_text("\n".join(hyp_lines) + "\n", encoding="utf-8")
    # YAML 은 JSON 의 상위집합이다. 한 줄 한 항목 리스트로 쓴다.
    (out / "segments.yaml").write_text(
        "[\n" + ",\n".join(seg_entries) + "\n]\n", encoding="utf-8")
    (out / "refs.txt").write_text("\n".join(refs) + "\n", encoding="utf-8")
    (out / "srcs.txt").write_text("\n".join(srcs) + "\n", encoding="utf-8")
    return {"n_talks": len(hyp_lines), "n_ref_sentences": n_ref,
            "n_commits": diag.n_commits, "n_units": diag.n_units,
            "n_empty_commits_dropped": diag.n_empty_commits_dropped}


def run_omnisteval(work: Path, lang: str) -> Dict[str, float]:
    exe = Path(sys.executable).parent / "omnisteval"
    cmd = [str(exe), "longform",
           "--speech_segmentation", str(work / "segments.yaml"),
           "--ref_sentences_file", str(work / "refs.txt"),
           "--hypothesis_file", str(work / "hyp.jsonl"),
           "--hypothesis_format", "jsonl",
           "--bleu_tokenizer", BLEU_TOK[lang],
           "--output_folder", str(work / "omnisteval")]
    if LANG_UNIT[lang] == "char":
        cmd.append("--char_level")          # ja/zh: --lang 없이 문자 단위
    else:
        cmd += ["--lang", lang, "--word_level"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    (work / "omnisteval.log").write_text(r.stdout + "\n--- stderr ---\n" + r.stderr,
                                         encoding="utf-8")
    if r.returncode != 0:
        raise SystemExit(f"!! omnisteval 실패 ({work}):\n{r.stderr[-2000:]}")
    scores: Dict[str, float] = {}
    for line in (work / "omnisteval" / "scores.tsv").read_text(encoding="utf-8").splitlines()[1:]:
        k, v = line.split("\t")
        try:
            scores[k] = float(v)
        except ValueError:
            scores[k] = v
    return scores


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-root", default=str(HERE / "results" / "ACL6060"))
    p.add_argument("--manifest-dir", default=str(HERE / "manifests"))
    p.add_argument("--tag", required=True)
    p.add_argument("--split", default="eval")
    p.add_argument("--axes", nargs="+", default=None)
    p.add_argument("--langs", nargs="+", default=["de", "ja", "zh"])
    p.add_argument("--delay-field", default="dispatch_audio_sec", choices=list(DELAY_FIELDS))
    p.add_argument("--out", default=None)
    a = p.parse_args()

    root = Path(a.results_root).expanduser().resolve()
    axes = a.axes or sorted(d.name for d in root.iterdir() if d.is_dir()
                            and any((d / f"{a.split}-{lg}" / a.tag).exists() for lg in a.langs))
    axes = [x for x in ORDER if x in axes] + [x for x in axes if x not in ORDER]
    workroot = root / f"longyaal_{a.split}_{a.tag}"

    # StreamLAAL 병기 — 같은 tag 로 score_acl6060.py 가 남긴 값
    sl_path = root / f"streamlaal_{a.split}_{a.tag}.json"
    stream = {}
    if sl_path.exists():
        for r in json.loads(sl_path.read_text(encoding="utf-8"))["results"]:
            stream[(r["axis"], r["lang"])] = r
    else:
        print(f"[주의] StreamLAAL 결과 없음: {sl_path} — 병기 생략")

    results = []
    for axis in axes:
        for lang in a.langs:
            run_dir = root / axis / f"{a.split}-{lang}" / a.tag
            if not (run_dir / "metric.json").exists():
                print(f"[건너뜀] {run_dir}")
                continue
            manifest = load_manifest(Path(a.manifest_dir) / f"acl6060_{a.split}_en-{lang}.jsonl")
            work = workroot / f"{axis}-{lang}"
            info = write_inputs(run_dir, manifest, lang, a.delay_field, work)
            print(f"[{axis}/{lang}] 입력 {info} → omnisteval ...", flush=True)
            sc = run_omnisteval(work, lang)
            s = stream.get((axis, lang), {})
            rec = {
                "axis": axis, "lang": lang, "latency_unit": LANG_UNIT[lang],
                "delay_field": a.delay_field,
                "long_yaal_sec": round(sc.get("LongYAAL (CU)", float("nan")) / 1000.0, 4),
                "long_yaal_ca_sec": round(sc.get("LongYAAL (CA)", float("nan")) / 1000.0, 4),
                "stream_laal_sec": s.get("stream_laal_sec"),
                "stream_laal_ca_sec": s.get("stream_laal_ca_sec"),
                "bleu_softseg": sc.get("BLEU"),
                "chrf_softseg": sc.get("chrF"),
                "bleu_mwerseg": s.get("bleu"),
                "omnisteval_scores": sc,
                **info,
            }
            results.append(rec)
            print(f"[{axis}/{lang}] LongYAAL {rec['long_yaal_sec']:.3f}s  "
                  f"(CA {rec['long_yaal_ca_sec']:.3f})  StreamLAAL {rec['stream_laal_sec']}  "
                  f"BLEU soft {rec['bleu_softseg']} / mwer {rec['bleu_mwerseg']}", flush=True)

    out_path = Path(a.out) if a.out else root / f"longyaal_{a.split}_{a.tag}.json"
    out_path.write_text(json.dumps({
        "tag": a.tag, "split": a.split, "delay_field": a.delay_field,
        "omnisteval_version": "0.1.10", "results": results,
    }, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"\n=== LongYAAL vs StreamLAAL (NCA, 초) — {a.split} {a.tag} ===")
    print(f"{'axis':11s} " + "  ".join(f"{lg:>15s}" for lg in a.langs))
    print(f"{'':11s} " + "  ".join(f"{'LongYAAL/Stream':>15s}" for _ in a.langs))
    for axis in axes:
        cells = []
        for lang in a.langs:
            r = next((x for x in results if x["axis"] == axis and x["lang"] == lang), None)
            if r is None:
                cells.append(f"{'-':>15s}"); continue
            sl = r["stream_laal_sec"]
            cells.append(f"{r['long_yaal_sec']:6.3f} / {sl if sl is None else f'{sl:6.3f}':>6}")
        print(f"{axis:11s} " + "  ".join(f"{c:>15s}" for c in cells))
    print(f"\n저장: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
