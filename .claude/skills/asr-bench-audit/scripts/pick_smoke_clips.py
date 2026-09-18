# -*- coding: utf-8 -*-
"""스모크용 클립을 성질로 고른다.

앞에서 N 개를 자르면 이상치를 놓친다. 20클립 서브셋이 결론을 흔들었던 것도, 저레벨
클립에서 한 백엔드만 통째로 빈 전사를 냈던 것도 같은 이유다. 5클립으로 줄이면 그
위험이 더 커지므로 **무엇을 드러내는 클립인지** 보고 고른다.

    python3 pick_smoke_clips.py --lang ko
    python3 pick_smoke_clips.py --lang en --ids-only

고르는 축 다섯 — 전부 실제로 사고가 났던 자리다.

  숫자    표기 스타일이 갈린다. 아라비아 숫자로 내는 백엔드와 말로 쓰는 백엔드가 있고,
          정규화 없이 채점하면 인식이 맞아도 오답이 된다
  연도    같은 축인데 영어 연도 읽기가 따로 걸린 적이 있다(2001~2009 가 21~29 와 겹침)
  단위    `15m` 과 `15미터` 가 같은 인식인지 보는 자리
  긴 것   꼬리 잘림과 중복 전사가 드러난다. 짧은 클립만 보면 둘 다 안 보인다
  저레벨  입력 레벨 때문에 전사가 통째로 비는 백엔드가 있다

오디오를 읽어야 하는 축(긴 것·저레벨)은 `soundfile` 이 있을 때만 쓴다. 없으면 그 두
축을 건너뛰고 **건너뛴 사실을 출력한다** — 조용히 4개만 고르면 저레벨 함정이 그대로
남는다.
"""
from __future__ import print_function
import argparse
import os
import re
import sys

FLEURS_ROOT = os.path.join(os.path.expanduser("~"), "STiTy-team", "datasets",
                           "fleurs", "data")
LANG_DIR = {"ko": "ko_kr", "en": "en_us"}

HAS_NUMBER = re.compile(r"\d")
HAS_YEAR = re.compile(r"\b(?:1[89]|20)\d{2}\b")
# 약어와 단위어 양쪽을 본다. 한쪽만 보면 그 표기를 쓰는 백엔드만 걸린다.
HAS_UNIT = re.compile(
    r"\d\s*(?:km|cm|mm|kg|m|%)\b"
    r"|미터|킬로미터|센티미터|밀리미터|킬로그램|그램|퍼센트"
    r"|\bpercent\b|\bmet(?:er|re)s?\b|\bkilograms?\b", re.I)


def load_rows(lang, fleurs_root):
    d = os.path.join(fleurs_root, LANG_DIR[lang])
    tsv = os.path.join(d, "test.tsv")
    audio_dir = os.path.join(d, "audio", "test")
    rows = []
    with open(tsv, encoding="utf-8") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 3:
                continue
            wav = os.path.join(audio_dir, p[1])
            if not os.path.exists(wav):
                continue
            rows.append({"file_id": p[1].replace(".wav", ""),
                         "path": wav, "reference": p[2]})
    rows.sort(key=lambda r: r["file_id"])
    return rows


def measure(rows):
    """길이(초)와 피크를 잰다. soundfile 이 없으면 (False, 사유) 를 돌려준다."""
    try:
        import numpy as np
        import soundfile as sf
    except ImportError as exc:
        return False, "soundfile/numpy 없음 (%s)" % exc
    for r in rows:
        try:
            audio, sr = sf.read(r["path"], dtype="float32")
        except Exception as exc:                            # noqa: BLE001
            r["dur"], r["peak"] = None, None
            sys.stderr.write("[pick] 못 읽음 %s: %r\n" % (r["file_id"], exc))
            continue
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        r["dur"] = len(audio) / float(sr)
        r["peak"] = float(np.abs(audio).max()) if len(audio) else 0.0
    return True, ""


def pick(rows, have_audio):
    """축마다 하나씩. 이미 뽑힌 클립은 건너뛰어 다섯이 서로 다른 것을 드러내게 한다."""
    chosen, seen = [], set()

    def take(axis, why, candidates):
        for r in candidates:
            if r["file_id"] in seen:
                continue
            seen.add(r["file_id"])
            chosen.append((axis, why, r))
            return True
        return False

    missed = []
    # 짧은 것부터 본다 — 스모크는 싸야 한다. 길이 축만 예외로 가장 긴 것을 고른다.
    by_len = sorted(rows, key=lambda r: len(r["reference"]))

    if not take("숫자", "참조에 아라비아 숫자가 있어 표기 스타일이 드러난다",
                [r for r in by_len if HAS_NUMBER.search(r["reference"])
                 and not HAS_YEAR.search(r["reference"])]):
        missed.append("숫자")
    if not take("연도", "연도 읽기가 다른 수와 겹치는지 본다",
                [r for r in by_len if HAS_YEAR.search(r["reference"])]):
        missed.append("연도")
    if not take("단위", "약어와 단위어가 같은 인식으로 채점되는지 본다",
                [r for r in by_len if HAS_UNIT.search(r["reference"])]):
        missed.append("단위")

    if have_audio:
        ok = [r for r in rows if r.get("dur") is not None]
        if not take("긴 것", "꼬리 잘림과 중복 전사가 드러난다",
                    sorted(ok, key=lambda r: -r["dur"])):
            missed.append("긴 것")
        if not take("저레벨", "입력 레벨 때문에 전사가 비는 백엔드가 있다",
                    sorted([r for r in ok if r.get("peak") is not None],
                           key=lambda r: r["peak"])):
            missed.append("저레벨")
    else:
        missed += ["긴 것", "저레벨"]
    return chosen, missed


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--lang", choices=sorted(LANG_DIR), default="ko")
    ap.add_argument("--fleurs-root", default=FLEURS_ROOT)
    ap.add_argument("--ids-only", action="store_true",
                    help="file_id 만 쉼표로 출력한다 (smoke_client --file-ids 에 그대로 넘긴다)")
    a = ap.parse_args()

    rows = load_rows(a.lang, a.fleurs_root)
    if not rows:
        sys.stderr.write("클립이 없다: %s\n" % a.fleurs_root)
        return 2

    have_audio, why = measure(rows)
    chosen, missed = pick(rows, have_audio)

    if a.ids_only:
        print(",".join(r["file_id"] for _, _, r in chosen))
    else:
        print("%-7s %-34s %s" % ("축", "file_id", "왜 이 클립인가"))
        for axis, reason, r in chosen:
            print("%-7s %-34s %s" % (axis, r["file_id"], reason))
            ref = r["reference"]
            print("%-7s %-34s %s" % ("", "", ref[:70] + ("..." if len(ref) > 70 else "")))
        print()
        print("--file-ids %s" % ",".join(r["file_id"] for _, _, r in chosen))

    if missed:
        sys.stderr.write("\n[pick] 못 채운 축: %s\n" % ", ".join(missed))
        if not have_audio:
            sys.stderr.write("[pick] 오디오를 못 읽어 길이·레벨 축을 건너뛰었다 — %s\n" % why)
        sys.stderr.write("[pick] 그 축의 함정은 이 스모크로 안 걸린다. 감사 표에 적을 것.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
