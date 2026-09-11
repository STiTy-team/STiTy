# -*- coding: utf-8 -*-
"""이미 나와 있는 결과 JSON 을 표기 정규화 기준으로 다시 채점해 본다.

새 런은 `smoke_client` 가 채점 시점에 정규화해 `avg_<unit>`(정규화 후)와
`avg_<unit>_raw`(원값)를 함께 저장하므로 이 도구가 필요 없다. 이건 **그 전에
나온 JSON** 을 같은 기준으로 환산해 보는 용도다. 파일을 고치지 않고 출력만 한다.

    python3 itn_rescore.py <결과JSON 또는 디렉터리> [...]
"""
from __future__ import print_function
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from text_norm import score, score_pair          # noqa: E402


def collect(paths):
    out = []
    for p in paths:
        if os.path.isdir(p):
            for root, _, files in os.walk(p):
                for f in sorted(files):
                    if f.endswith(".json"):
                        out.append(os.path.join(root, f))
        else:
            out.append(p)
    return out


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if not args:
        print(__doc__)
        return 2
    print("%-22s %-5s %9s %10s %9s  %s"
          % ("백엔드", "단위", "원래", "정규화후", "차이", "바뀐 클립"))
    for path in collect(args):
        try:
            d = json.load(open(path))
        except Exception:                                   # noqa: BLE001
            continue
        if not isinstance(d, dict) or "rows" not in d:
            continue
        s = d.get("summary", {})
        lang = s.get("lang", "ko")
        unit = s.get("unit", "cer" if lang == "ko" else "wer")
        base, fixed, touched = [], [], 0
        for r in d["rows"]:
            ref, hyp = r.get("reference"), r.get("transcript")
            if ref is None:
                continue
            n, raw = score_pair(ref, hyp, unit, lang)
            if n is None:
                continue
            base.append(raw)
            fixed.append(n)
            if abs(n - raw) > 1e-12:
                touched += 1
        if not base:
            continue
        b = sum(base) / len(base)
        f = sum(fixed) / len(fixed)
        print("%-22s %-5s %9.4f %10.4f %+9.4f  %d/%d"
              % (s.get("backend", os.path.basename(path)), unit, b, f, f - b,
                 touched, len(base)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
