import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import auto_go  # noqa: E402


def _args(**kw):
    d = dict(min_kept=2, min_kept_share=0.5, dump_enko=0.05, dump_other=0.10, empty_ratio=2.0, first_after=6)
    d.update(kw)
    return type("A", (), d)()


def _row(lang, tgt, src, sch):
    return {"sid": f"{lang}{len(src)}", "lang": lang, "tgt": tgt, "src": src, "schedule": sch}


def test_pass():
    rows = [_row("en", "ko", "a b c d e f", [[2, "x"], [4, "y"], [6, "z"]]),
            _row("ko", "en", "가 나 다 라 마 바", [[2, "x"], [4, "y"], [6, "z"]])]
    _, reasons = auto_go.check(rows, 2, _args())
    assert reasons == []


def test_dump_and_kept_and_ratio():
    rows = [_row("en", "ko", "a b c d e f", [[6, "x y z"]]),
            _row("ko", "en", "가 나 다 라 마 바", [[1, "x"], [2, "y"], [3, "z"], [4, "w"], [5, "v"], [6, "u"]])]
    _, reasons = auto_go.check(rows, 10, _args())
    joined = " ".join(reasons)
    assert "en-ko dump_share 1.000 > 0.05" in joined
    assert "50% of labeled 10" in joined
    assert "empty_share ratio" in joined
