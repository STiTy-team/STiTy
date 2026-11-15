import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import restyle as R  # noqa: E402


def test_only_spoken_sources_into_korean_are_restyled():
    assert R.needs_restyle({"lang": "en", "tgt": "ko", "source": "aihub"})
    assert not R.needs_restyle({"lang": "en", "tgt": "ko", "source": "covost"})
    assert not R.needs_restyle({"lang": "en", "tgt": "ko", "source": "fleurs"})
    assert not R.needs_restyle({"lang": "ko", "tgt": "en", "source": "aihub"})


def test_accept_keeps_piece_count_empties_and_wording():
    old = ["그러니까 그들이 할 일은 추측을 하는 것인데", "", "그 추측을 확인합니다."]
    assert R.accept(old, ["그러니까 그들이 할 일은 추측을 하는 것인데", "", "그 추측을 확인해요."])
    assert not R.accept(old, ["그러니까 그들이 할 일은 추측을 하는 것인데", "그 추측을 확인해요."])  # piece count
    assert not R.accept(old, ["그러니까", "추가", "그 추측을 확인해요."])                              # empty filled
    assert not R.accept(old, ["완전히 다른 문장입니다", "", "그 추측을 확인해요."])                    # wording changed


def test_restyle_record_applies_and_falls_back():
    rec = {"sid": "a", "schedule": [[3, "그것을 해시하고"], [6, ""], [9, "확인합니다."]]}
    out = R.restyle_record(rec, lambda pieces: ["그것을 해시하고", "", "확인해요."])
    assert out["schedule"] == [[3, "그것을 해시하고"], [6, ""], [9, "확인해요."]] and out["restyled"] is True
    bad = R.restyle_record(rec, lambda pieces: ["엉뚱한 답"])
    assert bad["schedule"] == rec["schedule"] and bad["restyled"] is False


def test_parse_reply_reads_a_json_list():
    assert R.parse_reply('```json\n["a", "", "b"]\n```') == ["a", "", "b"]
    assert R.parse_reply("not json") is None


def test_accept_allows_only_words_with_a_formal_ending_to_change():
    assert R.accept(["이건 문제가 됩니다."], ["이건 문제가 돼요."])
    assert R.accept(["중성자는 전하가 없습니다만"], ["중성자는 전하가 없지만"])
    assert not R.accept(["이건 문제가 됩니다."], ["저건 문제가 돼요."])   # a word without a formal ending changed
    assert R.accept(["확인합니다."], ["확인해요."])


def test_needs_restyle_takes_the_source_set():
    allsrc = {"aihub", "covost", "fleurs"}
    assert R.needs_restyle({"lang": "en", "tgt": "ko", "source": "covost"}, allsrc)
    assert not R.needs_restyle({"lang": "en", "tgt": "ko", "source": "covost"})
