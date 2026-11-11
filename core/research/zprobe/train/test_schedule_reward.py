import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import schedule_reward as s  # noqa: E402


def test_grid_matches_rollout():
    assert s.schedule_grid(2.3, 1.0, 0.5) == [1.0, 1.5, 2.0, 2.3]
    assert s.schedule_grid(0.7, 1.0, 0.5) == [0.7]


def test_step_labels_due_wait_lag_and_tail():
    grid = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    # items due at 0.6 (before first step), 1.9, and 3.4; real audio ends at 3.0, tail to 4.0
    labels = s.step_labels([0.6, 1.9, 3.4], 3.0, grid)
    assert labels == ["due", None, "due", None, "due", "wait", "wait"]


def test_step_labels_item_after_audio_end_lands_on_last_step():
    grid = [1.0, 1.5, 2.0]
    assert s.step_labels([2.15], 2.0, grid) == ["wait", "wait", "due"]


def test_agreement():
    labels = ["due", None, "due", "wait", "wait"]
    assert s.agreement(labels, [True, True, False, False, True]) == (0.5, 0.5)
    assert s.agreement(["wait"], [False]) == (1.0, 1.0)
    assert s.agreement([], []) == (1.0, 1.0)


def test_label_latency():
    assert abs(s.label_latency([1.0, 3.0], [2, 2], 4.0) - 0.5) < 1e-9
    assert s.label_latency([], [], 4.0) == 0.0
    assert abs(s.label_latency([5.0], [1], 4.0) - 1.0) < 1e-9  # capped at dur


def test_parse_clip():
    assert s.parse_clip("/a.wav") == ("/a.wav", None, None)
    assert s.parse_clip("/a.wav#start=1.5#end=4.0") == ("/a.wav", 1.5, 4.0)


def test_load_pool(tmp_path):
    (tmp_path / "s.jsonl").write_text(
        '{"sid":"a","lang":"en","tgt":"ko","src":"the manager approved it","ref":"매니저가 승인했어요","recs":[{"audio":"/x/a.wav"}]}\n'
        '{"sid":"b","lang":"ko","tgt":"en","src":"안녕 하세요","ref":"hello","recs":[{"audio":"/x/b.wav#start=0.0#end=30.0"}]}\n'
        '{"sid":"c","lang":"en","tgt":"ko","src":"no words","ref":"x","recs":[{"audio":"/x/c.wav"}]}\n', encoding="utf-8")
    (tmp_path / "sch.jsonl").write_text(
        '{"sid":"a","schedule":[[2,"매니저가"],[4,"승인했어요"]]}\n{"sid":"b","schedule":[[2,"hello"]]}\n'
        '{"sid":"c","schedule":[[2,""]]}\n', encoding="utf-8")
    (tmp_path / "wt.jsonl").write_text(
        '{"audio":"/x/a.wav","words":[{"w":"the","end":0.5},{"w":"manager","end":1.0},{"w":"approved","end":1.6},{"w":"it","end":1.9}]}\n'
        '{"audio":"/x/b.wav#start=0.0#end=30.0","words":[{"w":"안녕","end":0.5},{"w":"하세요","end":1.0}]}\n', encoding="utf-8")
    pool = s.load_pool(str(tmp_path / "sch.jsonl"), str(tmp_path / "s.jsonl"), str(tmp_path / "wt.jsonl"),
                       ["en-ko", "ko-en"], max_dur=16.0, holdout=0.0)
    assert [u["pair"] for u in pool] == ["en-ko"]  # b exceeds max_dur, c has no text
    u = pool[0]
    assert u["due"] == [1.2, 2.1] and u["chars"] == [4, 5] and u["wav"] == "/x/a.wav"
