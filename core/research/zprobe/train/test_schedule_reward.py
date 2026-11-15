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


# --- prefix16 GRPO: label-aware pool and shape terms -----------------------------------------------------------
def _pool_files(tmp_path, sched_lines, sent_lines):
    (tmp_path / "s.jsonl").write_text("\n".join(sent_lines) + "\n", encoding="utf-8")
    (tmp_path / "sch.jsonl").write_text("\n".join(sched_lines) + "\n", encoding="utf-8")
    (tmp_path / "wt.jsonl").write_text(
        '{"audio":"/x/a.wav","words":[{"w":"It","end":0.5},{"w":"rained.","end":1.0},{"w":"We","end":1.4},{"w":"left.","end":2.0}]}\n'
        '{"audio":"/x/b.wav","words":[{"w":"It","end":0.5},{"w":"rained.","end":1.0},{"w":"We","end":1.4},{"w":"left.","end":2.0}]}\n'
        '{"audio":"/x/c.wav","words":[{"w":"It","end":0.5},{"w":"rained.","end":1.0},{"w":"We","end":1.4},{"w":"left.","end":2.0}]}\n',
        encoding="utf-8")
    return [str(tmp_path / f) for f in ("sch.jsonl", "s.jsonl", "wt.jsonl")]


def test_load_pool_skips_rescued_waits_and_missing_references(tmp_path):
    src = "It rained. We left."
    files = _pool_files(tmp_path,
                        ['{"sid":"a","schedule":[[2,"비가 왔고"],[4,"우리는 떠났습니다."]]}',
                         '{"sid":"b","schedule":[[4,"비가 와서 떠났습니다."]],"rescue":"wait"}',
                         '{"sid":"c","schedule":[[2,"비가 왔고"],[4,"떠났습니다."]]}'],
                        [f'{{"sid":"a","lang":"en","tgt":"ko","src":"{src}","ref":"비가 와서 떠났다","recs":[{{"audio":"/x/a.wav"}}]}}',
                         f'{{"sid":"b","lang":"en","tgt":"ko","src":"{src}","ref":"비가 와서 떠났다","recs":[{{"audio":"/x/b.wav"}}]}}',
                         f'{{"sid":"c","lang":"en","tgt":"ko","src":"{src}","ref":null,"recs":[{{"audio":"/x/c.wav"}}]}}'])
    pool = s.load_pool(*files, ["en-ko"], max_dur=16.0, holdout=0.0)
    assert [u["wav"] for u in pool] == ["/x/a.wav"]
    assert pool[0]["src_close"] == [1.0, 2.0]  # end times of the source words that close a sentence


def test_shape_terms_count_closes_before_the_source_closed():
    close = [1.0, 2.0]
    # "비가 왔습니다." at 0.8 s closes before the source closed its first sentence (1.0 s)
    assert s.shape_terms([(0.8, "비가 왔습니다."), (2.0, "우리는 떠났습니다.")], close, "ko") == (0.5, 0.0)
    # closing after hearing "rained." is allowed; the final close is too
    assert s.shape_terms([(1.5, "비가 왔습니다."), (2.0, "우리는 떠났습니다.")], close, "ko") == (0.0, 0.0)
    assert s.shape_terms([(1.5, "비가 왔고"), (2.0, "떠났습니다.")], close, "ko") == (0.0, 0.0)
    assert s.shape_terms([], close, "ko") == (0.0, 0.0)


def test_shape_terms_flag_a_bare_subject_first_piece_but_not_a_lone_final_piece():
    close = [2.0]
    assert s.shape_terms([(1.0, "그는"), (2.0, "떠났습니다.")], close, "ko")[1] == 1.0
    assert s.shape_terms([(2.0, "그는 떠났습니다.")], close, "ko")[1] == 0.0
    assert s.shape_terms([(1.0, "They"), (2.0, "left.")], close, "en")[1] == 1.0


def test_latency_floor_scales_the_label_latency():
    assert abs(s.latency_charge(0.5, 0.6, 1.0) - 0.6) < 1e-9
    assert abs(s.latency_charge(0.5, 0.6, 0.75) - 0.5) < 1e-9   # floor 0.45 is below the rollout's own 0.5
    assert abs(s.latency_charge(0.3, 0.6, 0.75) - 0.45) < 1e-9
    assert abs(s.latency_charge(0.3, 0.0, 0.75) - 0.3) < 1e-9   # no schedule: plain latency
