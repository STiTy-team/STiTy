import math

from core.components.langid.audio.decide import decide, lag_span, normalise, smooth, with_prior


def test_normalise_keeps_only_candidates():
    probs = normalise({"ko": -1.0, "en": -3.0, "ja": 0.0}, ("ko", "en"))
    assert set(probs) == {"ko", "en"}
    assert math.isclose(sum(probs.values()), 1.0)
    assert probs["ko"] > probs["en"]


def test_normalise_none_when_no_candidate_is_known():
    assert normalise({"ko": -math.inf, "en": -math.inf}, ("ko", "en")) is None
    assert normalise({}, ("ko", "en")) is None


def test_decide_keeps_previous_below_conf():
    assert decide({"ko": 0.6, "en": 0.4}, "en", 0.7) == "en"
    assert decide({"ko": 0.9, "en": 0.1}, "en", 0.7) == "ko"
    assert decide(None, "ko", 0.7) == "ko"
    assert decide(None, None, 0.7) is None
    assert decide({"ko": 0.6, "en": 0.4}, None, 0.7) is None


def test_with_prior_favours_previous():
    out = with_prior({"ko": -1.0, "en": -1.0}, "en", 0.7)
    assert out["en"] > out["ko"]
    assert with_prior({"ko": -1.0, "en": -2.0}, None, 0.7) == {"ko": -1.0, "en": -2.0}


def test_smooth_sticky_middle_step():
    steps = [{"ko": 0.0, "en": -5.0}, {"ko": -1.0, "en": 0.0}, {"ko": 0.0, "en": -5.0}]
    sticky = smooth(steps, ("ko", "en"), 0.9)
    assert sticky[1]["ko"] > sticky[1]["en"]
    loose = smooth(steps, ("ko", "en"), 0.5)
    assert loose[1]["en"] > loose[1]["ko"]
    assert all(math.isclose(sum(p.values()), 1.0) for p in sticky)


def test_smooth_skips_steps_without_scores():
    out = smooth([{"ko": 0.0, "en": -3.0}, None, {"ko": 0.0, "en": -3.0}], ("ko", "en"), 0.9)
    assert out[1]["ko"] > out[1]["en"]


def test_lag_span():
    assert lag_span(10.0, 12.0, 0.5) == (9.5, 11.5)
    assert lag_span(0.2, 1.0, 0.5) == (0.0, 0.5)


def test_decide_tie_keeps_previous():
    assert decide({"ko": 0.5, "en": 0.5}, "en", 0.5) == "en"
    assert decide({"ko": 0.5, "en": 0.5}, None, 0.5) is None


def test_span_mass_none_when_steps_in_span_have_no_speech():
    from core.components.langid.audio.decide import span_mass

    steps = [
        (1.0, {"ko": 0.0, "en": -5.0}),
        (1.5, None),
        (2.0, None),
        (2.5, {"ko": 0.0, "en": -5.0}),
    ]
    posts = [{"ko": 0.99, "en": 0.01}] * 4
    assert span_mass(steps, posts, (1.0, 1.6), 0.5, ("ko", "en")) is None
    assert span_mass(steps, posts, (0.3, 0.6), 0.5, ("ko", "en"))["ko"] > 0.9
    far = span_mass(steps, posts, (5.0, 5.2), 0.5, ("ko", "en"))
    assert far["ko"] > 0.9


def test_combine_mean_and_max():
    from core.components.langid.audio.decide import combine

    per = {
        "w": {"ko": math.log(0.9), "en": math.log(0.1)},
        "e": {"ko": math.log(0.4), "en": math.log(0.6)},
    }
    mean = normalise(combine(per, ("ko", "en"), "mean"), ("ko", "en"))
    assert mean["ko"] > mean["en"]
    weighted = normalise(combine(per, ("ko", "en"), "mean", {"w": 0.1, "e": 1.0}), ("ko", "en"))
    assert weighted["en"] > weighted["ko"]
    top = normalise(combine(per, ("ko", "en"), "max"), ("ko", "en"))
    assert math.isclose(top["ko"], 0.9, abs_tol=1e-6)
    assert combine({}, ("ko", "en"), "mean") is None
    assert combine({"w": {"ko": -math.inf, "en": -math.inf}}, ("ko", "en"), "max") is None


def test_smooth_switches_easily_after_a_gap():
    steps = [{"ko": 0.0, "en": -3.0}, {"ko": 0.0, "en": -3.0}, None, {"ko": -1.5, "en": 0.0}]
    stuck = smooth(steps, ("ko", "en"), 0.99)
    assert stuck[3]["ko"] > stuck[3]["en"]
    freed = smooth(steps, ("ko", "en"), 0.99, p_gap=0.5, gaps=[False, False, False, True])
    assert freed[3]["en"] > freed[3]["ko"]


def test_span_mass_read_modes():
    from core.components.langid.audio.decide import span_mass

    ko, en = {"ko": 0.95, "en": 0.05}, {"ko": 0.05, "en": 0.95}
    steps = [(t, {"x": 0}) for t in (1.0, 1.5, 2.0, 2.5, 3.0)]
    posts = [ko, ko, ko, en, en]
    cands = ("ko", "en")
    assert span_mass(steps, posts, (1.0, 3.0), 0.0, cands)["ko"] > 0.5
    assert span_mass(steps, posts, (1.0, 3.0), 0.0, cands, read="tail", read_sec=1.0)["en"] > 0.5
    assert span_mass(steps, posts, (1.0, 3.0), 0.0, cands, read="end")["en"] > 0.9
