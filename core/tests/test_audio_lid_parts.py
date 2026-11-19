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
