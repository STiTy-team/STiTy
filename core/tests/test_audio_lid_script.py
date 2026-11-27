from core.components.langid.audio.script import text_language


def test_distinct_scripts_decide():
    assert text_language("Wait, wait.", ("ko", "en")) == "en"
    assert text_language("부장님 앞에서는 좀", ("ko", "en")) == "ko"
    assert text_language("ありがとう", ("ja", "en")) == "ja"


def test_shared_script_gives_no_verdict():
    assert text_language("gracias amigo", ("es", "en")) is None
    assert text_language("bonjour", ("fr", "de", "en")) is None


def test_han_counts_as_japanese_only_without_chinese():
    assert text_language("東京", ("ja", "en")) == "ja"
    assert text_language("東京", ("ja", "zh")) is None
    assert text_language("东京", ("zh", "en")) == "zh"


def test_mixed_or_empty_text_gives_no_verdict():
    assert text_language("OK, uh, 감사합니다", ("ko", "en")) is None
    assert text_language("...", ("ko", "en")) is None


def test_any_kana_means_japanese_in_a_japanese_session():
    assert text_language("itmonthで話題", ("ko", "ja", "en")) == "ja"
    assert (
        text_language("itmonthで話題", ("ko", "en")) is None
        or text_language("itmonthで話題", ("ko", "en")) == "en"
    )
