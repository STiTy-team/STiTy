import asyncio
import unittest
from unittest import mock

from core.errors import ConfigError
from core.pipelines.translation import translators
from core.pipelines.translation.api import ApiTranslator
from core.translator.local_translator import LLMTranslator


class _Backend:
    SUPPORTS_CONTEXT = True

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.contexts = []

    def load(self):
        pass

    async def translate(self, text, target, source=None, context=None):
        self.contexts.append(context)
        return "ok", source


def _local(options: dict, context: list[dict], backend=_Backend):
    local = translators.get("local")
    options = {"model": "Qwen/Qwen3.5-4B", **options}
    translation = local(local.validate(options, kind="translation"), cfg=None)
    with mock.patch("core.translator.local_translator.make_translator",
                    lambda model_name=None, **kwargs: backend(**kwargs)):
        asyncio.run(translation.load())
    asyncio.run(translation.translate("새 조각", "en", "ko", context=context))
    return translation.translator


class _Recorder(ApiTranslator):
    KEY_ENV = "UNUSED"
    NAME = "recorder"

    def __init__(self, settings):
        super().__init__(settings, cfg=None)
        self.spend, self.budget, self.contexts = None, None, []
        self.context_finals = settings.get("context", 10)
        self.context_chars = settings.get("context_chars", 500)

    async def request(self, text, target_lang, source_lang, context, speaker):
        self.contexts.append(context)
        return "ok", source_lang


class LocalContextTest(unittest.TestCase):

    def test_count_and_characters_are_cut_before_the_backend(self):
        finals = [{"text": "가" * 100, "lang": "ko"} for _ in range(8)]
        self.assertEqual(len(_local({"context": 10}, finals).contexts[0]), 5)
        self.assertEqual(len(_local({"context": 3}, finals).contexts[0]), 3)
        self.assertEqual(_local({"context": 0}, finals).contexts[0], [])

    def test_defaults_match_the_api_translators(self):
        finals = [{"text": f"{i}.", "lang": "ko"} for i in range(12)]
        backend = _local({}, finals)
        self.assertEqual(len(backend.contexts[0]), 10)
        self.assertEqual(backend.kwargs["context_window"], 10)
        self.assertTrue(backend.kwargs["always_use_context"])

    def test_local_and_api_receive_the_same_context(self):
        finals = ([{"text": "가" * 90 + ".", "lang": "ko", "translation": "A."}
                   for _ in range(6)]
                  + [{"text": "나.", "lang": "ko"}, {"text": "Hi.", "lang": "en"}])
        for options in ({}, {"context": 3}, {"context": 10, "context_chars": 200}):
            with self.subTest(options=options):
                api = _Recorder(options)
                asyncio.run(api.translate("새 조각", "en", "ko", context=finals))
                self.assertEqual(_local(options, finals).contexts[0], api.contexts[0])

    def test_models_without_a_context_slot_refuse_context(self):
        local = translators.get("local")
        with self.assertRaises(ConfigError):
            local.validate({"model": "google/translategemma-4b-it", "context": 10},
                           kind="translation")
        with self.assertRaises(ConfigError):
            local.validate({"model": "google/madlad400-3b-mt", "context": 1},
                           kind="translation")
        local.validate({"model": "google/translategemma-4b-it", "context": 0},
                       kind="translation")

    def test_models_without_a_context_slot_default_to_none(self):
        class Seq2Seq(_Backend):
            SUPPORTS_CONTEXT = False

        backend = _local({"model": "google/madlad400-3b-mt"},
                         [{"text": "앞.", "lang": "ko"}], backend=Seq2Seq)
        self.assertEqual(backend.contexts[0], [])


class AlwaysUseContextTest(unittest.TestCase):

    def _contexts(self, **kwargs):
        translator = LLMTranslator(context_window=10, **kwargs)
        seen = []
        translator._translate_sync = lambda text, tgt, src, ctx: seen.append(ctx) or "x"
        asyncio.run(translator.translate("근데 결말이", "en", "ko",
                                         context=[{"text": "그 영화 봤어?", "lang": "ko"}]))
        return seen

    def test_fragment_keeps_context_when_switched_on(self):
        self.assertEqual(self._contexts(always_use_context=True),
                         [[{"text": "그 영화 봤어?", "lang": "ko"}]])

    def test_default_still_drops_context_for_fragments(self):
        self.assertEqual(self._contexts(), [[]])


if __name__ == "__main__":
    unittest.main()
