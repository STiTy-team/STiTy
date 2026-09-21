import asyncio
import unittest

from core.translator.local_translator import (
    HyMTTranslator, LLMTranslator, NLLBTranslator, TranslateGemmaTranslator, make_translator,
)


class DispatchTest(unittest.TestCase):

    def test_model_names_pick_their_prompt_format(self):
        cases = {
            "google/translategemma-4b-it": TranslateGemmaTranslator,
            "tencent/Hy-MT2-1.8B": HyMTTranslator,
            "Qwen/Qwen3.5-4B": LLMTranslator,
            "unsloth/gemma-3-4b-it": LLMTranslator,
            "facebook/nllb-200-distilled-600M": NLLBTranslator,
        }
        for name, expected in cases.items():
            with self.subTest(name=name):
                self.assertIs(type(make_translator(name)), expected)

    def test_quant_reaches_llm_backends(self):
        self.assertEqual(make_translator("tencent/Hy-MT2-1.8B", quant="none").quant, "none")


class HyMTPromptTest(unittest.TestCase):

    def test_no_system_message_and_card_instruction(self):
        messages = HyMTTranslator()._messages("안녕하세요.", "en", "ko")
        self.assertEqual(messages, [{"role": "user", "content": (
            "Translate the following text into English. Note that you should only output "
            "the translated result without any additional explanation:\n\n안녕하세요.")}])

    def test_context_goes_into_background_block(self):
        messages = HyMTTranslator()._messages(
            "네, 앉으세요.", "en", "ko",
            context=[{"text": "이 자리 비었나요?", "lang": "ko", "translation": "Is this seat free?"}])
        content = messages[0]["content"]
        self.assertTrue(content.startswith("[Background Information]\n- [Korean] 이 자리 비었나요?"))
        self.assertTrue(content.endswith("[Source Text]\n네, 앉으세요."))

    def test_braces_in_source_do_not_break_formatting(self):
        content = HyMTTranslator()._messages("{x} 값", "en", "ko")[0]["content"]
        self.assertTrue(content.endswith("{x} 값"))


class TranslateGemmaTest(unittest.TestCase):

    def test_structured_user_message_without_system(self):
        messages = TranslateGemmaTranslator()._messages("안녕", "en", "ko")
        self.assertEqual(messages, [{"role": "user", "content": [{
            "type": "text", "source_lang_code": "ko", "target_lang_code": "en",
            "text": "안녕"}]}])

    def test_context_is_dropped(self):
        translator = TranslateGemmaTranslator()
        seen = []
        translator._translate_sync = lambda text, tgt, src, ctx: seen.append(ctx) or "Hi."
        result = asyncio.run(translator.translate("안녕.", "en", "ko", context=["앞 문장."]))
        self.assertEqual(result, ("Hi.", "ko"))
        self.assertEqual(seen, [[]])
        self.assertFalse(TranslateGemmaTranslator.SUPPORTS_CONTEXT)


class _Tokenizer:
    eos_token_id = 0

    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, **_):
        return "<bos>" + messages[-1]["content"]

    def __call__(self, text, **kwargs):
        import torch

        self.calls.append(kwargs)
        return {"input_ids": torch.tensor([[1, 2]])}

    def decode(self, ids, **_):
        return "Hello."


class _Model:
    def generate(self, **_):
        import torch

        return torch.tensor([[1, 2, 3]])


class TokenizationTest(unittest.TestCase):

    def test_template_bos_is_not_doubled(self):
        translator = LLMTranslator()
        translator._tokenizer, translator._model, translator._device = _Tokenizer(), _Model(), "cpu"
        self.assertEqual(translator._translate_sync("안녕.", "en", "ko"), "Hello.")
        self.assertIs(translator._tokenizer.calls[0]["add_special_tokens"], False)


if __name__ == "__main__":
    unittest.main()
