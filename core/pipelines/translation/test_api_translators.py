import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import httpx

from core.errors import ConfigError
from core.pipelines.translation import translators


class _Server:

    def __init__(self, *replies):
        self.replies = list(replies)
        self.requests = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else None
        self.requests.append((request.method, str(request.url), dict(request.headers), body))
        status, payload = self.replies.pop(0)
        return httpx.Response(status, json=payload)

    def posts(self):
        return [r for r in self.requests if r[0] == "POST"]


def _run(backend: str, options: dict, server: _Server, env: dict, calls, usage_log=None):
    real_client = httpx.AsyncClient

    def client(**kwargs):
        return real_client(transport=httpx.MockTransport(server), **kwargs)

    async def go():
        translator = translators.get(backend)(
            translators.get(backend).validate(options, kind="translation"), cfg=None)
        translator.usage_log = usage_log
        await translator.load()
        try:
            return translator, [await translator.translate(*args, **kwargs)
                                for args, kwargs in calls]
        finally:
            await translator.close()

    with mock.patch.dict(os.environ, env), \
            mock.patch("httpx.AsyncClient", client), \
            mock.patch("core.pipelines.translation.api.asyncio.sleep", mock.AsyncMock()):
        return asyncio.run(go())


def _gemini_reply(text, *, prompt=100, candidates=10, thoughts=5, thought_text=None):
    parts = ([{"text": thought_text, "thought": True}] if thought_text else []) + [
        {"text": json.dumps({"translation": text}, ensure_ascii=False)}]
    return 200, {"candidates": [{"content": {"parts": parts}, "finishReason": "STOP"}],
                 "usageMetadata": {"promptTokenCount": prompt,
                                   "candidatesTokenCount": candidates,
                                   "thoughtsTokenCount": thoughts}}


def _gpt_reply(text, *, prompt=100, cached=0, completion=10):
    content = json.dumps({"translation": text}, ensure_ascii=False)
    return 200, {"choices": [{"message": {"content": content}, "finish_reason": "stop"}],
                 "usage": {"prompt_tokens": prompt, "completion_tokens": completion,
                           "prompt_tokens_details": {"cached_tokens": cached},
                           "completion_tokens_details": {"reasoning_tokens": 0}}}


class DeepLTest(unittest.TestCase):

    def test_request_shape_cost_and_usage_log(self):
        server = _Server(
            (200, {"character_count": 10, "character_limit": 1000}),
            (200, {"translations": [{"text": " Hello. ", "detected_source_language": "KO",
                                     "billed_characters": 4000,
                                     "model_type_used": "quality_optimized"}]}),
        )
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "usage.jsonl"
            translator, results = _run(
                "deepl", {"model_type": "quality_optimized", "context": 2}, server,
                {"DEEPL_API_KEY": "pro-key"},
                [(("안녕하세요.", "en", "ko"), {"context": [
                    {"text": "첫 줄.", "lang": "ko"},
                    {"text": "둘째 줄.", "lang": "ko"},
                    {"text": "셋째 줄", "lang": "ko"},
                ]})],
                usage_log=log)
            lines = [json.loads(line) for line in log.read_text().splitlines()]

        self.assertEqual(results, [("Hello.", "ko")])
        method, url, headers, body = server.posts()[0]
        self.assertEqual(url, "https://api.deepl.com/v2/translate")
        self.assertEqual(headers["authorization"], "DeepL-Auth-Key pro-key")
        self.assertEqual(body["target_lang"], "EN-US")
        self.assertEqual(body["source_lang"], "KO")
        self.assertEqual(body["model_type"], "quality_optimized")
        self.assertEqual(body["context"], "둘째 줄. 셋째 줄")
        self.assertTrue(body["show_billed_characters"])
        self.assertEqual(lines[0]["billed_characters"], 4000)
        self.assertEqual(lines[0]["model_type_used"], "quality_optimized")
        self.assertAlmostEqual(lines[0]["cost"], 0.1)
        self.assertAlmostEqual(translator.usage()["cost"], 0.1)
        self.assertEqual(translator.usage()["by_purpose"]["translation"]["calls"], 1)

    def test_free_key_uses_free_host_and_costs_nothing(self):
        server = _Server(
            (200, {"character_count": 0, "character_limit": 500000}),
            (200, {"translations": [{"text": "Hi", "detected_source_language": "KO",
                                     "billed_characters": 2}]}),
        )
        translator, _ = _run("deepl", {"model_type": "latency_optimized"}, server,
                             {"DEEPL_API_KEY": "free-key:fx"}, [(("안녕", "en", "ko"), {})])
        self.assertTrue(server.posts()[0][1].startswith("https://api-free.deepl.com/"))
        self.assertEqual(translator.usage()["billed_characters"], 2)
        self.assertEqual(translator.usage()["cost"], 0.0)

    def test_earlier_translations_go_into_the_context_too(self):
        server = _Server((200, {}), (200, {"translations": [{"text": "Yes."}]}))
        _run("deepl", {"model_type": "latency_optimized"}, server, {"DEEPL_API_KEY": "k"},
             [(("응.", "en", "ko"), {"context": [
                 {"text": "봤어?", "lang": "ko", "translation": "Did you?"}]})])
        self.assertEqual(server.posts()[0][3]["context"], "봤어?\nDid you?")

    def test_asr_speakers_become_labelled_context_lines(self):
        server = _Server((200, {}), (200, {"translations": [{"text": "Yes."}]}))
        _run("deepl", {"model_type": "latency_optimized"}, server, {"DEEPL_API_KEY": "k"},
             [(("응.", "en", "ko"), {"speaker": "B", "context": [
                 {"speaker": "A", "text": "봤어?", "lang": "ko"}]})])
        self.assertEqual(server.posts()[0][3]["context"], "A (Korean): 봤어?")

    def test_quota_error_is_not_retried(self):
        server = _Server((200, {}), (456, {"message": "Quota exceeded"}))
        translator, results = _run("deepl", {"model_type": "latency_optimized"}, server,
                                   {"DEEPL_API_KEY": "k"}, [(("안녕", "en", "ko"), {})])
        self.assertEqual(results, [("", "ko")])
        self.assertEqual(len(server.posts()), 1)
        self.assertEqual(translator.failed, 1)


class GeminiTest(unittest.TestCase):

    def test_thoughts_are_billed_and_hidden(self):
        server = _Server((200, {"name": "models/gemini-3.1-flash-lite"}),
                         _gemini_reply('"Hello."', prompt=1000, candidates=100, thoughts=100,
                                       thought_text="thinking"))
        translator, results = _run("gemini", {}, server, {"GEMINI_API_KEY": "g"},
                                   [(("안녕하세요.", "en", "ko"), {})])
        self.assertEqual(results, [("Hello.", "ko")])
        _, url, headers, body = server.posts()[0]
        self.assertTrue(url.endswith("/models/gemini-3.1-flash-lite:generateContent"))
        self.assertEqual(body["generationConfig"]["responseMimeType"], "application/json")
        self.assertEqual(headers["x-goog-api-key"], "g")
        self.assertEqual(body["generationConfig"]["thinkingConfig"]["thinkingLevel"], "minimal")
        self.assertNotIn("temperature", body["generationConfig"])
        self.assertIn("translation engine", body["systemInstruction"]["parts"][0]["text"])
        usage = translator.usage()
        self.assertEqual(usage["output_tokens"], 200)
        self.assertEqual(usage["reasoning_tokens"], 100)
        self.assertAlmostEqual(usage["cost"], (1000 * 0.25 + 200 * 1.50) / 1_000_000)

    def test_without_speakers_the_last_finals_go_in_as_one_block(self):
        finals = [{"text": f"문장{i}.", "lang": "ko", "translation": f"Line {i}."}
                  for i in range(7)]
        server = _Server((200, {}), _gemini_reply("was odd"))
        _run("gemini", {"context": 5}, server, {"GEMINI_API_KEY": "g"},
             [(("좀 이상했어", "en", "ko"), {"context": finals})])
        user = server.posts()[0][3]["contents"][0]["parts"][0]["text"]
        self.assertIn("Speakers are not identified", user)
        self.assertIn("\n문장2. 문장3. 문장4. 문장5. 문장6.\n", user)
        self.assertIn("\nLine 2. Line 3. Line 4. Line 5. Line 6.\n", user)
        self.assertNotIn("문장1.", user)
        self.assertNotIn("A (", user)
        self.assertTrue(user.endswith("Translate it into English:\n좀 이상했어"))

    def test_default_is_ten_finals_within_five_hundred_characters(self):
        finals = [{"text": f"{i:02d}" + "가" * 58, "lang": "ko"} for i in range(12)]
        server = _Server((200, {}), _gemini_reply("x"))
        _run("gemini", {}, server, {"GEMINI_API_KEY": "g"},
             [(("새 조각", "en", "ko"), {"context": finals})])
        user = server.posts()[0][3]["contents"][0]["parts"][0]["text"]
        kept = [i for i in range(12) if f"{i:02d}가" in user]
        self.assertEqual(kept, list(range(4, 12)))

    def test_mixed_languages_are_tagged_in_the_block(self):
        server = _Server((200, {}), _gemini_reply("x"))
        _run("gemini", {}, server, {"GEMINI_API_KEY": "g"},
             [(("그래.", "en", "ko"), {"context": [
                 {"text": "봤어?", "lang": "ko"}, {"text": "Yes.", "lang": "en"}]})])
        user = server.posts()[0][3]["contents"][0]["parts"][0]["text"]
        self.assertIn("[Korean] 봤어? [English] Yes.", user)

    def test_missing_translation_hides_the_translation_block(self):
        server = _Server((200, {}), _gemini_reply("b"))
        _run("gemini", {}, server, {"GEMINI_API_KEY": "g"},
             [(("둘.", "en", "ko"), {"context": [{"text": "하나.", "lang": "ko"}]})])
        user = server.posts()[0][3]["contents"][0]["parts"][0]["text"]
        self.assertIn("하나.", user)
        self.assertNotIn("already seen", user)


class GPTTest(unittest.TestCase):

    def test_request_shape_and_cost(self):
        server = _Server((200, {"id": "gpt-5.4-nano"}),
                         _gpt_reply("Hello.", prompt=1000, cached=200, completion=100))
        translator, results = _run("gpt", {"model": "gpt-5.4-nano"}, server,
                                   {"OPENAI_API_KEY": "sk"}, [(("안녕하세요.", "en", "ko"), {})])
        self.assertEqual(results, [("Hello.", "ko")])
        _, url, headers, body = server.posts()[0]
        self.assertEqual(url, "https://api.openai.com/v1/chat/completions")
        self.assertEqual(headers["authorization"], "Bearer sk")
        self.assertEqual(body["reasoning_effort"], "none")
        self.assertEqual(body["temperature"], 0)
        self.assertAlmostEqual(translator.usage()["cost"],
                               (800 * 0.20 + 200 * 0.02 + 100 * 1.25) / 1_000_000)

    def test_speaker_change_is_marked_and_label_stripped(self):
        server = _Server((200, {}), _gpt_reply("B: Yeah, I did."))
        _, results = _run("gpt", {}, server, {"OPENAI_API_KEY": "sk"}, [
            (("응, 봤어.", "en", "ko"), {"speaker": "B", "context": [
                {"speaker": "A", "text": "그 영화 봤어?", "lang": "ko"}]})])
        self.assertEqual(results, [("Yeah, I did.", "ko")])
        body = server.posts()[0][3]
        self.assertEqual(body["response_format"], {"type": "json_object"})
        user = body["messages"][1]["content"]
        self.assertIn("A (Korean): 그 영화 봤어?", user)
        self.assertIn("speaker change, now B", user)

    def test_reply_without_translation_field_fails_the_segment(self):
        server = _Server((200, {}), (200, {
            "choices": [{"message": {"content": "Hello"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2}}))
        translator, results = _run("gpt", {}, server, {"OPENAI_API_KEY": "sk"},
                                   [(("안녕", "en", "ko"), {})])
        self.assertEqual(results, [("", "ko")])
        self.assertEqual(translator.failed, 1)
        self.assertEqual(translator.usage()["calls"], 1)

    def test_context_zero_sends_no_context(self):
        server = _Server((200, {}), _gpt_reply("Hi"))
        _run("gpt", {"context": 0}, server, {"OPENAI_API_KEY": "sk"},
             [(("안녕", "en", "ko"), {"context": [{"speaker": "A", "text": "앞 턴."}]})])
        self.assertNotIn("앞 턴.", server.posts()[0][3]["messages"][1]["content"])

    def test_reasoning_drops_temperature(self):
        server = _Server((200, {}), _gpt_reply("Hi"))
        _run("gpt", {"reasoning_effort": "low"}, server, {"OPENAI_API_KEY": "sk"},
             [(("안녕", "en", "ko"), {})])
        self.assertNotIn("temperature", server.posts()[0][3])

    def test_rate_limit_is_retried(self):
        server = _Server((200, {}), (429, {"error": "slow down"}), _gpt_reply("Hi"))
        translator, results = _run("gpt", {}, server, {"OPENAI_API_KEY": "sk"},
                                   [(("안녕", "en", "ko"), {})])
        self.assertEqual(results, [("Hi", "ko")])
        self.assertEqual(len(server.posts()), 2)
        self.assertEqual(translator.usage()["calls"], 1)

    def test_budget_stops_calls(self):
        server = _Server((200, {}), _gpt_reply("Hi", prompt=1_000_000, completion=1_000_000))
        translator, results = _run("gpt", {"budget_usd": 0.5}, server, {"OPENAI_API_KEY": "sk"},
                                   [(("안녕", "en", "ko"), {}), (("또", "en", "ko"), {})])
        self.assertEqual(results, [("Hi", "ko"), ("", "ko")])
        self.assertEqual(len(server.posts()), 1)
        self.assertEqual(translator.failed, 1)


class CommonTest(unittest.TestCase):

    def test_missing_key_fails_before_any_request(self):
        server = _Server()
        with self.assertRaises(ConfigError):
            _run("gpt", {}, server, {"OPENAI_API_KEY": ""}, [])
        self.assertEqual(server.requests, [])

    def test_same_language_is_not_sent(self):
        server = _Server((200, {}))
        translator, results = _run("gpt", {}, server, {"OPENAI_API_KEY": "sk"},
                                   [(("hello", "en", "en"), {})])
        self.assertEqual(results, [("hello", "en")])
        self.assertEqual(server.posts(), [])


if __name__ == "__main__":
    unittest.main()
