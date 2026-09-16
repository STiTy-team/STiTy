#!/usr/bin/env python3
"""dot + <SEG> 동시 축(SEGDOT)에서 롤백 창 안의 `. <SEG>` 가 즉시 커밋되는지 (GPU/모델 불필요).

dot 을 켜면 경계 탐색이 DOT_COMMIT_BOUNDARY_RE 하나로 가고, `"produced. <SEG>"` 에서는
마침표가 <SEG> 보다 앞이라 매치가 `'. '` 로 잡혀 trigger 가 dot 이 된다. 그러면 확정
게이트로 들어가는데, 꼬리가 `<SEG>` 뿐이라 규칙 1(context)은 불발이고 규칙 3(stall)은
`after` 가 비어 있지 않아 불발이라 규칙 2(다음 청크 끝)까지 한 청크를 그냥 기다렸다.
seg 축이라면 생성 도중 즉시 커밋될 자리다.

규칙 0 은 마침표가 롤백 창 안이고 그 창 안에 <SEG> 가 있으면 <SEG> 까지를 seg 로
자른다. 여기서는 그 규칙과, 그 규칙이 건드리면 안 되는 경계를 검증한다.

모델을 띄우지 않으므로 핸들러는 `object.__new__` 로 만들고 필요한 속성만 채운다.
`asr` 이 없어 `_count_tokens` 는 단어 수로 근사한다.
"""
import asyncio
import importlib.util
import logging
import os
import sys
import unittest

_SERVER = os.path.join(os.path.dirname(__file__), "..", "examples",
                       "streaming_websocket_server.py")
_spec = importlib.util.spec_from_file_location("_sws_under_test_segdot", _SERVER)
_sws = importlib.util.module_from_spec(_spec)
sys.modules["_sws_under_test_segdot"] = _sws
_spec.loader.exec_module(_sws)

H = _sws.Qwen3ASRStreamingHandler


class _FakeState:
    def __init__(self, text, unfixed_token_num=5):
        self.text = text
        self.language = "en"
        self.audio_accum = None
        self.unfixed_token_num = unfixed_token_num
        self._raw_decoded = text


def _make_handler():
    h = object.__new__(H)
    h.log = logging.getLogger("test_seg_in_dot_rollback_window")
    h.asr_lock = asyncio.Lock()
    h.active_slot = "A"
    h.standby_slot = "B"
    h.current_time = 5.63
    h._in_generate_loop = False
    h._pending_gpt_tasks = []
    h.enable_dot_commit = True
    h.dot_commit_confirm = True
    h.dot_commit_stall_chunks = 1
    h.always_commit = False
    h.rep_dedup = True
    h.emitted = []  # [(reason, original)]

    async def _correct_and_translate(text, lang, audio_end_sec=None):
        return text, "<de>" + text, "en", {}

    async def _emit_final_payload(**kw):
        h.emitted.append((kw["reason"], kw["original"]))

    h._correct_and_translate = _correct_and_translate
    h._emit_final_payload = _emit_final_payload
    return h


def _make_slot(h, full_text, unfixed_token_num=5):
    slot = {
        "state": _FakeState(full_text, unfixed_token_num),
        "flush_lock": asyncio.Lock(),
        "last_text": "",
        "last_text_lang": "en",
        "committed_len": 0,
        "committed_prefix": "",
        "committed_display": "",
        "committed_seg_count": 0,
        "audio_anchor_sec": 0.0,
        "committed_asr_set": set(),
        "committed_fuzzy_keys": [],
    }
    h.stream_slots = {"A": slot, "B": None}
    return slot


def _leftover(slot):
    cur = H._strip_asr_text((slot["state"].text or "").strip())
    unc = H._uncommitted_from(cur, slot["committed_display"],
                              slot["committed_seg_count"])
    return unc.replace("<SEG>", "").strip()


class SegInDotRollbackWindowTests(unittest.IsolatedAsyncioTestCase):

    async def test_dot_then_seg_commits_immediately_as_seg(self):
        """`. <SEG>`: 생성 도중 콜백(chunk_end=False)에서도 즉시 seg 커밋."""
        h = _make_handler()
        slot = _make_slot(h, "The season was cancelled. <SEG>")
        await h._process_slot_updates("A", chunk_end=False)
        self.assertEqual(h.emitted, [("seg", "The season was cancelled.")])
        self.assertEqual(_leftover(slot), "")
        self.assertEqual(slot["committed_seg_count"], 1)
        self.assertNotIn("pending_dot_text", slot)

    async def test_question_mark_then_seg(self):
        h = _make_handler()
        slot = _make_slot(h, "Is that right? <SEG> and then")
        await h._process_slot_updates("A", chunk_end=False)
        self.assertEqual(h.emitted, [("seg", "Is that right?")])
        self.assertEqual(_leftover(slot), "and then")

    async def test_short_sentence_between_dot_and_seg_goes_out_together(self):
        """dot 뒤 짧은 문장 + <SEG>: seg 축과 같은 단위(한 덩어리)로 나간다."""
        h = _make_handler()
        slot = _make_slot(h, "Yes. Okay <SEG>")
        await h._process_slot_updates("A", chunk_end=False)
        self.assertEqual(h.emitted, [("seg", "Yes. Okay")])
        self.assertEqual(_leftover(slot), "")

    async def test_dot_without_seg_in_window_still_waits(self):
        """<SEG> 가 없는 롤백 창 안 마침표는 그대로 보류(회귀 방지)."""
        h = _make_handler()
        slot = _make_slot(h, "The season was cancelled.")
        await h._process_slot_updates("A", chunk_end=True)
        self.assertEqual(h.emitted, [])
        self.assertEqual(slot.get("pending_dot_text"), "The season was cancelled.")

    async def test_dot_outside_window_with_seg_far_behind_uses_rule1(self):
        """꼬리가 창보다 길면 규칙 1 이 dot 을 따로 확정하고, 뒤 <SEG> 는 같은 호출에서 seg."""
        h = _make_handler()
        slot = _make_slot(h, "First one. Then six more words come here <SEG>")
        await h._process_slot_updates("A", chunk_end=False)
        self.assertEqual(h.emitted, [("dot", "First one."),
                                     ("seg", "Then six more words come here")])
        self.assertEqual(_leftover(slot), "")


if __name__ == "__main__":
    unittest.main(verbosity=2)
