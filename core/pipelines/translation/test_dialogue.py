import unittest

from core.pipelines.translation.dialogue import Dialogue, recent, speaker_label


class DialogueTest(unittest.TestCase):

    def test_finals_carry_across_items_of_a_group(self):
        dialogue = Dialogue()
        dialogue.open(group="g")
        dialogue.said("어제 그 영화", "That movie yesterday", lang="ko", target="en")
        dialogue.said("봤어?", "did you see it?", lang="ko", target="en")
        dialogue.open(group="g")
        dialogue.said("응 봤어.", "Yeah, I did.", lang="ko", target="en")
        self.assertEqual(dialogue.context("en"), [
            {"text": "어제 그 영화", "lang": "ko", "translation": "That movie yesterday"},
            {"text": "봤어?", "lang": "ko", "translation": "did you see it?"},
            {"text": "응 봤어.", "lang": "ko", "translation": "Yeah, I did."},
        ])

    def test_new_group_forgets_the_old_dialogue(self):
        dialogue = Dialogue()
        dialogue.open(group="g1")
        dialogue.said("안녕.", "Hi.", lang="ko", target="en")
        dialogue.open(group="g2")
        self.assertEqual(dialogue.context("en"), [])

    def test_asr_speakers_become_letters_in_order_of_appearance(self):
        dialogue = Dialogue()
        dialogue.open(group="g")
        dialogue.said("그 영화 봤어?", "", lang="ko", target="en", speaker="spk-17")
        dialogue.said("응.", "", lang="ko", target="en", speaker="spk-03")
        self.assertEqual([entry["speaker"] for entry in dialogue.context("en")], ["A", "B"])
        self.assertEqual(dialogue.label("spk-17"), "A")

    def test_translation_is_shown_only_for_the_same_target(self):
        dialogue = Dialogue()
        dialogue.open(group="g")
        dialogue.said("안녕하세요.", "Hello.", lang="ko", target="en")
        self.assertEqual(dialogue.context("ko"), [{"text": "안녕하세요.", "lang": "ko"}])
        self.assertEqual(dialogue.context("en")[0]["translation"], "Hello.")

    def test_empty_finals_are_not_kept(self):
        dialogue = Dialogue()
        dialogue.open(group="g")
        dialogue.said("  ", "", lang="ko", target="en")
        self.assertEqual(dialogue.context("en"), [])

    def test_recent_keeps_the_newest_finals_under_both_limits(self):
        finals = [{"text": text} for text in ("aaaa", "bbb", "", "cc", "d")]
        texts = lambda **kw: [e["text"] for e in recent(finals, "en", **kw)]
        self.assertEqual(texts(count=10, max_chars=100), ["aaaa", "bbb", "cc", "d"])
        self.assertEqual(texts(count=2, max_chars=100), ["cc", "d"])
        self.assertEqual(texts(count=10, max_chars=6), ["bbb", "cc", "d"])
        self.assertEqual(recent(finals, "en", 0, 100), [])
        self.assertEqual(recent([{"text": "x" * 600}], "en", 10, 500), [])
        self.assertEqual(recent(["plain"], "en"), [{"text": "plain"}])

    def test_one_missing_translation_removes_translations_for_everyone(self):
        finals = [{"text": "하나.", "lang": "ko", "translation": "One."},
                  {"text": "둘.", "lang": "ko"},
                  {"text": "Three.", "lang": "en"}]
        self.assertEqual(recent(finals, "en"), [
            {"text": "하나.", "lang": "ko"}, {"text": "둘.", "lang": "ko"},
            {"text": "Three.", "lang": "en"}])
        self.assertEqual(recent(finals[:1] + finals[2:], "en")[0]["translation"], "One.")

    def test_labels_run_past_z(self):
        self.assertEqual([speaker_label(i) for i in (0, 1, 25, 26)], ["A", "B", "Z", "AA"])


if __name__ == "__main__":
    unittest.main()
