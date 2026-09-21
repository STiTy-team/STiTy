import asyncio
import os
import tempfile
import unittest
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from core.utils.metrics import annotate_quality
from core.utils import metrics


class AnnotateQualityTest(unittest.TestCase):

    def test_summary_carries_judge_usage(self):
        args = Namespace(judge_model="gpt-5.4-mini", skip_critical=False, skip_judge=False,
                         skip_pseudo_perplexity=True, lm_en="en-lm", lm_ko="ko-lm")
        now = datetime.now(timezone.utc)
        usage = {"calls": 2, "prompt_tokens": 3000, "completion_tokens": 300, "cost": .0036}
        summary = annotate_quality._summary(
            rows=[], score=metrics.RunScore(aggregate={}, unavailable={}),
            source_run=Path("source"), args=args, started=now, finished=now, usage=usage)
        self.assertEqual(summary["usage"], usage)

    def test_pseudo_perplexity_stores_what_token_pooling_needs(self):
        rows = [{"id": "a", "status": "ok", "src_lang": "ko", "hypothesis_translation": "Hi."}]
        scored = {"per_item": {"a": 3.0}, "per_item_nll_sum": {"a": 2.19},
                  "per_item_token_count": {"a": 2}}
        with patch.object(annotate_quality.fluency, "target_lm_pseudo_perplexity",
                          return_value=scored):
            annotate_quality.add_pseudo_perplexity(rows, models={"en": "en-lm"})
        block = rows[0]["metric_inputs"]["fluency"]
        self.assertEqual(block["target_lm_nll_sum"], 2.19)
        self.assertEqual(block["target_lm_token_count"], 2)

    def test_every_system_is_scored_against_one_reference_span_set(self):
        class _Judge:
            model = "fake"

            def __init__(self):
                self.purposes = []

            async def ask(self, *, purpose, system, payload):
                self.purposes.append(purpose)
                if purpose == "critical_information_reference":
                    # A second extraction would come back different; it must not happen.
                    name = "Paris" if self.purposes.count(purpose) == 1 else "France"
                    start = payload["reference"].find(name)
                    return {"reference_spans": [] if start < 0 else [
                        {"type": "location", "text": name, "start": start,
                         "end": start + len(name), "canonical_value": name.lower()}]}
                return {"candidate_spans": [], "alignments": []}

        def rows(candidate):
            return [{"id": "a", "status": "ok", "src_lang": "ko", "reference": "파리 프랑스",
                     "hypothesis": "파리 프랑스", "hypothesis_translation": candidate,
                     "reference_translations": {"en": "Paris, France"}}]

        judge = _Judge()
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "reference.jsonl"
            first = asyncio.run(annotate_quality.annotate_rows(
                rows("Paris."), judge=judge, annotate_spoken_fluency=False,
                reference_cache=annotate_quality.ReferenceSpanCache(cache)))
            second = asyncio.run(annotate_quality.annotate_rows(
                rows("France."), judge=judge, annotate_spoken_fluency=False,
                reference_cache=annotate_quality.ReferenceSpanCache(cache)))
        spans = [row["metric_inputs"]["critical_information"]["reference_spans"]
                 for row in first + second]
        self.assertEqual(judge.purposes.count("critical_information_reference"), 1)
        self.assertEqual(spans[0], spans[1])

    def test_main_reads_the_api_key_from_dotenv(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, ".env").write_text("OPENAI_API_KEY=from-dotenv\n", encoding="utf-8")
            previous = os.getcwd()
            os.chdir(directory)
            try:
                with patch.dict(os.environ, {}, clear=False):
                    os.environ.pop("OPENAI_API_KEY", None)
                    with self.assertRaises(ValueError):
                        annotate_quality.main(["missing-run", "out"])
                    self.assertEqual(os.environ.get("OPENAI_API_KEY"), "from-dotenv")
            finally:
                os.chdir(previous)


if __name__ == "__main__":
    unittest.main()
