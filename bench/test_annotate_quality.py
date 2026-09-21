import os
import tempfile
import unittest
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from bench import annotate_quality
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
