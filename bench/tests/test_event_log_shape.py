#!/usr/bin/env python3
"""The event-sourcing and reporting chain, with no model and no GPU.

Covers sink -> JSONL -> projections -> three output files. The handler is built
with object.__new__ and only the attributes the commit path reads are filled in,
the idiom Qwen3-ASR/tests/test_final_residual_commit.py established. That means
this exercises the real ASTStreamingHandler.send_message -- including the
decisionAudioSec correction and the utterance-attribution guard -- without vLLM.

What it deliberately does not cover: process_audio_chunk and the VAD path, which
need the real model. That is step S6's parity run.
"""
import asyncio
import json
import logging
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bench import config, events, metrics, report


def make_cfg(**patch):
    raw = {
        "name": "evtest",
        "dataset": {"name": "tiny"},
        "languages": {"map": {"en": "ko", "fr": "ko"}},
        "stity": {
            "transcription": {"name": "qwen3", "model": "Qwen/Qwen3-ASR-1.7B"},
            "translation": "none",
            "commit": "seg",
            "gpu_memory_utilization": 0.5,
        },
        "metrics": ["wer", "fsl", "commit", "routing"],
        "logs": {"top_k": 2, "rank_by": "wer"},
    }
    raw.update(patch)
    return config.parse(raw)


class FakeDataset:
    name = "tiny"
    has_transcript = True
    translation_langs = ["ko"]
    languages = ["en", "fr"]
    audio_format = "flac"
    sample_rate = 16000
    primary_metric = "wer"
    group_rule = "conv"
    manifest_sha256 = "deadbeef"
    split = "test"
    root = Path("/nonexistent")
    bench_defaults: dict = {}
    items: list = []
    n_sessions = 2

    def provenance(self):
        return {"name": self.name, "n_items": 2}


# A final payload exactly as the three handler layers emit it: camelCase from the
# base and AST layers, snake_case from FSL.
SERVER_FINAL = {
    "type": "final",
    "start": "00:00:00.000",
    "end": "00:00:02.940",
    "original": "Mister Quilter is the apostle of the middle classes.",
    "translation": "퀼터 씨는 중산층의 사도다",
    "language": "en",
    "commitReason": "seg",
    "segmentId": 1,
    "audioStartSec": 0.0,
    "audioEndSec": 2.94,
    "decisionAudioSec": 2.94,
    "emitElapsedSec": 3.07,
    "audioReceivedSec": 3.0,
    "fsl_sec": 0.141,
    "trans_sec": 0.42,
    "chunk_encode_log": [{"chunk_id": 0, "encode_sec": 0.08}],
}


class TestSinkAndStream(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "events.jsonl"
        self.rec = events.Recorder()
        self.handler = logging.FileHandler(self.path, encoding="utf-8")
        self.handler.setFormatter(events.JsonlFormatter(self.rec))
        self.rec._logger = logging.getLogger("bench.test.event")
        self.rec._logger.handlers = [self.handler]
        self.rec._logger.propagate = False
        self.rec._logger.setLevel(logging.INFO)

    def tearDown(self):
        self.handler.close()
        self._tmp.cleanup()

    def read(self):
        self.handler.flush()
        return list(events.read_stream(self.path))

    def test_final_is_normalized_to_snake_case(self):
        sink = events.EventSink(self.rec)
        self.rec.bind(item="a")
        self.rec.clock.start()
        asyncio.run(sink.send(json.dumps(SERVER_FINAL)))
        stream = self.read()
        self.assertEqual(len(stream), 1)
        event = stream[0]
        self.assertEqual(event["type"], "final")
        self.assertEqual(event["item"], "a")
        self.assertEqual(event["commit_reason"], "seg")
        self.assertEqual(event["decision_audio_sec"], 2.94)
        self.assertEqual(event["segment_id"], 1)
        self.assertEqual(event["audio_end_sec"], 2.94)
        self.assertEqual(event["fsl_sec"], 0.141)
        self.assertEqual(event["chunk_encode_log"][0]["chunk_id"], 0)
        self.assertNotIn("commitReason", event)
        self.assertIsInstance(event["t"], float)

    def test_sink_collects_finals_and_vad_dones(self):
        sink = events.EventSink(self.rec)
        self.rec.clock.start()
        asyncio.run(sink.send(json.dumps(SERVER_FINAL)))
        asyncio.run(sink.send(json.dumps({"type": "vad_done", "has_remaining": False})))
        asyncio.run(sink.send(json.dumps({"type": "partial", "text": "mis", "seq": 1})))
        self.assertEqual(len(sink.finals), 1)
        self.assertEqual(len(sink.vad_dones), 1)
        self.assertEqual(sink.partials, 1)
        self.assertFalse(sink.vad_dones[0]["has_remaining"])

    def test_reset_item_clears_per_item_state(self):
        sink = events.EventSink(self.rec)
        self.rec.clock.start()
        asyncio.run(sink.send(json.dumps(SERVER_FINAL)))
        sink.reset_item()
        self.assertEqual(sink.finals, [])
        self.assertEqual(sink.vad_dones, [])

    def test_server_log_tags_become_events(self):
        self.rec.clock.start()
        self.rec.bind(item="b")
        log = logging.getLogger("bench.test.event")
        log.info("[SILENCE-DROP] reason=vad span=(1.0~2.0) text='uh'")
        log.info("plain message with no tag")
        stream = self.read()
        notes = [e for e in stream if e["type"] == "server_note"]
        self.assertEqual(len(notes), 1)
        self.assertEqual(notes[0]["tag"], "SILENCE-DROP")
        self.assertIn("reason=vad", notes[0]["msg"])
        self.assertEqual(notes[0]["item"], "b")
        self.assertEqual(len([e for e in stream if e["type"] == "log"]), 1)

    def test_malformed_payload_is_recorded_not_swallowed(self):
        sink = events.EventSink(self.rec)
        self.rec.clock.start()
        asyncio.run(sink.send("not json"))
        self.assertEqual(self.read()[0]["type"], "sink_error")

    def test_clock_is_per_item(self):
        self.rec.clock.start()
        first = self.rec.emit("item_open")["t"]
        self.rec.clock.stop()
        self.assertIsNone(self.rec.emit("between")["t"])
        self.rec.clock.start()
        second = self.rec.emit("item_open")["t"]
        self.assertIsNotNone(first)
        self.assertLess(second, 0.5)


class TestHandlerAttribution(unittest.TestCase):
    """Drives the real ASTStreamingHandler.send_message without a model."""

    @classmethod
    def setUpClass(cls):
        from bench import driver
        cls.driver = driver
        cls.ast_server, cls.base_server = driver._import_server()
        cls.BenchHandler = driver.make_handler_class(cls.ast_server)

    def make_handler(self, sink, ctx, *, utt_id="a"):
        import time as _time
        h = object.__new__(self.BenchHandler)
        h.websocket = sink
        h._bench_sink = sink
        h._bench_ctx = ctx
        h._bench_pending = {}
        h.bench_item_id = utt_id
        h.log = logging.getLogger("bench.test.handler")
        h._last_partial_text = None
        h.current_time = 3.0
        h.stream_start_perf = _time.perf_counter()
        h.config = self.base_server.StreamingConfig(**make_cfg().streaming_config_kwargs())
        h.gpt_translator = None
        h._committed_utterance_count = 0
        return h

    def test_subclass_chain_is_the_instrumented_handler(self):
        self.assertTrue(issubclass(self.BenchHandler, self.ast_server.ASTStreamingHandler))
        mro = [c.__name__ for c in self.BenchHandler.__mro__]
        self.assertIn("FSLStreamingHandler", mro)
        self.assertIn("Qwen3ASRStreamingHandler", mro)

    def test_utt_id_comes_from_the_driver(self):
        rec = events.Recorder()
        sink = events.EventSink(rec)
        h = self.make_handler(sink, self.driver.Context(recorder=rec, src_lang="en"),
                              utt_id="item-7")
        self.assertEqual(h._ast_utt_id, "item-7")

    def test_attribution_fields_reach_the_final_event(self):
        rec = events.Recorder()
        rec.clock.start()
        sink = events.EventSink(rec)
        ctx = self.driver.Context(recorder=rec, src_lang="en")
        h = self.make_handler(sink, ctx, utt_id="item-7")

        h._bench_reset_commit()
        ctx.commit_attr["target_lang"] = "ko"
        # stamp as _emit_final_payload would, then send
        h._bench_pending[SERVER_FINAL["original"]] = dict(ctx.commit_attr)
        asyncio.run(h.send_message("final", **{k: v for k, v in SERVER_FINAL.items()
                                               if k != "type"}))
        self.assertEqual(len(sink.finals), 1)
        event = sink.finals[0]
        self.assertEqual(event["target_lang"], "ko")
        self.assertEqual(event["translator"], "none")
        self.assertFalse(event["direction_refixed"])
        self.assertEqual(event["utt_id"], "item-7")
        self.assertEqual(event["sent_at_utt_id"], "item-7")

    def test_gpt_skip_on_first_commit_is_recorded(self):
        rec = events.Recorder()
        sink = events.EventSink(rec)
        ctx = self.driver.Context(recorder=rec, src_lang="en")
        h = self.make_handler(sink, ctx)
        h.config = self.base_server.StreamingConfig(
            **make_cfg(stity={"transcription": {"name": "qwen3"},
                              "translation": {"name": "gpt"},
                              "commit": "seg",
                              "gpu_memory_utilization": 0.5}).streaming_config_kwargs())
        h.gpt_translator = object()

        h._committed_utterance_count = 0
        h._bench_reset_commit()
        self.assertEqual(ctx.commit_attr["translator"], "google_or_local")

        h._committed_utterance_count = 1
        h._bench_reset_commit()
        self.assertEqual(ctx.commit_attr["translator"], "gpt")


class TestReportChain(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.events_path = self.dir / "evtest-20260911T120000.jsonl"
        self.items_path = self.dir / "evtest-20260911T120000.items.jsonl"
        self.cfg = make_cfg()

    def tearDown(self):
        self._tmp.cleanup()

    def rows(self):
        def seg(**kw):
            base = {"segment_id": 1, "original": "", "translation": "", "language": "en",
                    "target_lang": "ko", "commit_reason": "seg", "fsl_sec": 0.1,
                    "decision_audio_sec": 1.0, "recv_elapsed_sec": 1.1,
                    "translator": "none", "direction_refixed": False}
            base.update(kw)
            return base
        return [
            {"id": "good", "status": "ok", "group": "conv-1", "src_lang": "en",
             "speaker": "S1", "duration_sec": 2.0, "audio_sec": 2.0,
             "reference": "hello world", "hypothesis": "hello world",
             "reference_translations": {"ko": "안녕"}, "wer": 0.0, "cer": 0.0,
             "n_segments": 1, "avg_fsl_sec": 0.1, "n_foreign_finals": 0,
             "segments": [seg(original="hello world")]},
            {"id": "bad", "status": "ok", "group": "conv-1", "src_lang": "fr",
             "speaker": "S2", "duration_sec": 2.0, "audio_sec": 2.0,
             "reference": "bonjour le monde", "hypothesis": "bonjour",
             "reference_translations": {"ko": "안녕"}, "wer": 0.667, "cer": 0.5,
             "n_segments": 1, "avg_fsl_sec": 0.2, "n_foreign_finals": 0,
             "segments": [seg(original="bonjour", language="fr")]},
            {"id": "broken", "status": "error", "error": "audio_load_failed",
             "group": "conv-2", "src_lang": "en", "speaker": "S1",
             "duration_sec": 1.0, "audio_sec": 0.0, "reference": "good bye",
             "hypothesis": "", "reference_translations": {}, "segments": []},
        ]

    def write_events(self):
        rec = events.Recorder()
        handler = logging.FileHandler(self.events_path, encoding="utf-8")
        handler.setFormatter(events.JsonlFormatter(rec))
        log = logging.getLogger("bench.test.report")
        log.handlers = [handler]
        log.propagate = False
        log.setLevel(logging.INFO)
        rec._logger = log
        for item in ("good", "bad", "broken"):
            rec.bind(item=item)
            rec.clock.start()
            rec.emit("item_open", audio_sec=2.0)
            rec.emit("chunk", audio_sec=0.2, silence=False)
            rec.emit("final", output=f"text for {item}")
            rec.emit("item_close", n_finals=1)
            rec.clock.stop()
        handler.flush()
        handler.close()

    def test_item_writer_is_append_and_resumable(self):
        writer = report.ItemWriter(self.items_path, resume=True)
        writer.write({"id": "a"})
        writer.write({"id": "b"})
        writer.close()
        again = report.ItemWriter(self.items_path, resume=True)
        self.assertEqual(again.done_ids, {"a", "b"})
        self.assertEqual(len(again.read_back()), 2)
        again.close()

    def test_top_k_puts_failures_first(self):
        chosen = report.select_top_k(self.rows(), rank_by="wer", order="worst", top_k=2)
        self.assertEqual(chosen[0]["id"], "broken")
        self.assertEqual(chosen[1]["id"], "bad")

    def test_top_k_is_worst_first_among_scored(self):
        rows = [r for r in self.rows() if r["status"] == "ok"]
        chosen = report.select_top_k(rows, rank_by="wer", order="worst", top_k=2)
        self.assertEqual([r["id"] for r in chosen], ["bad", "good"])

    def test_replay_correlates_events_back_to_items(self):
        self.write_events()
        chosen = report.select_top_k(self.rows(), rank_by="wer", order="worst", top_k=2)
        out = self.events_path.with_suffix(".replay.json")
        report.write_replay(self.events_path, out, chosen, cfg=self.cfg, stamp="s")
        payload = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(payload["timestamp_origin"], "first_audio_chunk_of_item")
        self.assertEqual([d["item"] for d in payload["data"]], ["broken", "bad"])
        for entry in payload["data"]:
            self.assertTrue(entry["events"])
            self.assertTrue(all(e["item"] == entry["item"] for e in entry["events"]))
            self.assertEqual(entry["events"][0]["type"], "item_open")
        # the requested {timestamp, type, output} shape survives
        finals = [e for e in payload["data"][0]["events"] if e["type"] == "final"]
        self.assertIn("output", finals[0])
        self.assertIn("t", finals[0])

    def test_write_all_emits_summary_and_flags_degraded(self):
        self.write_events()
        out = report.write_all(
            cfg=self.cfg, dataset=FakeDataset(), rows=self.rows(),
            stamp="20260911T120000", status="ok",
            started=datetime(2026, 9, 11, 12, 0, tzinfo=timezone.utc),
            finished=datetime(2026, 9, 11, 12, 5, tzinfo=timezone.utc),
            events_path=self.events_path, items_path=self.items_path,
            usage={"calls": 0}, translation={"backend": "none"},
            transcription={"backend": "qwen3"})
        payload = json.loads(Path(out).read_text(encoding="utf-8"))
        self.assertEqual(payload["status"], "degraded")  # one item errored
        self.assertEqual(payload["counts"]["n_errored"], 1)
        self.assertIn("wer", payload["metrics"])
        self.assertIn("resolved", payload["config"])
        self.assertIn("commit", payload["config"]["resolved"]["stity"])
        self.assertTrue(payload["diagnostics"]["warnings"])
        self.assertEqual(payload["dataset"]["name"], "tiny")
        Path(out).unlink()

    def test_errored_item_does_not_inflate_the_score(self):
        aggregate, _ = metrics.run_metrics(self.rows(), self.cfg)
        # 'broken' is excluded; 'good' 0/2 errors and 'bad' 2/3 -> 2/5
        self.assertAlmostEqual(aggregate["wer"], 0.4)


class TestLoggingOrder(unittest.TestCase):
    def test_server_configure_logging_would_clear_our_handler(self):
        """Why events.configure must run before the server module is configured.

        _configure_logging does root.handlers.clear(). bench never calls it, but
        this pins the reason the ordering is not incidental.
        """
        from bench import driver
        _ast, base_server = driver._import_server()
        root = logging.getLogger()
        saved = list(root.handlers)
        try:
            marker = logging.NullHandler()
            root.addHandler(marker)
            self.assertIn(marker, root.handlers)
            with tempfile.TemporaryDirectory() as tmp:
                base_server._configure_logging(use_json=True,
                                               log_file=str(Path(tmp) / "s.log"))
            self.assertNotIn(marker, root.handlers)
        finally:
            root.handlers = saved

    def test_configure_attaches_a_jsonl_handler(self):
        root = logging.getLogger()
        saved = list(root.handlers)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "e.jsonl"
                handler = events.configure(path)
                self.assertIn(handler, root.handlers)
                logging.getLogger("bench.somewhere").info("[AST-LATE] x")
                handler.flush()
                stream = list(events.read_stream(path))
                self.assertTrue(any(e.get("tag") == "AST-LATE" for e in stream))
                handler.close()
        finally:
            root.handlers = saved


if __name__ == "__main__":
    unittest.main(verbosity=2)
