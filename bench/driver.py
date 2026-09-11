"""The in-process driver. The only module that imports the ASR server.

There is no WebSocket server, no port, and no subprocess. The handler's sole
egress is `await self.websocket.send(...)`, so an object with one `send` method
stands in for the socket; the other two socket touchpoints live in handle(), which
is never called.

Model loading reuses Qwen3ASRStreamingServer.init_model() verbatim -- ~90 lines of
vLLM, SamplingParams, LoRA and VAD-bytes setup that would otherwise be forked and
drift. start() is never called: it opens a listening socket. Four things that live
in start()/handle_connection() are therefore done here by hand, and the third one
matters most -- skipping _ast_hide_seg_token() silently mixes a SEG-producing
checkpoint into the punct/static axes.
"""
import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

from . import config as config_mod
from . import cost, events, langs, metrics, paths, report
from .data import audio as audio_mod
from .data import manifest
from .errors import BenchConfigError, BenchDataError

logger = logging.getLogger("bench.driver")

SAMPLING_RATE = 16000


@dataclass
class Context:
    """Shared mutable state the components write into."""
    recorder: events.Recorder
    src_lang: str
    usage_log: cost.UsageLog | None = None
    commit_attr: dict = field(default_factory=dict)


def _import_server():
    """Imported lazily and after logging is configured.

    Two modules matter. `ast_server` supplies the handler and server classes;
    `base_server` is where the injection points live -- StreamingConfig,
    set_local_translator, set_google_translate_api_key, warmup_streaming, and the
    VAD module globals. The FSL layer in between is reached only to get there, so
    it is not returned.

    evaluation/streaming_websocket_server_ast.py does its own sys.path inserts and
    binds streaming_websocket_server_fsl and trans_guard as top-level module names.
    Never import those under a second name -- trans_guard keeps module-level config
    and stats, and two copies would split them.
    """
    import evaluation.streaming_websocket_server_ast as ast_server
    base_server = ast_server.fsl_server.base_server
    return ast_server, base_server


def make_handler_class(ast_server):
    """Built at call time so importing this module does not import torch."""

    class BenchHandler(ast_server.ASTStreamingHandler):
        """ASTStreamingHandler with two additions and no behaviour changes.

        The utterance id comes from the driver instead of a sniffed `start` frame,
        and each committed segment is stamped with which translator actually ran
        and where it was routed.
        """

        def __init__(self, *args, sink, ctx, **kwargs):
            super().__init__(sink, *args, **kwargs)
            self._bench_sink = sink
            self._bench_ctx = ctx
            self.bench_item_id = None
            self._bench_pending: dict[str, dict] = {}

        @property
        def _ast_utt_id(self):
            """Overrides ast.py's property, which reads the sniffer.

            Nothing feeds a `start` frame through the sniffer in-process, so
            without this the late-final guard would compare None to None and
            cross-item leakage would be undetectable.
            """
            return self.bench_item_id

        def _bench_reset_commit(self) -> None:
            attr = self._bench_ctx.commit_attr
            attr.clear()
            # gpt_translator is skipped for the first commit of every stream
            # (_committed_utterance_count > 0), so a per-clip dataset translates
            # segment 1 of every item with Google. Record the default, and let the
            # wrapped GPT call overwrite it if it actually ran.
            if self.gpt_translator is not None and self._committed_utterance_count > 0:
                attr["translator"] = "gpt"
            elif self.config.no_translation:
                attr["translator"] = "none"
            else:
                attr["translator"] = "google_or_local"
            attr["direction_refixed"] = False

        def _maybe_fix_direction(self, detected, used_target):
            out = super()._maybe_fix_direction(detected, used_target)
            if out:
                self._bench_ctx.commit_attr["direction_refixed"] = True
            return out

        async def _translate(self, text, target_lang, audio_end_sec=None, context=None):
            self._bench_ctx.commit_attr["target_lang"] = target_lang
            return await super()._translate(text, target_lang, audio_end_sec,
                                            context=context)

        async def _correct_and_translate(self, text, current_lang, audio_end_sec):
            self._bench_reset_commit()
            return await super()._correct_and_translate(text, current_lang, audio_end_sec)

        async def _emit_final_payload(self, **kwargs):
            # Keyed by the original text because a deferred SEG emit can be
            # flushed after a later commit has already overwritten commit_attr.
            original = kwargs.get("original") or ""
            if original:
                self._bench_pending[original] = dict(self._bench_ctx.commit_attr)
            return await super()._emit_final_payload(**kwargs)

        async def send_message(self, msg_type: str, **kwargs):
            if msg_type == "final":
                attr = self._bench_pending.pop(kwargs.get("original") or "", None)
                if attr:
                    for key, value in attr.items():
                        kwargs.setdefault(key, value)
            return await super().send_message(msg_type, **kwargs)

    return BenchHandler


class Engine:
    """Owns the model for the whole run. Handlers are rebuilt; the model is not."""

    def __init__(self, cfg: config_mod.BenchConfig, ctx: Context):
        self.cfg = cfg
        self.ctx = ctx
        self.base_server = None
        self.server = None
        self.handler_cls = None
        self.http_session = None
        self.attachment = None
        self.transcription = None

    def load(self) -> None:
        from .components import transcription as transcription_mod
        from .components import translation as translation_mod

        ast_server, base_server = _import_server()
        self.base_server = base_server
        self.handler_cls = make_handler_class(ast_server)

        st = self.cfg.stity

        # Module globals. VADIterator reads VAD_MIN_SILENCE_MS at construction and
        # _trim_tail_silence reads it again, so it must be set before any handler.
        base_server.VAD_MIN_SILENCE_MS = st.vad.min_silence_ms
        base_server.VAD_THRESHOLD = st.vad.threshold
        base_server.VAD_SPEECH_PAD_MS = st.vad.speech_pad_ms
        # HIDE_SEG is a module global that --ast-hide-seg normally sets.
        ast_server.HIDE_SEG = st.commit.hide_seg

        self.transcription = transcription_mod.build(
            st.transcription.name, st.transcription.options, self.ctx)
        backend = translation_mod.build(st.translation.name, st.translation.options, self.ctx)
        self.attachment = backend.attach(base_server)

        kwargs = self.cfg.streaming_config_kwargs()
        kwargs.update(self.transcription.streaming_config_kwargs())
        kwargs.update(self.attachment.config_overrides)
        streaming_config = base_server.StreamingConfig(**kwargs)

        logger.info("loading model: %s", streaming_config.model_path)
        self.server = ast_server.ASTStreamingServer(streaming_config)
        self.server.init_model()

        # Done by start() / handle_connection(), which are never called here.
        self.server._ast_hide_seg_token()
        self.server._ast_install_cap_freeze()

        self.attachment.verify(self.server)
        self._log_gpu("after init_model")

    async def start_async(self) -> None:
        import aiohttp

        await self.base_server.warmup_streaming(self.server.asr)
        self.http_session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=3))
        self.server._http_session = self.http_session
        self._log_gpu("after warmup")

    async def close_async(self) -> None:
        if self.http_session is not None:
            await self.http_session.close()
            self.http_session = None
        if self.attachment is not None:
            self.attachment.teardown()

    def make_handler(self):
        return self.handler_cls(
            self.server.asr, self.server.config,
            sink=events.EventSink(self.ctx.recorder),
            ctx=self.ctx,
            http_session=self.http_session,
            vad_model_bytes=self.server.vad_model_bytes,
            corrector=self.server.corrector,
            **self.attachment.handler_kwargs,
        )

    def _log_gpu(self, when: str) -> None:
        try:
            import torch
            if not torch.cuda.is_available():
                return
            free, total = torch.cuda.mem_get_info()
            free_gib, total_gib = round(free / 2**30, 2), round(total / 2**30, 2)
            events.emit("gpu_memory", when=when, free_gib=free_gib, total_gib=total_gib)
            logger.info("GPU %s: %.2f GiB free of %.2f GiB", when, free_gib, total_gib)
        except Exception:  # noqa: BLE001 - diagnostics only
            pass


def _prepare_handler(handler, *, item, cfg):
    """What handle() does, minus everything socket-shaped."""
    languages = cfg.languages
    if languages.mode == "map":
        handler.client_lang = "auto"
        handler.client_target_lang = languages.map.get(item.src_lang, "")
        handler.client_lang_map = dict(languages.map)
    else:
        handler.client_lang = languages.lang
        handler.client_target_lang = languages.target
        handler.client_lang_map = {}
    handler.session_logger = None
    handler.recorder = None
    handler.init_streaming_state()
    handler.running = True


async def _stream_item(handler, item, cfg, dataset, rec) -> dict:
    """One item: paced chunks, then finish. Returns the row."""
    trailing_ms = (cfg.pacing.trailing_silence_ms
                   if cfg.pacing.trailing_silence_ms is not None
                   else int(dataset.bench_defaults.get("trailing_silence_ms", 1000)))
    sink = handler._bench_sink
    sink.reset_item()
    handler.bench_item_id = item.id

    try:
        audio = audio_mod.load_window(item.audio, dataset.audio_format,
                                      offset=item.offset, duration=item.duration,
                                      sample_rate=dataset.sample_rate)
    except BenchDataError as e:
        rec.clock.start()
        events.emit("item_error", error="audio_load_failed", detail=str(e))
        return {"id": item.id, "status": "error", "error": str(e),
                "src_lang": item.src_lang, "speaker": item.speaker,
                "duration_sec": item.duration, "reference": item.transcript,
                "hypothesis": "", "reference_translations": dict(item.translations),
                "segments": []}

    pcm = audio_mod.to_pcm_bytes(audio)
    bytes_per_chunk = max(2, int(SAMPLING_RATE * cfg.pacing.chunk_size_ms / 1000) * 2)
    audio_sec = len(pcm) / 2 / SAMPLING_RATE

    rec.clock.start()
    events.emit("item_open", audio_sec=round(audio_sec, 3), src_lang=item.src_lang,
                target_lang=cfg.languages.expected_target(item.src_lang),
                trailing_silence_ms=trailing_ms,
                started_at=datetime.now(timezone.utc).isoformat())

    # One absolute origin, one target per chunk. Cumulative sleeps drift.
    origin = time.perf_counter()
    sent_samples = 0

    async def feed(chunk: bytes, *, silence: bool) -> None:
        nonlocal sent_samples
        sent_samples += len(chunk) // 2
        if cfg.pacing.realtime:
            delay = origin + sent_samples / SAMPLING_RATE - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
        await handler.process_audio_chunk(chunk)
        events.emit("chunk", audio_sec=round(sent_samples / SAMPLING_RATE, 3),
                    silence=silence)

    for start in range(0, len(pcm), bytes_per_chunk):
        await feed(pcm[start:start + bytes_per_chunk], silence=False)

    # Trailing silence gives VAD a chance to close the utterance. Aborted as soon
    # as VAD reports it has nothing left, so short clips do not pay the full wait.
    silence_total = 0
    vad_seen = len(sink.vad_dones)
    while silence_total < trailing_ms:
        step = min(cfg.pacing.chunk_size_ms, trailing_ms - silence_total)
        silence_total += step
        await feed(audio_mod.silence_bytes(step, dataset.sample_rate), silence=True)
        if any(not d.get("has_remaining") for d in sink.vad_dones[vad_seen:]):
            events.emit("silence_aborted", after_ms=silence_total)
            break
        vad_seen = len(sink.vad_dones)

    await handler.finish_streaming()

    # finish_streaming drains GPT twice; anything still in flight would leak into
    # the next item, so it is recorded rather than ignored.
    inflight = getattr(handler, "_ast_flush_inflight", 0)
    pending = len(getattr(handler, "_pending_gpt_tasks", []) or [])

    segments = [dict(f) for f in sink.finals]
    own = [s for s in segments
           if not s.get("utt_id") or s.get("utt_id") == item.id]
    foreign = len(segments) - len(own)

    hypothesis = " ".join((s.get("original") or "").strip() for s in own).strip()
    hyp_translation = " ".join((s.get("translation") or "").strip() for s in own).strip()

    events.emit("item_close", n_finals=len(own), n_foreign_finals=foreign,
                empty_hypothesis=not hypothesis, flush_inflight=inflight,
                pending_gpt=pending)
    rec.clock.stop()

    return {
        "id": item.id,
        "status": "ok",
        "group": item.group,
        "src_lang": item.src_lang,
        "speaker": item.speaker,
        "duration_sec": item.duration,
        "audio_sec": round(audio_sec, 3),
        "reference": item.transcript,
        "hypothesis": hypothesis,
        "hypothesis_translation": hyp_translation,
        "reference_translations": dict(item.translations),
        "n_foreign_finals": foreign,
        "flush_inflight": inflight,
        "segments": own,
    }


def _finalize_row(row, cfg) -> dict:
    """Per-item metric fields, so ranking and the replay selection have something
    to sort on."""
    if row.get("status") != "ok":
        return row
    segments = row.get("segments") or []
    target = cfg.languages.expected_target(row["src_lang"])

    row["wer"] = metrics.asr.per_item_wer(row["reference"], row["hypothesis"])
    row["cer"] = metrics.asr.per_item_cer(row["reference"], row["hypothesis"])
    row["n_segments"] = len(segments)
    row.update(metrics.latency.fsl_stats(
        segments, min_silence_ms=cfg.stity.vad.min_silence_ms))
    row["laal"] = metrics.latency.laal_for_item(
        segments,
        src_duration_sec=row.get("duration_sec") or 0.0,
        ref_text=(row.get("reference_translations") or {}).get(target, ""),
        unit=langs.laal_unit(target),
    )
    row.update(row["laal"])
    row["route_errors"] = sum(
        1 for s in segments if (s.get("target_lang") or "").lower() not in ("", target))
    return row


async def _run(cfg, dataset, engine, rec, writer) -> list[dict]:
    rows: list[dict] = []
    handler = None
    current_group = None
    await engine.start_async()
    try:
        for index, item in enumerate(dataset.items, start=1):
            rec.bind(item=item.id, session=item.group)
            if item.group != current_group:
                # A new handler per group. init_streaming_state() does not reset
                # six attributes set only in __init__ (vad_speech_spans,
                # _last_final_end_sec, _last_final_text, _deferred_fragment,
                # _gpt_flush_tasks, vad_last_speech_start_sample), so reusing one
                # handler across unrelated items leaks the previous item's
                # coordinates and trailing text into the next.
                current_group = item.group
                handler = engine.make_handler()
                events.emit("session_open", group=item.group)
            _prepare_handler(handler, item=item, cfg=cfg)
            row = await _stream_item(handler, item, cfg, dataset, rec)
            row = _finalize_row(row, cfg)
            rows.append(row)
            writer.write(row)
            logger.info("[%d/%d] %s wer=%s segments=%d", index, len(dataset.items),
                        item.id,
                        "-" if row.get("wer") is None else f"{row['wer']:.3f}",
                        len(row.get("segments") or []))
    finally:
        await engine.close_async()
    return rows


def main(cfg: config_mod.BenchConfig, *, resume: bool = True) -> int:
    started = datetime.now(timezone.utc)
    stamp = started.strftime("%Y%m%dT%H%M%S")
    paths.ensure_output_dirs()

    dataset = manifest.load(cfg.dataset.name, limit=cfg.dataset.limit)
    metrics.check_requirements(cfg.metrics, dataset=dataset, languages=cfg.languages,
                               realtime=cfg.pacing.realtime)
    uncovered = cfg.languages.covers(dataset.languages)
    if uncovered:
        raise BenchConfigError(
            f"dataset {dataset.name!r} contains {uncovered} but the run's languages "
            f"only cover {cfg.languages.source_langs}. Utterances in an uncovered "
            f"language would be decoded under a restricted language set, so the "
            f"score would measure the config, not the model."
        )

    events_path = paths.LOGS_DIR / f"{cfg.name}-{stamp}.jsonl"
    items_path = paths.ITEMS_DIR / f"{cfg.name}-{stamp}.jsonl"
    usage_path = paths.LOGS_DIR / f"{cfg.name}-{stamp}.usage.jsonl"

    # Before the server module is imported: its _configure_logging clears root
    # handlers, and anything attached afterwards would be dropped.
    events.configure(events_path)
    rec = events.recorder()
    rec.bind(run=cfg.name)

    usage_log = cost.UsageLog(usage_path)
    ctx = Context(recorder=rec, src_lang=dataset.languages[0], usage_log=usage_log)

    events.emit("run_open", name=cfg.name, fingerprint=cfg.fingerprint(),
                dataset=dataset.name, n_items=len(dataset.items),
                n_sessions=dataset.n_sessions, started_at=started.isoformat())

    engine = Engine(cfg, ctx)
    engine.load()

    writer = report.ItemWriter(items_path, resume=resume)
    status = "ok"
    try:
        rows = asyncio.run(_run(cfg, dataset, engine, rec, writer))
    except KeyboardInterrupt:
        logger.warning("interrupted; scoring what was written so far")
        rows = writer.read_back()
        status = "degraded"
    finally:
        writer.close()
        usage_log.close()

    rec.bind(item="", session="")
    events.emit("run_close", n_rows=len(rows))

    report.write_all(
        cfg=cfg, dataset=dataset, rows=rows, stamp=stamp, status=status,
        started=started, finished=datetime.now(timezone.utc),
        events_path=events_path, items_path=items_path,
        usage=usage_log.summary(),
        translation=dict(engine.attachment.provenance,
                         fingerprint=engine.attachment.fingerprint,
                         stats=engine.attachment.stats()),
        transcription=engine.transcription.provenance(),
    )
    return 0
