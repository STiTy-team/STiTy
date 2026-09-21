from __future__ import annotations

import argparse
import asyncio
import copy
import json
from datetime import datetime, timezone
from pathlib import Path

from core.utils import metrics
from core.utils.metrics import asr_robustness
from core.utils.metrics.metric_inputs import gold_inputs
from core.utils.metrics.text_noise import NOISE_VERSION, perturb_transcript


OUTPUT_FILES = ("items.jsonl", "summary.json", "robustness_config.json")


async def _translate(translator, *, text: str, source_lang: str,
                     target_lang: str) -> tuple[str, str]:
    if not text.strip():
        return "", source_lang
    value, detected = await translator.translate(
        text, target_lang, source_lang, context=[])
    return str(value or "").strip(), str(detected or source_lang).lower()


async def evaluate_robustness_rows(rows: list[dict], *, translator, languages,
                                   noise_levels: list[float], seed: int = 0,
                                   similarity_model: str = asr_robustness.DEFAULT_INVARIANCE_MODEL,
                                   similarity_device: str | None = None,
                                   similarity_batch_size: int = 32,
                                   similarity_scorer=None) -> list[dict]:
    """Translate each item's clean transcript, its real ASR output and synthetic variants.

    Clean is the gold transcript. Its real ASR output, when the row has one that
    differs, is the noisy condition behind ``quality_drop`` and ``similarity`` --
    the ASR-induced change the axis is named for. Synthetic variants of the clean
    transcript at each requested level give the degradation curve, placed at the WER
    each variant actually has.

    An empty ASR output or an empty translation is a catastrophic failure and is
    scored (quality of an empty string, similarity 0). A translator that raised is
    an infrastructure failure: that condition is left out of every statistic, so
    quality drop and invariance always cover the same pairs.
    """
    if not noise_levels or any(not 0 < value <= 1 for value in noise_levels):
        raise ValueError("noise_levels must contain values in (0, 1]")
    levels = sorted(set(float(value) for value in noise_levels))
    output = []
    # (pair id, clean translation, noisy translation, [(dict, key), ...] to fill)
    similarity_pairs = []

    for row_index, source_row in enumerate(rows):
        row = copy.deepcopy(source_row)
        row["metric_inputs"] = gold_inputs(row.get("metric_inputs"))
        if row.get("status") not in (None, "ok"):
            output.append(row)
            continue
        item_id = str(row.get("id") or row_index)
        source_lang = str(row.get("src_lang") or "").lower()
        if source_lang not in {"ko", "en"}:
            raise ValueError(f"ASR text robustness supports only ko/en, got {source_lang!r}")
        target_lang = languages.expected_target(source_lang)
        gold = str(row.get("reference") or "").strip()
        asr = str(row.get("hypothesis") or "").strip()
        clean_text = gold or asr
        reference = str((row.get("reference_translations") or {}).get(target_lang) or "").strip()
        if not clean_text:
            output.append(row)
            continue
        translator.start(language=source_lang)
        translation_errors = []
        try:
            clean_translation, detected = await _translate(
                translator, text=clean_text, source_lang=source_lang, target_lang=target_lang)
        except Exception as exc:
            row["robustness_translation_status"] = "failed_clean_translation"
            row["robustness_translation_errors"] = [
                {"condition": "clean", "error": type(exc).__name__, "detail": str(exc)}]
            output.append(row)
            continue

        async def condition(name: str, text: str) -> dict | None:
            try:
                translation, _ = await _translate(
                    translator, text=text, source_lang=source_lang, target_lang=target_lang)
            except Exception as exc:
                translation_errors.append({"condition": name, "error": type(exc).__name__,
                                           "detail": str(exc)})
                return None
            score = metrics.meaning.chrfpp_sentence(translation, reference)
            return {"transcript": text, "translation": translation,
                    "quality": {"chrfpp": score} if score is not None else {},
                    "wer_from_clean": asr_robustness.word_error_rate(clean_text, text),
                    "cer_from_clean": asr_robustness.character_error_rate(clean_text, text),
                    "catastrophic": not text or not translation}

        clean_score = metrics.meaning.chrfpp_sentence(clean_translation, reference)
        clean_quality = {"chrfpp": clean_score} if clean_score is not None else {}
        block = {
            "clean_source": "reference_transcript" if gold else "asr_hypothesis",
            "clean_transcript": clean_text,
            "clean_translation": clean_translation,
            "clean_quality": clean_quality,
            "noise_generator_version": NOISE_VERSION,
            "seed": seed,
            "detected_source_lang": detected,
        }

        if gold and asr != gold:
            real = await condition("asr", asr)
            if real is not None:
                block.update({"noisy_source": "asr_hypothesis",
                              "noisy_transcript": asr,
                              "noisy_translation": real["translation"],
                              "noisy_quality": real["quality"],
                              "noise_level": real["wer_from_clean"],
                              "cer_from_clean": real["cer_from_clean"],
                              "catastrophic": real["catastrophic"]})
                similarity_pairs.append((f"{item_id}@asr", clean_translation,
                                         real["translation"], [(block, "similarity")]))

        variants = []
        quality_by_noise = [{"noise_level": 0.0, "requested_noise_level": 0.0,
                             "quality": clean_quality}]
        for level in levels:
            noisy_text, operations = perturb_transcript(
                clean_text, lang=source_lang, level=level, seed=seed, item_id=item_id)
            pair_id = f"{item_id}@{level:g}"
            variant = {"pair_id": pair_id, "requested_noise_level": level,
                       "noisy_transcript": noisy_text, "operations": operations}
            variants.append(variant)
            synthetic = await condition(f"noise:{level:g}", noisy_text)
            if synthetic is None:
                variant["translation_failed"] = True
                continue
            point = {"noise_level": synthetic["wer_from_clean"],
                     "requested_noise_level": level,
                     "cer_from_clean": synthetic["cer_from_clean"],
                     "quality": dict(synthetic["quality"])}
            variant.update({"wer_from_clean": synthetic["wer_from_clean"],
                            "cer_from_clean": synthetic["cer_from_clean"],
                            "noisy_translation": synthetic["translation"],
                            "quality": synthetic["quality"],
                            "catastrophic": synthetic["catastrophic"]})
            quality_by_noise.append(point)
            similarity_pairs.append((pair_id, clean_translation, synthetic["translation"],
                                     [(variant, "similarity"), (point["quality"], "similarity")]))
        block.update({"quality_by_noise": quality_by_noise, "variants": variants})
        row.setdefault("metric_inputs", {})["asr_robustness"] = block
        row["hypothesis_translation"] = clean_translation
        row["robustness_translation_status"] = "degraded" if translation_errors else "ok"
        if translation_errors:
            row["robustness_translation_errors"] = translation_errors
        output.append(row)

    # Both outputs present: the encoder decides. Exactly one empty: nothing of the
    # clean meaning survived (0). Both empty: the output did not change (1).
    scoreable = [{"id": pair_id, "clean_translation": clean, "noisy_translation": noisy}
                 for pair_id, clean, noisy, _ in similarity_pairs if clean and noisy]
    similarities = {}
    if scoreable:
        if similarity_scorer is None:
            similarities = asr_robustness.multilingual_semantic_similarity(
                scoreable, model_name=similarity_model, device=similarity_device,
                batch_size=similarity_batch_size)["per_item"]
        else:
            similarities = similarity_scorer(scoreable)
    for pair_id, clean, noisy, targets in similarity_pairs:
        if clean and noisy:
            if pair_id not in similarities:
                continue
            value = float(similarities[pair_id])
        else:
            value = 0.0 if clean or noisy else 1.0
        for holder, key in targets:
            holder[key] = value
    for row in output:
        block = (row.get("metric_inputs") or {}).get("asr_robustness")
        if block:
            block["similarity_model"] = similarity_model
    return output


def score_robustness(rows: list[dict]) -> metrics.RunScore:
    items = [metrics.Utterance.from_row(row) for row in rows
             if row.get("status") in (None, "ok")]
    values, unavailable = asr_robustness.corpus(items)
    return metrics.RunScore(aggregate=values, unavailable=unavailable)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m bench.asr_text_robustness",
        description="저장된 전사문을 ASR처럼 교란하고 clean/noisy 번역 강건성을 평가합니다.",
    )
    parser.add_argument("source_run", help="원본 run 디렉터리 또는 items.jsonl")
    parser.add_argument("config", help="bench retranslate 형식의 번역기 config.yml")
    parser.add_argument("output", help="강건성 결과 디렉터리")
    parser.add_argument("--noise-levels", default="0.08,0.18,0.32")
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--similarity-model", default=asr_robustness.DEFAULT_INVARIANCE_MODEL)
    parser.add_argument("--similarity-device")
    parser.add_argument("--similarity-batch-size", type=int, default=32)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    from core.pipelines.translation import translators
    from core.utils import env
    from . import derived, report, translation_config

    env.load()
    cfg = translation_config.load(args.config)
    if {cfg.languages.lang, cfg.languages.target} != {"ko", "en"}:
        raise ValueError("ASR text robustness currently supports only ko<->en")
    levels = [float(value.strip()) for value in args.noise_levels.split(",") if value.strip()]
    source_run, source_items = derived.source_items(args.source_run)
    output_dir = Path(args.output).resolve()
    if output_dir == source_run:
        raise ValueError("robustness results must be written to a derived run directory")
    derived.prepare_output(output_dir, OUTPUT_FILES, overwrite=args.overwrite)
    rows = derived.read_rows(source_items)
    translator_cls = translators.get(cfg.translation.name)
    translator = translator_cls(cfg.translation.options, cfg=cfg)
    started = datetime.now(timezone.utc)

    async def execute():
        try:
            await translator.load()
            return await evaluate_robustness_rows(
                rows, translator=translator, languages=cfg.languages,
                noise_levels=levels, seed=args.seed,
                similarity_model=args.similarity_model,
                similarity_device=args.similarity_device,
                similarity_batch_size=args.similarity_batch_size)
        finally:
            await translator.close()

    evaluated = asyncio.run(execute())
    derived.write_rows(output_dir / "items.jsonl", evaluated)
    score = score_robustness(evaluated)
    finished = datetime.now(timezone.utc)
    translation_errors = sum(len(row.get("robustness_translation_errors") or [])
                             for row in evaluated)
    config_payload = {
        "noise_levels": levels,
        "seed": args.seed,
        "noise_generator_version": NOISE_VERSION,
        "similarity_model": args.similarity_model,
        "translation": {"backend": cfg.translation.name, **cfg.translation.options},
        "languages": {"lang": cfg.languages.lang, "target": cfg.languages.target},
    }
    summary = {
        "mode": "asr_text_robustness",
        "status": "degraded" if translation_errors else "ok",
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "source_run": {"path": str(source_run), "items_sha256": derived.sha256(source_items)},
        "counts": {"items": len(evaluated),
                   "evaluated": sum("asr_robustness" in (row.get("metric_inputs") or {})
                                    for row in evaluated),
                   "translation_errors": translation_errors},
        "metrics": score.aggregate,
        "unavailable": score.unavailable,
        "config": config_payload,
        "environment": report.environment(),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    (output_dir / "robustness_config.json").write_text(
        json.dumps(config_payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps({"output": str(output_dir), "metrics": score.aggregate,
                      "unavailable": score.unavailable}, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
