from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from core.utils import metrics
from core.utils.metrics import asr_robustness
from core.utils.metrics.text import levenshtein, normalize_words, strip_for_cer

from .retranslate import _gold_metric_inputs
from .text_noise import NOISE_VERSION, perturb_transcript


def _read_jsonl(path: Path) -> list[dict]:
    output = []
    with open(path, encoding="utf-8") as source:
        for lineno, line in enumerate(source, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{lineno} is not a JSON object")
            output.append(value)
    return output


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def _source_items(path: str | Path) -> tuple[Path, Path]:
    candidate = Path(path).resolve()
    items = candidate if candidate.is_file() else candidate / "items.jsonl"
    if not items.is_file():
        raise ValueError(f"items.jsonl not found: {items}")
    return items.parent, items


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _word_error_rate(reference: str, hypothesis: str) -> float:
    ref = normalize_words(reference)
    if not ref:
        return 0.0
    return levenshtein(ref, normalize_words(hypothesis)) / len(ref)


def _character_error_rate(reference: str, hypothesis: str) -> float:
    ref = strip_for_cer(reference)
    if not ref:
        return 0.0
    return levenshtein(ref, strip_for_cer(hypothesis)) / len(ref)


def _chrf_quality(candidate: str, reference: str) -> float | None:
    if not reference:
        return None
    from sacrebleu.metrics import CHRF
    return float(CHRF(char_order=6, word_order=2, beta=2)
                 .sentence_score(candidate, [reference]).score)


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
    if not noise_levels or any(not 0 < value <= 1 for value in noise_levels):
        raise ValueError("noise_levels must contain values in (0, 1]")
    levels = sorted(set(float(value) for value in noise_levels))
    output = []
    similarity_pairs = []

    for row_index, source_row in enumerate(rows):
        row = copy.deepcopy(source_row)
        row["metric_inputs"] = _gold_metric_inputs(row.get("metric_inputs"))
        if row.get("status") not in (None, "ok"):
            output.append(row)
            continue
        item_id = str(row.get("id") or row_index)
        source_lang = str(row.get("src_lang") or "").lower()
        if source_lang not in {"ko", "en"}:
            raise ValueError(f"ASR text robustness supports only ko/en, got {source_lang!r}")
        target_lang = languages.expected_target(source_lang)
        clean_text = str(row.get("hypothesis") or row.get("reference") or "").strip()
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
        clean_score = _chrf_quality(clean_translation, reference)
        clean_quality = {"chrfpp": clean_score} if clean_score is not None else {}
        variants = []
        quality_by_noise = [{"noise_level": 0.0, "requested_noise_level": 0.0,
                             "quality": clean_quality}]
        for level in levels:
            noisy_text, operations = perturb_transcript(
                clean_text, lang=source_lang, level=level, seed=seed, item_id=item_id)
            try:
                noisy_translation, _ = await _translate(
                    translator, text=noisy_text, source_lang=source_lang,
                    target_lang=target_lang)
            except Exception as exc:
                translation_errors.append({"condition": f"noise:{level:g}",
                                           "error": type(exc).__name__, "detail": str(exc)})
                noisy_translation = ""
            actual_wer = _word_error_rate(clean_text, noisy_text)
            actual_cer = _character_error_rate(clean_text, noisy_text)
            noisy_score = _chrf_quality(noisy_translation, reference)
            noisy_quality = {"chrfpp": noisy_score} if noisy_score is not None else {}
            pair_id = f"{item_id}@{level:g}"
            similarity_pairs.append({"id": pair_id,
                                     "clean_translation": clean_translation,
                                     "noisy_translation": noisy_translation})
            variants.append({
                "pair_id": pair_id,
                "requested_noise_level": level,
                "wer_from_clean": actual_wer,
                "cer_from_clean": actual_cer,
                "noisy_transcript": noisy_text,
                "noisy_translation": noisy_translation,
                "quality": noisy_quality,
                "operations": operations,
            })
            quality_by_noise.append({"noise_level": level,
                                     "requested_noise_level": level,
                                     "wer_from_clean": actual_wer,
                                     "cer_from_clean": actual_cer,
                                     "quality": noisy_quality})
        block = {
            "clean_transcript": clean_text,
            "clean_translation": clean_translation,
            "clean_quality": clean_quality,
            "quality_by_noise": quality_by_noise,
            "variants": variants,
            "noise_generator_version": NOISE_VERSION,
            "seed": seed,
            "detected_source_lang": detected,
        }
        worst = variants[-1]
        block.update({"noisy_transcript": worst["noisy_transcript"],
                      "noisy_translation": worst["noisy_translation"],
                      "noisy_quality": worst["quality"],
                      "noise_level": worst["requested_noise_level"]})
        row.setdefault("metric_inputs", {})["asr_robustness"] = block
        row["hypothesis_translation"] = clean_translation
        row["robustness_translation_status"] = "degraded" if translation_errors else "ok"
        if translation_errors:
            row["robustness_translation_errors"] = translation_errors
        output.append(row)

    scoreable_pairs = [pair for pair in similarity_pairs
                       if pair["clean_translation"] and pair["noisy_translation"]]
    if scoreable_pairs:
        if similarity_scorer is None:
            similarity_result = asr_robustness.multilingual_semantic_similarity(
                scoreable_pairs, model_name=similarity_model,
                device=similarity_device, batch_size=similarity_batch_size)
            similarities = similarity_result["per_item"]
        else:
            similarities = similarity_scorer(scoreable_pairs)
        for row in output:
            block = (row.get("metric_inputs") or {}).get("asr_robustness")
            if not block:
                continue
            for variant in block["variants"]:
                if variant["pair_id"] in similarities:
                    variant["similarity"] = float(similarities[variant["pair_id"]])
            worst = block["variants"][-1]
            if "similarity" in worst:
                block["similarity"] = worst["similarity"]
            block["similarity_model"] = similarity_model
    return output


def score_robustness(rows: list[dict]) -> metrics.RunScore:
    items = [metrics.Utterance.from_row(row) for row in rows
             if row.get("status") in (None, "ok")]
    values, unavailable = asr_robustness.corpus(items)
    return metrics.RunScore(aggregate=values, unavailable=unavailable)


def _prepare_output(path: Path, *, overwrite: bool) -> None:
    path.mkdir(parents=True, exist_ok=True)
    targets = [path / name for name in ("items.jsonl", "summary.json", "robustness_config.json")]
    existing = [target for target in targets if target.exists()]
    if existing and not overwrite:
        raise ValueError(f"output already exists; pass --overwrite: {path}")
    for target in existing:
        target.unlink()


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
    from . import config, report

    env.load()
    cfg = config.load_retranslate(args.config)
    if {cfg.languages.lang, cfg.languages.target} != {"ko", "en"}:
        raise ValueError("ASR text robustness currently supports only ko<->en")
    levels = [float(value.strip()) for value in args.noise_levels.split(",") if value.strip()]
    source_run, source_items = _source_items(args.source_run)
    output_dir = Path(args.output).resolve()
    if output_dir == source_run:
        raise ValueError("robustness results must be written to a derived run directory")
    _prepare_output(output_dir, overwrite=args.overwrite)
    rows = _read_jsonl(source_items)
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
    _write_jsonl(output_dir / "items.jsonl", evaluated)
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
        "source_run": {"path": str(source_run), "items_sha256": _sha256(source_items)},
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
