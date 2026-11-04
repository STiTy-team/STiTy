from core.components.registry import Component, Transcribed, Translated
from core.utils import logging, timing
from core.utils.audio import MultiChannelAudio

log = logging.getLogger(__name__)


def start(parts: dict[str, Component], *, languages: list[str], target_lang: str = "") -> None:
    detector = parts.get("vad")
    for part in parts.values():
        part.start(languages=languages, target_lang=target_lang,
                   vad=None if part is detector else detector)


async def close(parts: dict[str, Component]) -> None:
    for part in parts.values():
        try:
            await part.close()
        except Exception as e:  # noqa: BLE001
            log.warning("[CLOSE-FAILED] %s: %s", type(part).__name__, e)


async def listen(parts: dict[str, Component], audio: MultiChannelAudio) -> list:
    return await hear(parts, parts["mixer"].mix(audio))


async def hear(parts: dict[str, Component], pcm: bytes) -> list:
    transcriber, detector = parts["transcription"], parts.get("vad")
    speech = None if detector is None else detector.detect(pcm)
    produced = [] if speech is None else [speech]
    produced.extend(await transcriber.transcribe(pcm))
    if speech is not None:
        produced.extend(await transcriber.flush("vad", speech))
    return produced


async def finish(parts: dict[str, Component]) -> list:
    return await parts["transcription"].finish()


async def translate(parts: dict[str, Component], item: Transcribed, *, target_lang: str,
                    context: list[str], allowed: bool = True) -> Translated:
    corrector, translator = parts.get("correction"), parts.get("translation")
    original = item.original
    if corrector is not None:
        original = await corrector.correct(original, item.language)

    translation, detected = "", ""
    if item.language and item.language == target_lang:
        translation = original
    elif translator is not None and allowed:
        translation, detected = await translator.translate(
            original, target_lang, item.language or None, context=context)

    return Translated(
        original=original,
        translation=translation,
        language=item.language or detected or "",
        target_lang=target_lang,
        commit_reason=item.commit_reason,
        decision_audio_sec=item.decision_audio_sec,
        committed_elapsed_sec=item.committed_elapsed_sec,
        translated_elapsed_sec=timing.elapsed(),
    )


def untranslated(item: Transcribed, target_lang: str) -> Translated:
    return Translated(
        original=item.original,
        translation="",
        language=item.language,
        target_lang=target_lang,
        commit_reason=item.commit_reason,
        decision_audio_sec=item.decision_audio_sec,
        committed_elapsed_sec=item.committed_elapsed_sec,
    )
