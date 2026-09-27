from core.components.registry import Component, Transcribed, Translated
from core.utils import timing
from core.utils.audio import MultiChannelAudio


async def listen(parts: dict[str, Component], audio: MultiChannelAudio) -> list:
    pcm = parts["mixer"].mix(audio)
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

    translation = ""
    if item.language and item.language == target_lang:
        translation = original
    elif translator is not None and allowed:
        translation, _ = await translator.translate(
            original, target_lang, item.language or None, context=context)

    return Translated(
        original=original,
        translation=translation,
        language=item.language,
        target_lang=target_lang,
        commit_reason=item.commit_reason,
        decision_audio_sec=item.decision_audio_sec,
        committed_elapsed_sec=item.committed_elapsed_sec,
        translated_elapsed_sec=timing.elapsed(),
    )
