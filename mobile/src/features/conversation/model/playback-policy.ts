import type { OutputMode } from '@/domain/output-modes';
import type { OutputRoute } from '@/services/audio/audio-output';

type Pair = { myLanguage: string; peerLanguage: string };

/** What a sentence in `detected` should be translated into: the other side's language. */
export const translationTarget = (detected: string, { myLanguage, peerLanguage }: Pair): string =>
  detected === myLanguage ? peerLanguage : myLanguage;

/** Output used for the whole session; "both" switches per utterance in planPlayback. */
export const sessionRoute = (mode: OutputMode): OutputRoute =>
  mode === 'earphone' ? 'earpiece' : 'speaker';

/**
 * Decides whether and where a translation of speech in `detected` is spoken.
 * - speaker: every translation, through the speaker.
 * - earphone: only the peer's speech, translated into my language, in my earphone.
 * - both: translations into my language go to my earphone; into the peer's, to the speaker.
 */
export const planPlayback = (
  detected: string,
  pair: Pair,
  mode: OutputMode,
): { language: string; route: OutputRoute } | null => {
  const language = translationTarget(detected, pair);
  switch (mode) {
    case 'speaker':
      return { language, route: 'speaker' };
    case 'earphone':
      return detected === pair.myLanguage ? null : { language, route: 'earpiece' };
    case 'both':
      return { language, route: language === pair.myLanguage ? 'earpiece' : 'speaker' };
  }
};
