import type { AsrSessionHandlers } from '@/api/live/ws/asr-session';

import type { ScriptLine } from './scripts';

export type ScriptTiming = {
  firstLineMs: number;
  /** `words` types a line out word by word; `line` shows it whole while it is "heard". */
  typing: 'words' | 'line';
  /** Per word for `words`, per line for `line`. */
  typingMs: number;
  lineGapMs: number;
};

/**
 * Plays a script the way the server streams: partials, then a final per line, then
 * `onComplete`. Returns a stop function.
 */
export const playScript = (
  script: readonly ScriptLine[],
  handlers: Pick<AsrSessionHandlers, 'onPartial' | 'onFinal' | 'onComplete'>,
  timing: ScriptTiming,
): (() => void) => {
  const timers: ReturnType<typeof setTimeout>[] = [];
  const at = (ms: number, run: () => void) => timers.push(setTimeout(run, ms));
  let time = timing.firstLineMs;

  script.forEach(([language, original, translation], index) => {
    const words = original.split(' ');
    const typed = timing.typing === 'words' ? words.map((_, n) => words.slice(0, n + 1)) : [words];
    for (const shown of typed) {
      const text = shown.join(' ');
      at(time, () => handlers.onPartial({ type: 'partial', text, language }));
      time += timing.typingMs;
    }
    at(time, () =>
      handlers.onFinal({ type: 'final', original, translation, language, segmentId: `${index}` }),
    );
    time += timing.lineGapMs;
  });
  at(time - timing.lineGapMs, () => handlers.onComplete?.());

  return () => timers.forEach(clearTimeout);
};
