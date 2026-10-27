import { z } from 'zod';
import { create } from 'zustand';
import { persist } from 'zustand/middleware';

import { persistedStorage, savedField } from '@/stores/persisted-storage';

export const TRANSCRIPT_VIEWS = ['mine', 'original', 'bilingual'] as const;
export type TranscriptView = (typeof TRANSCRIPT_VIEWS)[number];

/** Steps of the viewer's text size control, as multiples of the base reading size. */
export const READING_SCALES = [0.9, 1, 1.15, 1.3, 1.5] as const;

type ReadingSettings = {
  /** Which text the transcript viewer shows. */
  transcriptView: TranscriptView;
  /** Index into READING_SCALES. */
  readingScale: number;
};

const DEFAULT_READING: ReadingSettings = { transcriptView: 'mine', readingScale: 1 };

type ReadingState = ReadingSettings & {
  setTranscriptView: (view: TranscriptView) => void;
  setReadingScale: (index: number) => void;
  reset: () => void;
};

const savedSchema = z
  .object({
    transcriptView: savedField(z.enum(TRANSCRIPT_VIEWS)),
    readingScale: savedField(
      z
        .number()
        .int()
        .min(0)
        .max(READING_SCALES.length - 1),
    ),
  })
  .catch({});

/** How the saved-transcript viewer reads. */
export const useReadingPreferences = create<ReadingState>()(
  persist(
    (set) => ({
      ...DEFAULT_READING,
      setTranscriptView: (transcriptView) => set({ transcriptView }),
      setReadingScale: (index) =>
        set({ readingScale: Math.min(READING_SCALES.length - 1, Math.max(0, Math.round(index))) }),
      reset: () => set(DEFAULT_READING),
    }),
    {
      name: 'stity.reading-preferences',
      version: 1,
      storage: persistedStorage(1),
      merge: (saved, current) => {
        const restored = savedSchema.parse(saved);
        return {
          ...current,
          transcriptView: restored.transcriptView ?? current.transcriptView,
          readingScale: restored.readingScale ?? current.readingScale,
        };
      },
      partialize: ({ transcriptView, readingScale }) => ({ transcriptView, readingScale }),
    },
  ),
);
