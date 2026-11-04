import { z } from 'zod';
import { create } from 'zustand';
import { persist } from 'zustand/middleware';

import type { OutputMode } from '@/domain/output-modes';
import { peerLanguageOf, type LanguageSettings } from '@/stores/preferences-store';
import { persistedStorage, savedField } from '@/stores/persisted-storage';

import type { SessionPreferences } from './ports';

/** Optional behaviours layered on the one conversation screen, switched in its settings. */
export type AddOns = {
  /** The mic sends audio only while the button is held. */
  holdToTalk: boolean;
  /** The live screen shows one sentence at a time, filling the screen. */
  largeText: boolean;
  /** A short vibration when a translation into my language arrives. */
  vibrate: boolean;
};

export const ADD_ONS = [
  'holdToTalk',
  'largeText',
  'vibrate',
] as const satisfies readonly (keyof AddOns)[];

const DEFAULT_ADD_ONS: AddOns = { holdToTalk: false, largeText: false, vibrate: false };

type ConversationPreferencesState = AddOns & {
  outputMode: OutputMode;
  muted: boolean;
  /** Ask the server to save the next conversation; chosen before Start and kept across runs. */
  saveRequested: boolean;
  setAddOn: (addOn: keyof AddOns, on: boolean) => void;
  setOutputMode: (mode: OutputMode) => void;
  setMuted: (muted: boolean) => void;
  setSaveRequested: (saveRequested: boolean) => void;
  reset: () => void;
};

const DEFAULTS: AddOns &
  Pick<ConversationPreferencesState, 'outputMode' | 'muted' | 'saveRequested'> = {
  ...DEFAULT_ADD_ONS,
  outputMode: 'speaker',
  muted: false,
  saveRequested: true,
};

const savedSchema = z
  .object({
    holdToTalk: savedField(z.boolean()),
    largeText: savedField(z.boolean()),
    vibrate: savedField(z.boolean()),
    saveRequested: savedField(z.boolean()),
  })
  .catch({});

/** How conversations run on this phone. */
export const useConversationPreferences = create<ConversationPreferencesState>()(
  persist(
    (set) => ({
      ...DEFAULTS,
      setAddOn: (addOn, on) => set({ [addOn]: on }),
      setOutputMode: (outputMode) => set({ outputMode }),
      setMuted: (muted) => set({ muted }),
      setSaveRequested: (saveRequested) => set({ saveRequested }),
      reset: () => set(DEFAULTS),
    }),
    {
      name: 'stity.conversation-preferences',
      version: 1,
      storage: persistedStorage(1),
      merge: (saved, current) => {
        const restored = savedSchema.parse(saved);
        return {
          ...current,
          holdToTalk: restored.holdToTalk ?? current.holdToTalk,
          largeText: restored.largeText ?? current.largeText,
          vibrate: restored.vibrate ?? current.vibrate,
          saveRequested: restored.saveRequested ?? current.saveRequested,
        };
      },
      // The output mode follows whether an earphone is plugged in at launch, and mute is per
      // conversation, so neither is saved.
      partialize: ({ holdToTalk, largeText, vibrate, saveRequested }) => ({
        holdToTalk,
        largeText,
        vibrate,
        saveRequested,
      }),
    },
  ),
);

export const sessionPreferences = (
  languages: LanguageSettings,
  conversation: ConversationPreferencesState,
): SessionPreferences => ({
  myLanguage: languages.myLanguage,
  peerLanguage: peerLanguageOf(languages),
  outputMode: conversation.outputMode,
  muted: conversation.muted,
  holdToTalk: conversation.holdToTalk,
  vibrate: conversation.vibrate,
  saveRequested: conversation.saveRequested,
});
