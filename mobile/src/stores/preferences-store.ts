import { z } from 'zod';
import { create } from 'zustand';
import { persist } from 'zustand/middleware';

import {
  isSupportedLanguage,
  MIN_CONVERSATION_LANGUAGES,
  SUPPORTED_LANGUAGES,
  type LanguageCode,
} from '@/domain/languages';
import { deviceLanguage, UI_LANGUAGES, type UiLanguage } from '@/i18n/device-language';

import { LEGACY_PREFERENCES_KEY, persistedStorage, savedField } from './persisted-storage';

export type LanguageSettings = {
  /** Every language spoken in the conversation, in the order they were added. */
  languages: LanguageCode[];
  /** The language this phone reads and hears translations in; always one of `languages`. */
  myLanguage: LanguageCode;
};

/** 'system' follows the phone's setting. */
export type UiLanguagePreference = UiLanguage | 'system';
export const THEME_PREFERENCES = ['system', 'light', 'dark'] as const;
export type ThemePreference = (typeof THEME_PREFERENCES)[number];

type DisplaySettings = {
  uiLanguage: UiLanguagePreference;
  theme: ThemePreference;
};

const DEFAULT_DISPLAY: DisplaySettings = { uiLanguage: 'system', theme: 'system' };

/**
 * App-wide settings. A setting used by one feature lives with that feature: conversation
 * add-ons in `features/conversation`, the transcript viewer's in `features/records`.
 */
type PreferencesState = LanguageSettings &
  DisplaySettings & {
    setMyLanguage: (language: LanguageCode) => void;
    addLanguage: (language: LanguageCode) => void;
    removeLanguage: (language: LanguageCode) => void;
    setUiLanguage: (uiLanguage: UiLanguagePreference) => void;
    setTheme: (theme: ThemePreference) => void;
    reset: () => void;
  };

const defaultLanguages = (): LanguageSettings => {
  const device = deviceLanguage();
  const myLanguage: LanguageCode = isSupportedLanguage(device) ? device : 'en';
  const others = SUPPORTED_LANGUAGES.filter((code) => code !== myLanguage);
  return {
    languages: [myLanguage, ...others.slice(0, MIN_CONVERSATION_LANGUAGES - 1)],
    myLanguage,
  };
};

/** Drops languages the server no longer supports and refills to a usable conversation. */
const sanitize = (saved: { languages?: string[]; myLanguage?: string }): LanguageSettings => {
  const fallback = defaultLanguages();
  const languages = [...new Set(saved.languages ?? [])].filter(isSupportedLanguage);
  for (const code of fallback.languages) {
    if (languages.length >= MIN_CONVERSATION_LANGUAGES) break;
    if (!languages.includes(code)) languages.push(code);
  }
  const myLanguage = languages.find((code) => code === saved.myLanguage) ?? languages[0]!;
  return { languages, myLanguage };
};

/** Each saved field is checked on its own, so one bad value falls back without losing the rest. */
const savedSchema = z
  .object({
    languages: savedField(z.array(z.string())),
    myLanguage: savedField(z.string()),
    uiLanguage: savedField(z.enum([...UI_LANGUAGES, 'system'])),
    theme: savedField(z.enum(THEME_PREFERENCES)),
  })
  .catch({});

export const canRemoveLanguage = (
  { languages, myLanguage }: LanguageSettings,
  language: LanguageCode,
): boolean => language !== myLanguage && languages.length > MIN_CONVERSATION_LANGUAGES;

export const usePreferencesStore = create<PreferencesState>()(
  persist(
    (set) => ({
      ...defaultLanguages(),
      ...DEFAULT_DISPLAY,
      setMyLanguage: (myLanguage) =>
        set(({ languages }) => (languages.includes(myLanguage) ? { myLanguage } : {})),
      addLanguage: (language) =>
        set(({ languages }) =>
          languages.includes(language) ? {} : { languages: [...languages, language] },
        ),
      removeLanguage: (language) =>
        set((state) =>
          canRemoveLanguage(state, language)
            ? { languages: state.languages.filter((code) => code !== language) }
            : {},
        ),
      setUiLanguage: (uiLanguage) => set({ uiLanguage }),
      setTheme: (theme) => set({ theme }),
      reset: () => set({ ...defaultLanguages(), ...DEFAULT_DISPLAY }),
    }),
    {
      name: LEGACY_PREFERENCES_KEY,
      version: 2,
      storage: persistedStorage(2),
      // Version 1 stored a fixed pair of languages.
      migrate: (saved, version) => {
        if (version >= 2) return saved;
        const { myLanguage, peerLanguage } = saved as {
          myLanguage?: string;
          peerLanguage?: string;
        };
        return { myLanguage, languages: [myLanguage, peerLanguage] };
      },
      merge: (saved, current) => {
        const restored = savedSchema.parse(saved);
        return {
          ...current,
          ...sanitize(restored),
          uiLanguage: restored.uiLanguage ?? current.uiLanguage,
          theme: restored.theme ?? current.theme,
        };
      },
      partialize: ({ languages, myLanguage, uiLanguage, theme }) => ({
        languages,
        myLanguage,
        uiLanguage,
        theme,
      }),
    },
  ),
);

/**
 * The wire protocol carries one source and one target language, so the peer is the first
 * conversation language that is not mine.
 */
export const peerLanguageOf = ({ languages, myLanguage }: LanguageSettings): LanguageCode =>
  languages.find((code) => code !== myLanguage) ?? myLanguage;
