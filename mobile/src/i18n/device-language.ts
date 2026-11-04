import { getLocales } from 'expo-localization';

export const UI_LANGUAGES = ['en', 'ko', 'ja', 'zh', 'es', 'fr', 'id', 'vi', 'th', 'de'] as const;

export type UiLanguage = (typeof UI_LANGUAGES)[number];

export const isUiLanguage = (code: string | null | undefined): code is UiLanguage =>
  !!code && (UI_LANGUAGES as readonly string[]).includes(code);

export const deviceLanguage = (): UiLanguage => {
  const code = getLocales()[0]?.languageCode;
  return isUiLanguage(code) ? code : 'en';
};

/** Each UI language named in itself, so people can find theirs whatever the app is showing. */
export const UI_LANGUAGE_NAMES: Record<UiLanguage, string> = {
  en: 'English',
  ko: '한국어',
  ja: '日本語',
  zh: '中文',
  es: 'Español',
  fr: 'Français',
  id: 'Bahasa Indonesia',
  vi: 'Tiếng Việt',
  th: 'ไทย',
  de: 'Deutsch',
};
