export const CONVERSATION_LANGUAGES = ['ko', 'ja', 'zh', 'es', 'en'] as const;

export type LanguageCode = (typeof CONVERSATION_LANGUAGES)[number];

/** Languages the ASR server can transcribe today; the rest are listed as coming soon. */
export const SUPPORTED_LANGUAGES: readonly LanguageCode[] = ['ko', 'en'];

/** A conversation needs at least two languages to translate between. */
export const MIN_CONVERSATION_LANGUAGES = 2;

const BCP47: Record<string, string> = {
  ko: 'ko-KR',
  en: 'en-US',
  ja: 'ja-JP',
  zh: 'zh-CN',
  es: 'es-ES',
  fr: 'fr-FR',
  de: 'de-DE',
  id: 'id-ID',
  vi: 'vi-VN',
  th: 'th-TH',
};

const ENGLISH_NAMES: Record<string, string> = {
  korean: 'ko',
  english: 'en',
  japanese: 'ja',
  chinese: 'zh',
  spanish: 'es',
  french: 'fr',
  german: 'de',
  indonesian: 'id',
  vietnamese: 'vi',
  thai: 'th',
};

export const isConversationLanguage = (code: string): code is LanguageCode =>
  (CONVERSATION_LANGUAGES as readonly string[]).includes(code);

export const isSupportedLanguage = (code: string): code is LanguageCode =>
  (SUPPORTED_LANGUAGES as readonly string[]).includes(code);

/** Servers report either an ISO code ("ko") or an English name ("Korean"). */
export const normalizeLanguageCode = (value: string): string => {
  const lower = value.trim().toLowerCase();
  return ENGLISH_NAMES[lower] ?? lower.slice(0, 2);
};

export const toLanguageTag = (code: string): string => BCP47[code] ?? code;
