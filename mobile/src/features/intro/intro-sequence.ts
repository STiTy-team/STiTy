import { CONVERSATION_LANGUAGES } from '@/domain/languages';
import type { UiLanguage } from '@/i18n/device-language';

const GREETINGS: Record<UiLanguage, string> = {
  en: 'Hello',
  ko: '안녕하세요',
  ja: 'こんにちは',
  zh: '你好',
  es: 'Hola',
  fr: 'Bonjour',
  id: 'Halo',
  vi: 'Xin chào',
  th: 'สวัสดี',
  de: 'Hallo',
};

/** Greetings in the conversation languages, ending on the one the app is shown in. */
export const greetingSequence = (uiLanguage: UiLanguage): string[] => [
  ...CONVERSATION_LANGUAGES.filter((code) => code !== uiLanguage).map((code) => GREETINGS[code]),
  GREETINGS[uiLanguage],
];

export const INTRO_TIMING = {
  liftMs: 420,
  firstWordAtMs: 240,
  wordMs: 190,
  holdLastMs: 560,
  fadeOutMs: 280,
  reducedMotionHoldMs: 350,
} as const;
