import { createInstance } from 'i18next';
import { initReactI18next } from 'react-i18next';

import { deviceLanguage, type UiLanguage } from './device-language';
import de from './locales/de.json';
import en from './locales/en.json';
import es from './locales/es.json';
import fr from './locales/fr.json';
import id from './locales/id.json';
import ja from './locales/ja.json';
import ko from './locales/ko.json';
import th from './locales/th.json';
import vi from './locales/vi.json';
import zh from './locales/zh.json';

export const resources = {
  en: { translation: en },
  ko: { translation: ko },
  ja: { translation: ja },
  zh: { translation: zh },
  es: { translation: es },
  fr: { translation: fr },
  id: { translation: id },
  vi: { translation: vi },
  th: { translation: th },
  de: { translation: de },
} satisfies Record<UiLanguage, unknown>;

const i18n = createInstance();

void i18n.use(initReactI18next).init({
  resources,
  lng: deviceLanguage(),
  fallbackLng: 'en',
  interpolation: { escapeValue: false },
  returnNull: false,
});

export default i18n;
