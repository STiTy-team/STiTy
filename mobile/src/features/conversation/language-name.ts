import type { TFunction } from 'i18next';

import { isConversationLanguage } from '@/domain/languages';

/** The language's own display name, or its code for languages the app has no name for. */
export const languageName = (t: TFunction, code: string): string =>
  isConversationLanguage(code) ? t(`languages.${code}`) : code.toUpperCase();
