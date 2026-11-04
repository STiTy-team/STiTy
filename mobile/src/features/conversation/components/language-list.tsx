import { Ionicons } from '@expo/vector-icons';
import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Pressable } from 'react-native';

import { ListBadge, ListRow, ListSection } from '@/components/list';
import { CONVERSATION_LANGUAGES } from '@/domain/languages';
import { languageName } from '@/features/conversation/language-name';
import { canRemoveLanguage, usePreferencesStore } from '@/stores/preferences-store';
import { useColors } from '@/theme';
import { LanguagePickerSheet } from './language-picker-sheet';

/** The conversation's languages. Tapping one makes it mine; extra languages can be removed. */
export const LanguageList = () => {
  const { t } = useTranslation();
  const colors = useColors();
  const languages = usePreferencesStore((state) => state.languages);
  const myLanguage = usePreferencesStore((state) => state.myLanguage);
  const [picking, setPicking] = useState(false);
  const { setMyLanguage, removeLanguage } = usePreferencesStore.getState();
  const canAdd = CONVERSATION_LANGUAGES.some((code) => !languages.includes(code));

  return (
    <>
      <ListSection title={t('home.languagesTitle')} footer={t('home.languagesHint')}>
        {languages.map((code) => {
          const mine = code === myLanguage;
          const name = languageName(t, code);
          return (
            <ListRow
              key={code}
              title={name}
              onPress={() => setMyLanguage(code)}
              accessibilityState={{ selected: mine }}
            >
              {mine && <ListBadge>{t('home.myLanguage')}</ListBadge>}
              {canRemoveLanguage({ languages, myLanguage }, code) && (
                <Pressable
                  onPress={() => removeLanguage(code)}
                  hitSlop={10}
                  accessibilityRole="button"
                  accessibilityLabel={t('home.removeLanguage', { language: name })}
                >
                  <Ionicons name="remove-circle-outline" size={22} color={colors.textTertiary} />
                </Pressable>
              )}
            </ListRow>
          );
        })}
        {canAdd && (
          <ListRow
            title={t('home.addLanguage')}
            icon="add"
            tone="action"
            onPress={() => setPicking(true)}
          />
        )}
      </ListSection>
      <LanguagePickerSheet visible={picking} onClose={() => setPicking(false)} />
    </>
  );
};
