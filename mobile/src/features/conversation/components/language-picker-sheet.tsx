import { useTranslation } from 'react-i18next';
import { ScrollView } from 'react-native';

import { BottomSheet } from '@/components/bottom-sheet';
import { ListBadge, ListRow, ListSection } from '@/components/list';
import { CONVERSATION_LANGUAGES, isSupportedLanguage } from '@/domain/languages';
import { languageName } from '@/features/conversation/language-name';
import { usePreferencesStore } from '@/stores/preferences-store';

type Props = {
  visible: boolean;
  onClose: () => void;
};

/** Languages not yet in the conversation; ones the server can't handle yet are shown disabled. */
export const LanguagePickerSheet = ({ visible, onClose }: Props) => {
  const { t } = useTranslation();
  const languages = usePreferencesStore((state) => state.languages);
  const candidates = CONVERSATION_LANGUAGES.filter((code) => !languages.includes(code)).sort(
    (a, b) => Number(isSupportedLanguage(b)) - Number(isSupportedLanguage(a)),
  );

  return (
    <BottomSheet visible={visible} title={t('languagePicker.title')} onClose={onClose}>
      <ScrollView>
        <ListSection>
          {candidates.map((code) => {
            const supported = isSupportedLanguage(code);
            return (
              <ListRow
                key={code}
                title={languageName(t, code)}
                tone={supported ? 'default' : 'disabled'}
                accessibilityState={{ disabled: !supported }}
                onPress={() => {
                  usePreferencesStore.getState().addLanguage(code);
                  onClose();
                }}
              >
                {!supported && <ListBadge muted>{t('languagePicker.comingSoon')}</ListBadge>}
              </ListRow>
            );
          })}
        </ListSection>
      </ScrollView>
    </BottomSheet>
  );
};
