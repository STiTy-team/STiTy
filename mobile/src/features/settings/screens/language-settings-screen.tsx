import { Ionicons } from '@expo/vector-icons';
import { useTranslation } from 'react-i18next';

import { ListRow, ListSection, ListValue } from '@/components/list';
import { ScreenScroll } from '@/features/settings/components/screen-scroll';
import {
  deviceLanguage,
  UI_LANGUAGE_NAMES,
  UI_LANGUAGES,
  type UiLanguage,
} from '@/i18n/device-language';
import { usePreferencesStore, type UiLanguagePreference } from '@/stores/preferences-store';
import { useColors } from '@/theme';

/** Which language the app's own screens are shown in; separate from the conversation languages. */
export const LanguageSettingsScreen = () => {
  const { t } = useTranslation();
  const colors = useColors();
  const selected = usePreferencesStore((state) => state.uiLanguage);
  const { setUiLanguage } = usePreferencesStore.getState();

  const check = (option: UiLanguagePreference) =>
    option === selected && <Ionicons name="checkmark" size={20} color={colors.accent} />;

  return (
    <ScreenScroll>
      <ListSection footer={t('settings.languageHint')}>
        <ListRow
          title={t('settings.systemLanguage')}
          onPress={() => setUiLanguage('system')}
          accessibilityState={{ selected: selected === 'system' }}
        >
          <ListValue>{UI_LANGUAGE_NAMES[deviceLanguage()]}</ListValue>
          {check('system')}
        </ListRow>
      </ListSection>
      <ListSection>
        {UI_LANGUAGES.map((code: UiLanguage) => (
          <ListRow
            key={code}
            title={UI_LANGUAGE_NAMES[code]}
            onPress={() => setUiLanguage(code)}
            accessibilityState={{ selected: selected === code }}
          >
            {check(code)}
          </ListRow>
        ))}
      </ListSection>
    </ScreenScroll>
  );
};
