import { useEffect } from 'react';
import { Appearance, Platform } from 'react-native';

import i18n from '@/i18n';
import { deviceLanguage } from '@/i18n/device-language';
import { usePreferencesStore } from '@/stores/preferences-store';
import { setColorSchemeOverride } from '@/theme';

/** Applies the saved UI language and theme, and keeps them applied as they change. */
export const useDisplayPreferences = () => {
  const uiLanguage = usePreferencesStore((state) => state.uiLanguage);
  const theme = usePreferencesStore((state) => state.theme);

  useEffect(() => {
    void i18n.changeLanguage(uiLanguage === 'system' ? deviceLanguage() : uiLanguage);
  }, [uiLanguage]);

  // The app's own colors follow the override everywhere. On phones the platform override is set
  // too, so native alerts and switches match; React Native Web has no such override.
  useEffect(() => {
    setColorSchemeOverride(theme === 'system' ? null : theme);
    if (Platform.OS !== 'web')
      Appearance.setColorScheme(theme === 'system' ? 'unspecified' : theme);
  }, [theme]);
};
