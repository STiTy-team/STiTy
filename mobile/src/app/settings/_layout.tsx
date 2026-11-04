import { Stack } from 'expo-router';
import { useTranslation } from 'react-i18next';

import { goBack } from '@/components/go-back';
import { IconButton } from '@/components/icon-button';
import { typography, useColors } from '@/theme';

export default function SettingsLayout() {
  const { t } = useTranslation();
  const colors = useColors();
  return (
    <Stack
      screenOptions={{
        headerShadowVisible: false,
        headerTintColor: colors.accent,
        headerStyle: { backgroundColor: colors.background },
        headerTitleStyle: { ...typography.title, color: colors.text },
        headerTitleAlign: 'center',
        headerLeft: () => (
          <IconButton icon="chevron-back" label={t('common.back')} onPress={goBack} />
        ),
        contentStyle: { backgroundColor: colors.background },
      }}
    >
      <Stack.Screen name="language" options={{ title: t('settings.language') }} />
      <Stack.Screen name="privacy" options={{ title: t('settings.privacy') }} />
      <Stack.Screen name="terms" options={{ title: t('settings.terms') }} />
    </Stack>
  );
}
