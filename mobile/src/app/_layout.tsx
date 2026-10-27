import '@/i18n';

import { useFonts } from 'expo-font';
import { Stack } from 'expo-router';
import * as SplashScreen from 'expo-splash-screen';
import { StatusBar } from 'expo-status-bar';
import { useCallback, useState } from 'react';
import { SafeAreaProvider } from 'react-native-safe-area-context';

import { DialogHost } from '@/components/dialog';
import { useConversationLifecycle } from '@/features/conversation/hooks/use-conversation-lifecycle';
import { useDisplayPreferences } from '@/features/display/use-display-preferences';
import { IntroOverlay } from '@/features/intro/intro-overlay';
import { useStartup } from '@/features/startup/use-startup';
import { useConversationPreferences } from '@/features/conversation/conversation-runtime';
import { useReadingPreferences } from '@/features/records/records-runtime';
import { usePreferencesStore } from '@/stores/preferences-store';
import { useHydrated } from '@/stores/use-hydrated';
import { fonts, useColors, useColorSchemeName } from '@/theme';

void SplashScreen.preventAutoHideAsync();

// Release builds embed these through the expo-font config plugin; loading them here as well
// keeps development builds that predate the plugin rendering the right faces.
const FONT_FILES = {
  [fonts.regular]: require('../../assets/fonts/Pretendard-Regular.otf'),
  [fonts.medium]: require('../../assets/fonts/Pretendard-Medium.otf'),
  [fonts.semibold]: require('../../assets/fonts/Pretendard-SemiBold.otf'),
  [fonts.bold]: require('../../assets/fonts/Pretendard-Bold.otf'),
};

export default function RootLayout() {
  const colors = useColors();
  const scheme = useColorSchemeName();
  const [fontsLoaded, fontError] = useFonts(FONT_FILES);
  const hydrated = [
    useHydrated(usePreferencesStore),
    useHydrated(useConversationPreferences),
    useHydrated(useReadingPreferences),
  ].every(Boolean);
  const [introDone, setIntroDone] = useState(false);
  const finishIntro = useCallback(() => setIntroDone(true), []);
  const hideSplash = useCallback(() => void SplashScreen.hideAsync(), []);
  useStartup();
  useDisplayPreferences();
  useConversationLifecycle();

  if ((!fontsLoaded && !fontError) || !hydrated) return null;

  return (
    <SafeAreaProvider>
      <StatusBar style={scheme === 'dark' ? 'light' : 'dark'} />
      <Stack
        screenOptions={{ headerShown: false, contentStyle: { backgroundColor: colors.background } }}
      >
        <Stack.Screen name="(tabs)" />
        <Stack.Screen name="conversation" options={{ gestureEnabled: false }} />
        <Stack.Screen name="conversation-settings" />
        <Stack.Screen
          name="present"
          options={{ gestureEnabled: false, animation: 'fade', orientation: 'landscape' }}
        />
        <Stack.Screen name="records/[id]" />
        <Stack.Screen name="settings" />
        <Stack.Screen name="sign-in" options={{ presentation: 'modal' }} />
      </Stack>
      <DialogHost />
      {!introDone && <IntroOverlay onShown={hideSplash} onDone={finishIntro} />}
    </SafeAreaProvider>
  );
}
