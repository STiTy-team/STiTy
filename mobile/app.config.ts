import type { ConfigContext, ExpoConfig } from 'expo/config';

type AppVariant = 'development' | 'preview' | 'production';

const VARIANT = (process.env.APP_VARIANT ?? 'development') as AppVariant;

const VARIANTS: Record<AppVariant, { name: string; idSuffix: string; scheme: string }> = {
  development: { name: 'STiTy (Dev)', idSuffix: '.dev', scheme: 'stity-dev' },
  preview: { name: 'STiTy (Preview)', idSuffix: '.preview', scheme: 'stity-preview' },
  production: { name: 'STiTy', idSuffix: '', scheme: 'stity' },
};

const PLACEHOLDER_ID = 'com.anonymous.stitymobile';
/** Permanent after the first store upload, so a release build refuses the placeholder. */
const BASE_ID = process.env.APP_ID ?? PLACEHOLDER_ID;
if (VARIANT === 'production' && BASE_ID === PLACEHOLDER_ID) {
  throw new Error('Set APP_ID to the real bundle id before building for production.');
}
const EAS_PROJECT_ID = 'ed856a8e-782a-4b00-92dc-024fddd15d17';
const MICROPHONE_USAGE =
  'STiTy uses the microphone to transcribe and translate your conversation in real time.';

export default ({ config }: ConfigContext): ExpoConfig => {
  const variant = VARIANTS[VARIANT];
  const appId = `${BASE_ID}${variant.idSuffix}`;

  return {
    ...config,
    name: variant.name,
    slug: 'stity-mobile',
    owner: 'stity-team',
    // Its own scheme per variant, so a server QR code opens the dev build, never the store app.
    scheme: variant.scheme,
    version: '1.1.0',
    // Portrait everywhere; Present mode alone turns to landscape, so rotation stays unlocked natively.
    orientation: 'default',
    icon: './assets/images/icon.png',
    userInterfaceStyle: 'automatic',
    ios: {
      bundleIdentifier: appId,
      icon: {
        light: './assets/images/icon.png',
        dark: './assets/images/icon-dark.png',
        tinted: './assets/images/icon-tinted.png',
      },
      supportsTablet: true,
      infoPlist: {
        NSMicrophoneUsageDescription: MICROPHONE_USAGE,
        ITSAppUsesNonExemptEncryption: false,
      },
    },
    android: {
      package: appId,
      adaptiveIcon: {
        foregroundImage: './assets/images/adaptive-icon.png',
        monochromeImage: './assets/images/adaptive-icon-monochrome.png',
        backgroundColor: '#3352C6',
      },
      permissions: ['android.permission.RECORD_AUDIO', 'android.permission.MODIFY_AUDIO_SETTINGS'],
      blockedPermissions: ['android.permission.SYSTEM_ALERT_WINDOW'],
    },
    plugins: [
      'expo-router',
      'expo-localization',
      ['expo-screen-orientation', { initialOrientation: 'PORTRAIT_UP' }],
      [
        'expo-font',
        {
          fonts: [
            './assets/fonts/Pretendard-Regular.otf',
            './assets/fonts/Pretendard-Medium.otf',
            './assets/fonts/Pretendard-SemiBold.otf',
            './assets/fonts/Pretendard-Bold.otf',
          ],
        },
      ],
      [
        'expo-splash-screen',
        {
          image: './assets/images/splash-icon.png',
          imageWidth: 200,
          resizeMode: 'contain',
          backgroundColor: '#FFFFFF',
          dark: { image: './assets/images/splash-icon-dark.png', backgroundColor: '#0E1013' },
        },
      ],
      [
        'expo-build-properties',
        {
          android: { usesCleartextTraffic: VARIANT === 'development' },
        },
      ],
    ],
    // One index.html for every route; EAS Hosting serves it for deep links such as /dev-server.
    web: { output: 'single' },
    experiments: {
      typedRoutes: true,
    },
    extra: {
      appVariant: VARIANT,
      eas: { projectId: EAS_PROJECT_ID },
    },
  };
};
