import { Platform } from 'react-native';

import { appInfo } from '@/config/app-info';
import { env } from '@/config/env';

/** The link that opens /join; it names the server only when this phone was moved to another one. */
export const roomLink = (code: string): string => {
  const origin = Platform.OS === 'web' ? window.location.origin : appInfo.webUrl;
  const server = env.serverSource === 'app' ? `&server=${encodeURIComponent(env.serverUrl)}` : '';
  return `${origin}/join?room=${encodeURIComponent(code)}${server}`;
};
