import { z } from 'zod';

import { resolveApiModes, type AppVariant } from '@/api/modes';
import { appInfo } from '@/config/app-info';

/** Only development builds may fall back to a built-in server; release builds must set one. */
const DEVELOPMENT_SERVER_URL = 'wss://qx74ylzesr5sqr-8765.proxy.runpod.net';

const schema = z.object({
  EXPO_PUBLIC_SERVER_URL: z.url({ protocol: /^wss?$/ }).optional(),
  EXPO_PUBLIC_API_URL: z.url({ protocol: /^https?$/ }).optional(),
  EXPO_PUBLIC_API_MOCKS: z.string().optional(),
});

export type EnvVariables = z.input<typeof schema>;

/** An origin only: the app appends its own paths, so a path, query or hash is a mistake. */
const origin = (name: string, url: string): string => {
  const { pathname, search, hash } = new URL(url);
  if ((pathname && pathname !== '/') || search || hash) {
    throw new Error(`${name} must be an origin without a path, got ${url}`);
  }
  return url.replace(/\/+$/, '');
};

/** Release builds talk to the server over TLS only. */
const requireTls = (name: string, url: string, variant: AppVariant) => {
  if (variant !== 'development' && /^(ws|http):/.test(url)) {
    throw new Error(`${name} must use wss:// or https:// in ${variant} builds, got ${url}`);
  }
};

export const resolveEnv = (variables: EnvVariables, variant: AppVariant) => {
  const parsed = schema.parse(variables);
  const configured =
    parsed.EXPO_PUBLIC_SERVER_URL ??
    (variant === 'development' ? DEVELOPMENT_SERVER_URL : undefined);
  if (!configured) {
    throw new Error(`EXPO_PUBLIC_SERVER_URL must be set for ${variant} builds.`);
  }
  const serverUrl = origin('EXPO_PUBLIC_SERVER_URL', configured);
  const apiUrl = origin(
    'EXPO_PUBLIC_API_URL',
    parsed.EXPO_PUBLIC_API_URL ?? serverUrl.replace(/^ws/, 'http'),
  );
  requireTls('EXPO_PUBLIC_SERVER_URL', serverUrl, variant);
  requireTls('EXPO_PUBLIC_API_URL', apiUrl, variant);

  return {
    /** WebSocket origin of the ASR server. */
    serverUrl,
    /** REST origin; the same host as the WebSocket unless set. */
    apiUrl,
    apiModes: resolveApiModes(parsed.EXPO_PUBLIC_API_MOCKS, variant),
  } as const;
};

// Expo inlines EXPO_PUBLIC_* only when each variable is referenced by its full name.
export const env = resolveEnv(
  {
    EXPO_PUBLIC_SERVER_URL: process.env.EXPO_PUBLIC_SERVER_URL || undefined,
    EXPO_PUBLIC_API_URL: process.env.EXPO_PUBLIC_API_URL || undefined,
    EXPO_PUBLIC_API_MOCKS: process.env.EXPO_PUBLIC_API_MOCKS || undefined,
  },
  appInfo.variant,
);
