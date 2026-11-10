import { z } from 'zod';

import { resolveApiModes, type AppVariant } from '@/api/modes';
import { appInfo } from '@/config/app-info';
import { serverOverride } from '@/config/server-override';

/** A development build with no server set looks for one on the machine running Metro. */
const DEVELOPMENT_SERVER_URL = 'ws://localhost:8765';

/** Where the server address came from: set in the app, set at build time, or the fallback. */
export type ServerSource = 'app' | 'build' | 'fallback';

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

/** Only development builds may take a server set in the app or fall back to a built-in one. */
export const resolveEnv = (
  variables: EnvVariables,
  variant: AppVariant,
  appServerUrl: string | null = null,
) => {
  const parsed = schema.parse(variables);
  const fromApp = variant === 'development' ? appServerUrl : null;
  const source: ServerSource = fromApp
    ? 'app'
    : parsed.EXPO_PUBLIC_SERVER_URL
      ? 'build'
      : 'fallback';
  const configured =
    fromApp ??
    parsed.EXPO_PUBLIC_SERVER_URL ??
    (variant === 'development' ? DEVELOPMENT_SERVER_URL : undefined);
  if (!configured) {
    throw new Error(`EXPO_PUBLIC_SERVER_URL must be set for ${variant} builds.`);
  }
  const serverUrl = origin('EXPO_PUBLIC_SERVER_URL', configured);
  const conversationApiUrl = serverUrl.replace(/^ws/, 'http');
  // A server set in the app serves its own REST API; the build's API URL belongs to another one.
  const apiUrl = origin(
    'EXPO_PUBLIC_API_URL',
    (!fromApp && parsed.EXPO_PUBLIC_API_URL) || conversationApiUrl,
  );
  requireTls('EXPO_PUBLIC_SERVER_URL', serverUrl, variant);
  requireTls('EXPO_PUBLIC_API_URL', apiUrl, variant);

  return {
    /** WebSocket origin of the ASR server. */
    serverUrl,
    /** REST origin; the same host as the WebSocket unless set. */
    apiUrl,
    /** REST origin of the conversation server, which `apiUrl` may not point at. */
    conversationApiUrl,
    serverSource: source,
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
  appInfo.variant === 'development' ? serverOverride.read() : null,
);
