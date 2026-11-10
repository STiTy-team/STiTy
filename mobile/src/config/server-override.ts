import * as SecureStore from 'expo-secure-store';
import { Platform } from 'react-native';

const KEY = 'stity.dev.serverUrl';

/**
 * Turns what a developer types or scans into a WebSocket origin: `https://host` becomes
 * `wss://host`, a bare host gets `wss://`. Returns null for anything that isn't an origin.
 */
export const normalizeServerUrl = (input: string): string | null => {
  const trimmed = input.trim();
  if (!trimmed) return null;
  const withScheme = /^[a-z]+:\/\//i.test(trimmed) ? trimmed : `wss://${trimmed}`;
  let url: URL;
  try {
    url = new URL(withScheme.replace(/^http/i, 'ws'));
  } catch {
    return null;
  }
  if (!/^wss?:$/.test(url.protocol) || !url.hostname) return null;
  if ((url.pathname && url.pathname !== '/') || url.search || url.hash) return null;
  return `${url.protocol}//${url.host}`;
};

/** Synchronous so `env` can be resolved at import time, before anything opens a socket. */
export const serverOverride = {
  /** A saved value that no longer parses reads as unset, so it can't stop the app starting. */
  read(): string | null {
    const saved =
      Platform.OS === 'web' ? globalThis.localStorage?.getItem(KEY) : SecureStore.getItem(KEY);
    return saved ? normalizeServerUrl(saved) : null;
  },
  write(url: string): void {
    if (Platform.OS === 'web') return globalThis.localStorage?.setItem(KEY, url);
    SecureStore.setItem(KEY, url);
  },
  async clear(): Promise<void> {
    if (Platform.OS === 'web') return globalThis.localStorage?.removeItem(KEY);
    await SecureStore.deleteItemAsync(KEY);
  },
};
