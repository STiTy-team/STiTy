import { z } from 'zod';

import { createLogger } from '@/lib/logger';
import { secureStorage } from '@/services/secure-storage';

const TOKENS_KEY = 'stity.auth.tokens';

const log = createLogger('tokens');

export const tokensSchema = z.object({ accessToken: z.string(), refreshToken: z.string() });

export type Tokens = z.infer<typeof tokensSchema>;

const parse = (raw: string): Tokens | null => {
  try {
    const parsed = tokensSchema.safeParse(JSON.parse(raw));
    return parsed.success ? parsed.data : null;
  } catch {
    return null;
  }
};

/** Bumped by every clear(), so a refresh that started before a sign-out can't write tokens back. */
let epoch = 0;

/** The server's access and refresh tokens, kept in the phone's secure storage only. */
export const tokenStore = {
  /** A value that can't be read is removed, so it fails once instead of on every call. */
  async read(): Promise<Tokens | null> {
    const raw = await secureStorage.get(TOKENS_KEY);
    if (!raw) return null;
    const tokens = parse(raw);
    if (!tokens) {
      log.warn('stored tokens were unreadable and have been removed');
      await secureStorage.remove(TOKENS_KEY);
    }
    return tokens;
  },
  write: (tokens: Tokens) => secureStorage.set(TOKENS_KEY, JSON.stringify(tokens)),
  clear: () => {
    epoch += 1;
    return secureStorage.remove(TOKENS_KEY);
  },
  epoch: () => epoch,
};

export type TokenStore = typeof tokenStore;

/** When a JWT access token expires, in ms; null for tokens that carry no readable `exp`. */
export const tokenExpiry = (token: string): number | null => {
  const payload = token.split('.')[1];
  if (!payload) return null;
  try {
    const base64 = payload.replace(/-/g, '+').replace(/_/g, '/');
    const { exp } = JSON.parse(atob(base64.padEnd(Math.ceil(base64.length / 4) * 4, '='))) as {
      exp?: unknown;
    };
    return typeof exp === 'number' ? exp * 1000 : null;
  } catch {
    return null;
  }
};
