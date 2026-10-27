import { secureStorage } from '@/services/secure-storage';

import { tokenExpiry, tokenStore } from './tokens';

jest.mock('@/services/secure-storage', () => {
  const values = new Map<string, string>();
  return {
    secureStorage: {
      get: jest.fn(async (key: string) => values.get(key) ?? null),
      set: jest.fn(async (key: string, value: string) => void values.set(key, value)),
      remove: jest.fn(async (key: string) => void values.delete(key)),
    },
  };
});

describe('tokenStore', () => {
  it('removes corrupt stored JSON and reads as signed out', async () => {
    await secureStorage.set('stity.auth.tokens', '{not json');

    await expect(tokenStore.read()).resolves.toBeNull();
    expect(secureStorage.remove).toHaveBeenCalledWith('stity.auth.tokens');
    await expect(secureStorage.get('stity.auth.tokens')).resolves.toBeNull();
  });

  it('bumps the epoch on clear', async () => {
    const before = tokenStore.epoch();
    await tokenStore.clear();
    expect(tokenStore.epoch()).toBe(before + 1);
  });
});

describe('tokenExpiry', () => {
  it('reads exp from a JWT and returns null for anything else', () => {
    const payload = btoa(JSON.stringify({ exp: 1_700_000_000 })).replace(/=+$/, '');
    expect(tokenExpiry(`a.${payload}.b`)).toBe(1_700_000_000_000);
    expect(tokenExpiry('opaque-token')).toBeNull();
    expect(tokenExpiry('a.%%%.b')).toBeNull();
  });
});
