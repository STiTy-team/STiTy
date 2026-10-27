import { parseVariant } from './app-info';
import { resolveEnv } from './env';

describe('resolveEnv', () => {
  it('derives the REST origin from the WebSocket origin', () => {
    const env = resolveEnv({ EXPO_PUBLIC_SERVER_URL: 'wss://asr.example.com/' }, 'production');
    expect(env).toMatchObject({
      serverUrl: 'wss://asr.example.com',
      apiUrl: 'https://asr.example.com',
    });
  });

  it('allows cleartext only in development', () => {
    const local = { EXPO_PUBLIC_SERVER_URL: 'ws://192.168.0.2:8765' };
    expect(resolveEnv(local, 'development').apiUrl).toBe('http://192.168.0.2:8765');
    expect(() => resolveEnv(local, 'preview')).toThrow(/wss:\/\//);
    expect(() =>
      resolveEnv(
        { EXPO_PUBLIC_SERVER_URL: 'wss://asr.example.com', EXPO_PUBLIC_API_URL: 'http://api' },
        'production',
      ),
    ).toThrow(/https:\/\//);
  });

  it('rejects a server URL with a path, query or hash', () => {
    for (const url of ['wss://asr.example.com/api/v1/ws', 'wss://a.example.com?x=1', 'wss://a#b']) {
      expect(() => resolveEnv({ EXPO_PUBLIC_SERVER_URL: url }, 'development')).toThrow(/origin/);
    }
  });

  it('requires a server URL outside development', () => {
    expect(() => resolveEnv({}, 'preview')).toThrow(/must be set/);
    expect(resolveEnv({}, 'development').serverUrl).toMatch(/^wss:\/\//);
  });
});

describe('parseVariant', () => {
  it('accepts the three variants, defaults a missing one and refuses anything else', () => {
    expect(parseVariant('preview')).toBe('preview');
    expect(parseVariant(undefined)).toBe('development');
    expect(() => parseVariant('staging')).toThrow(/Unknown app variant/);
  });
});
