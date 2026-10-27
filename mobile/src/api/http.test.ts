import { z } from 'zod';

import type { Tokens, TokenStore } from '@/api/auth/tokens';
import { ApiError } from '@/api/errors';
import { createHttpClient } from '@/api/http';
import { onUnauthorized } from '@/api/session-events';

const OLD = { accessToken: 'old', refreshToken: 'refresh-1' };
const NEW = { accessToken: 'new', refreshToken: 'refresh-2' };

const memoryTokens = (initial: Tokens | null) => {
  let stored = initial;
  let epoch = 0;
  return {
    read: jest.fn(async () => stored),
    write: jest.fn(async (tokens: Tokens) => {
      stored = tokens;
    }),
    clear: jest.fn(async () => {
      epoch += 1;
      stored = null;
    }),
    epoch: () => epoch,
  } satisfies TokenStore;
};

const json = (status: number, body?: unknown) =>
  new Response(body === undefined ? null : JSON.stringify(body), { status });

type Route = (init: RequestInit) => Response | Promise<Response>;

const serve = (routes: Record<string, Route>) => {
  const fetchMock = jest.fn(async (url: string, init: RequestInit) => {
    const path = new URL(url).pathname.replace('/api/v1', '');
    const route = routes[path];
    if (!route) throw new Error(`unexpected ${path}`);
    return route(init);
  });
  globalThis.fetch = fetchMock as unknown as typeof fetch;
  return fetchMock;
};

const bearer = (init: RequestInit) =>
  (init.headers as Record<string, string>).Authorization?.replace('Bearer ', '');

/** `/me` accepts only the new token. */
const me: Route = (init) => (bearer(init) === 'new' ? json(200, { ok: true }) : json(401));

const getMe = (tokens: TokenStore) =>
  createHttpClient('https://api.test', tokens).request({
    path: '/me',
    schema: z.object({ ok: z.boolean() }),
    auth: true,
  });

let unauthorized: jest.Mock;
let stopListening: () => void;
beforeEach(() => {
  unauthorized = jest.fn();
  stopListening = onUnauthorized(unauthorized);
});
afterEach(() => stopListening());

describe('refreshing the access token', () => {
  it('retries with the refreshed token', async () => {
    const tokens = memoryTokens(OLD);
    serve({ '/me': me, '/auth/refresh': () => json(200, NEW) });

    await expect(getMe(tokens)).resolves.toEqual({ ok: true });
    expect(await tokens.read()).toEqual(NEW);
  });

  it('signs out when the server refuses the refresh token', async () => {
    const tokens = memoryTokens(OLD);
    serve({ '/me': me, '/auth/refresh': () => json(401) });

    await expect(getMe(tokens)).rejects.toMatchObject({ kind: 'unauthorized' });
    expect(await tokens.read()).toBeNull();
    expect(unauthorized).toHaveBeenCalledTimes(1);
  });

  it('keeps the tokens when the refresh hits a server error', async () => {
    const tokens = memoryTokens(OLD);
    serve({ '/me': me, '/auth/refresh': () => json(503) });

    await expect(getMe(tokens)).rejects.toMatchObject({ kind: 'server' });
    expect(await tokens.read()).toEqual(OLD);
    expect(unauthorized).not.toHaveBeenCalled();
  });

  it('keeps the tokens when the refresh times out or the network is down', async () => {
    const tokens = memoryTokens(OLD);
    serve({
      '/me': me,
      '/auth/refresh': () => {
        throw new TypeError('Network request failed');
      },
    });

    const error = await getMe(tokens).catch((caught: unknown) => caught);
    expect(error).toBeInstanceOf(ApiError);
    expect(error).toMatchObject({ kind: 'offline' });
    expect(await tokens.read()).toEqual(OLD);
    expect(unauthorized).not.toHaveBeenCalled();
  });

  it('keeps the tokens when a 2xx refresh has no tokens in it', async () => {
    const tokens = memoryTokens(OLD);
    serve({ '/me': me, '/auth/refresh': () => new Response('<html>', { status: 200 }) });

    await expect(getMe(tokens)).rejects.toMatchObject({ kind: 'server' });
    expect(await tokens.read()).toEqual(OLD);
  });

  it('does not write tokens back when the user signs out during the refresh', async () => {
    const tokens = memoryTokens(OLD);
    let answer!: (response: Response) => void;
    serve({
      '/me': me,
      '/auth/refresh': () => new Promise<Response>((resolve) => (answer = resolve)),
    });

    const pending = getMe(tokens).catch((caught: unknown) => caught);
    await new Promise((resolve) => setTimeout(resolve, 0));
    await tokens.clear();
    answer(json(200, NEW));

    expect(await pending).toMatchObject({ kind: 'unauthorized' });
    expect(await tokens.read()).toBeNull();
    expect(tokens.write).not.toHaveBeenCalled();
    expect(unauthorized).not.toHaveBeenCalled();
  });

  it('uses a token refreshed in parallel instead of refreshing again', async () => {
    const tokens = memoryTokens(OLD);
    const fetchMock = serve({
      '/me': async (init) => {
        if (bearer(init) === 'old') await tokens.write(NEW);
        return me(init);
      },
      '/auth/refresh': () => json(500),
    });

    await expect(getMe(tokens)).resolves.toEqual({ ok: true });
    expect(fetchMock.mock.calls.map(([url]) => new URL(url).pathname)).not.toContain(
      '/api/v1/auth/refresh',
    );
  });
});

describe('accessToken', () => {
  const jwt = (expSeconds: number) =>
    `h.${btoa(JSON.stringify({ exp: expSeconds })).replace(/=+$/, '')}.s`;

  it('refreshes an expired token before handing it out', async () => {
    const expired = { accessToken: jwt(Date.now() / 1000 - 10), refreshToken: 'refresh-1' };
    const tokens = memoryTokens(expired);
    const fetchMock = serve({ '/auth/refresh': () => json(200, NEW) });

    await expect(createHttpClient('https://api.test', tokens).accessToken()).resolves.toBe('new');
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it('hands out a token that is still valid without a refresh', async () => {
    const valid = { accessToken: jwt(Date.now() / 1000 + 3600), refreshToken: 'refresh-1' };
    const fetchMock = serve({});

    await expect(
      createHttpClient('https://api.test', memoryTokens(valid)).accessToken(),
    ).resolves.toBe(valid.accessToken);
    expect(fetchMock).not.toHaveBeenCalled();
  });
});
