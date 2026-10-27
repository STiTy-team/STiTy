import type { z } from 'zod';

import { tokenExpiry, tokensSchema, type TokenStore } from '@/api/auth/tokens';
import { ApiError } from '@/api/errors';
import { emitUnauthorized } from '@/api/session-events';

const TIMEOUT_MS = 15_000;
/** A token this close to expiry is refreshed before it is handed out. */
const EXPIRY_MARGIN_MS = 60_000;
const API_PREFIX = '/api/v1';

type Method = 'GET' | 'POST' | 'PATCH' | 'DELETE';

type Request = {
  method?: Method;
  path: string;
  query?: Record<string, string | undefined>;
  body?: unknown;
  /** Leave out for endpoints that return no body. */
  schema?: z.ZodType;
  /** Sends the access token; on 401 refreshes it once, and signs out if the server refuses that. */
  auth?: boolean;
};

export type HttpClient = {
  request<T extends z.ZodType>(request: Request & { schema: T }): Promise<z.infer<T>>;
  request(request: Omit<Request, 'schema'>): Promise<void>;
  /** The stored access token, refreshed first when it is about to expire; null when signed out. */
  accessToken(): Promise<string | null>;
};

/** `rejected`: the server refused the refresh token. Network and server failures throw instead. */
type RefreshResult = 'ok' | 'rejected';

export const createHttpClient = (baseUrl: string, tokens: TokenStore): HttpClient => {
  let refreshing: Promise<RefreshResult> | null = null;

  const send = async (request: Request, accessToken?: string): Promise<Response> => {
    const query = new URLSearchParams(
      Object.entries(request.query ?? {}).filter((entry): entry is [string, string] => !!entry[1]),
    ).toString();
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), TIMEOUT_MS);
    try {
      return await fetch(`${baseUrl}${API_PREFIX}${request.path}${query ? `?${query}` : ''}`, {
        method: request.method ?? 'GET',
        headers: {
          Accept: 'application/json',
          ...(request.body === undefined ? {} : { 'Content-Type': 'application/json' }),
          ...(accessToken ? { Authorization: `Bearer ${accessToken}` } : {}),
        },
        body: request.body === undefined ? undefined : JSON.stringify(request.body),
        signal: controller.signal,
      });
    } catch {
      throw new ApiError('offline');
    } finally {
      clearTimeout(timeout);
    }
  };

  const refresh = (): Promise<RefreshResult> =>
    (refreshing ??= (async (): Promise<RefreshResult> => {
      const startedIn = tokens.epoch();
      const current = await tokens.read();
      if (!current) return 'rejected';
      const response = await send({
        method: 'POST',
        path: '/auth/refresh',
        body: { refreshToken: current.refreshToken },
      });
      if (response.status === 401 || response.status === 403) return 'rejected';
      if (!response.ok) throw new ApiError('server', `Refresh failed: HTTP ${response.status}`);
      const next = tokensSchema.safeParse(await response.json().catch(() => undefined));
      if (!next.success) throw new ApiError('server', 'Refresh returned no tokens');
      if (tokens.epoch() !== startedIn) return 'rejected';
      await tokens.write(next.data);
      return 'ok';
    })().finally(() => {
      refreshing = null;
    }));

  /** Skipped when the user signed out since `epoch`, so a late failure can't sign out twice. */
  const signOut = async (epoch: number) => {
    if (tokens.epoch() !== epoch) return;
    await tokens.clear();
    emitUnauthorized();
  };

  /** A token newer than the refused one came from a parallel refresh, so it is tried first. */
  const renewedToken = async (refused: string, epoch: number): Promise<string> => {
    const latest = (await tokens.read())?.accessToken;
    if (latest && latest !== refused) return latest;
    if ((await refresh()) === 'ok') {
      const renewed = (await tokens.read())?.accessToken;
      if (renewed) return renewed;
    }
    await signOut(epoch);
    throw new ApiError('unauthorized');
  };

  async function request(request: Request): Promise<unknown> {
    const epoch = tokens.epoch();
    const token = request.auth ? (await tokens.read())?.accessToken : undefined;
    if (request.auth && !token) throw new ApiError('unauthorized');

    let response = await send(request, token);
    if (response.status === 401 && token) {
      response = await send(request, await renewedToken(token, epoch));
      if (response.status === 401) {
        await signOut(epoch);
        throw new ApiError('unauthorized');
      }
    }

    if (response.status === 404) throw new ApiError('notFound');
    if (response.status === 401 || response.status === 403) throw new ApiError('unauthorized');
    if (response.status === 400 || response.status === 422) throw new ApiError('invalid');
    if (!response.ok) throw new ApiError('server', `HTTP ${response.status}`);
    if (!request.schema) return undefined;

    const parsed = request.schema.safeParse(await response.json().catch(() => undefined));
    if (!parsed.success) throw new ApiError('invalid', parsed.error.message);
    return parsed.data;
  }

  const accessToken = async (): Promise<string | null> => {
    const epoch = tokens.epoch();
    const current = await tokens.read();
    if (!current) return null;
    const expiry = tokenExpiry(current.accessToken);
    if (expiry === null || expiry - Date.now() > EXPIRY_MARGIN_MS) return current.accessToken;
    if ((await refresh()) === 'ok') return (await tokens.read())?.accessToken ?? null;
    await signOut(epoch);
    return null;
  };

  return { request, accessToken } as HttpClient;
};
