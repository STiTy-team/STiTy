import type { z } from 'zod';
import { create } from 'zustand';

import { ApiError, type ApiErrorKind } from '@/api/errors';
import { emitUnauthorized } from '@/api/session-events';

export const MOCK_FAILURES = ['offline', 'unauthorized', 'server'] as const;

export type MockFailure = (typeof MOCK_FAILURES)[number];

/** The developer switch that forces every mocked call to fail, so error screens get built. */
export const useMockControls = create<{ failure: MockFailure | null }>(() => ({ failure: null }));

export const setMockFailure = (failure: MockFailure | null) =>
  useMockControls.setState({ failure });

const MIN_LATENCY_MS = 150;
const MAX_LATENCY_MS = 600;

export const mockLatency = () =>
  new Promise<void>((resolve) =>
    setTimeout(resolve, MIN_LATENCY_MS + Math.random() * (MAX_LATENCY_MS - MIN_LATENCY_MS)),
  );

const fail = (kind: ApiErrorKind): never => {
  if (kind === 'unauthorized') emitUnauthorized();
  throw new ApiError(kind, `Mock failure: ${kind}`);
};

/** Runs a mocked call with fake latency and the failure switch, validating its result. */
export async function mockCall<T extends z.ZodType>(
  schema: T,
  run: () => unknown,
): Promise<z.infer<T>>;
export async function mockCall(schema: null, run: () => unknown): Promise<void>;
export async function mockCall(schema: z.ZodType | null, run: () => unknown): Promise<unknown> {
  await mockLatency();
  const { failure } = useMockControls.getState();
  if (failure) fail(failure);
  const value = await run();
  return schema ? schema.parse(value) : undefined;
}
