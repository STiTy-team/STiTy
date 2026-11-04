export const API_AREAS = ['capabilities', 'auth', 'profile', 'conversations', 'live'] as const;

export type ApiArea = (typeof API_AREAS)[number];

export type ApiMode = 'live' | 'mock';

export type ApiModes = Record<ApiArea, ApiMode>;

export type AppVariant = 'development' | 'preview' | 'production';

/** `live` works today, so outside production everything else starts on mocks. */
const DEFAULT_MOCKS: readonly ApiArea[] = API_AREAS.filter((area) => area !== 'live');

const isApiArea = (value: string): value is ApiArea =>
  (API_AREAS as readonly string[]).includes(value);

export class MockedProductionError extends Error {}

/**
 * Reads EXPO_PUBLIC_API_MOCKS ("auth,profile", "all", "none"). Production refuses any mock, so
 * fixture data can never reach a store build.
 */
export const resolveApiModes = (list: string | undefined, variant: AppVariant): ApiModes => {
  const requested = parseList(list) ?? (variant === 'production' ? [] : DEFAULT_MOCKS);
  if (variant === 'production' && requested.length > 0) {
    throw new MockedProductionError(
      `Production builds can't mock API areas (${requested.join(', ')}).`,
    );
  }
  return Object.fromEntries(
    API_AREAS.map((area) => [area, requested.includes(area) ? 'mock' : 'live']),
  ) as ApiModes;
};

const parseList = (list: string | undefined): ApiArea[] | null => {
  const trimmed = list?.trim().toLowerCase();
  if (!trimmed) return null;
  if (trimmed === 'none') return [];
  if (trimmed === 'all') return [...API_AREAS];
  const areas = trimmed.split(',').map((item) => item.trim());
  const unknown = areas.filter((area) => !isApiArea(area));
  if (unknown.length > 0) throw new Error(`Unknown API areas in EXPO_PUBLIC_API_MOCKS: ${unknown}`);
  return areas.filter(isApiArea);
};

export const mockedAreas = (modes: ApiModes): ApiArea[] =>
  API_AREAS.filter((area) => modes[area] === 'mock');
