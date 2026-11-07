import Constants from 'expo-constants';
import { z } from 'zod';

import type { AppVariant } from '@/api/modes';

const variantSchema = z.enum(['development', 'preview', 'production'] satisfies AppVariant[]);

/** Missing means a local run (Expo Go, tests); a value that isn't a variant is a broken build. */
export const parseVariant = (value: unknown): AppVariant => {
  if (value === undefined) return 'development';
  const parsed = variantSchema.safeParse(value);
  if (!parsed.success) throw new Error(`Unknown app variant: ${String(value)}`);
  return parsed.data;
};

/** Missing means a local run (Expo Go, tests), which has no web app to link to. */
export const parseWebUrl = (value: unknown): string =>
  value === undefined ? '' : z.url({ protocol: /^https?$/ }).parse(value);

export const appInfo = {
  version: Constants.expoConfig?.version ?? '0.0.0',
  variant: parseVariant(Constants.expoConfig?.extra?.appVariant),
  /** The hosted web app: `STITY_WEB_URL` at build time. */
  webUrl: parseWebUrl(Constants.expoConfig?.extra?.webUrl),
} as const;
