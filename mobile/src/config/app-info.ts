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

export const appInfo = {
  version: Constants.expoConfig?.version ?? '0.0.0',
  variant: parseVariant(Constants.expoConfig?.extra?.appVariant),
} as const;
