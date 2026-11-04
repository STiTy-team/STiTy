import AsyncStorage from '@react-native-async-storage/async-storage';
import { z } from 'zod';
import { createJSONStorage } from 'zustand/middleware';

/** Every setting used to live under this key, so a store moved off it reads from it once. */
export const LEGACY_PREFERENCES_KEY = 'stity.preferences';

const legacySchema = z.object({ state: z.unknown() });

/**
 * AsyncStorage for a persisted store. Until the store has written its own key, it starts from the
 * settings saved under the old shared key; its `merge` keeps only the fields it owns.
 */
export const persistedStorage = <S>(version: number) =>
  createJSONStorage<S>(() => ({
    getItem: async (name) => {
      const own = await AsyncStorage.getItem(name);
      if (own !== null || name === LEGACY_PREFERENCES_KEY) return own;
      const legacy = await AsyncStorage.getItem(LEGACY_PREFERENCES_KEY);
      if (!legacy) return null;
      try {
        const parsed = legacySchema.safeParse(JSON.parse(legacy));
        return parsed.success ? JSON.stringify({ state: parsed.data.state, version }) : null;
      } catch {
        return null;
      }
    },
    setItem: (name, value) => AsyncStorage.setItem(name, value),
    removeItem: (name) => AsyncStorage.removeItem(name),
  }));

/** For `merge`: a saved field that fails its schema reads as missing, so its default stays. */
export const savedField = <T extends z.ZodType>(schema: T) => schema.optional().catch(undefined);
