import { useSyncExternalStore } from 'react';

type PersistedStore = {
  persist: { onFinishHydration: (listener: () => void) => () => void; hasHydrated: () => boolean };
};

/** True once this store's saved values are loaded, so the first frame already shows them. */
export const useHydrated = ({ persist }: PersistedStore): boolean =>
  useSyncExternalStore(persist.onFinishHydration, persist.hasHydrated, persist.hasHydrated);
