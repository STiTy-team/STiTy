import { create } from 'zustand';

import { getApi, type FeatureFlags } from '@/api';
import { createLogger } from '@/lib/logger';

const log = createLogger('capabilities');

/** Until the server answers, nothing beyond streaming is assumed. */
const STREAMING_ONLY: FeatureFlags = {
  accounts: false,
  records: false,
  summary: false,
  speakers: false,
};

type CapabilitiesState = {
  features: FeatureFlags;
  loaded: boolean;
  failed: boolean;
};

export const useCapabilitiesStore = create<CapabilitiesState>()(() => ({
  features: STREAMING_ONLY,
  loaded: false,
  failed: false,
}));

/** Asks the server what it can do; a failure keeps streaming only and is retried later. */
export const loadCapabilities = async (): Promise<void> => {
  try {
    const { features } = await getApi().capabilities.get();
    useCapabilitiesStore.setState({ features, loaded: true, failed: false });
  } catch (error) {
    log.warn('capabilities unavailable', error);
    useCapabilitiesStore.setState({ failed: true });
  }
};

export const useFeatures = (): FeatureFlags => useCapabilitiesStore((state) => state.features);
