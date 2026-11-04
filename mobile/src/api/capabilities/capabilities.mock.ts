import { mockCall } from '@/api/mock/runtime';
import { capabilitiesSchema } from '@/api/schemas';
import type { CapabilitiesApi } from '@/api/types';
import { SUPPORTED_LANGUAGES } from '@/domain/languages';

/** Everything is available, so every screen can be built before the server has it. */
export const createMockCapabilities = (): CapabilitiesApi => ({
  get: () =>
    mockCall(capabilitiesSchema, () => ({
      languages: [...SUPPORTED_LANGUAGES],
      features: { accounts: true, records: true, summary: true, speakers: true },
    })),
});
