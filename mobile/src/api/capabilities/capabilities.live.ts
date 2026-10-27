import { ApiError } from '@/api/errors';
import type { HttpClient } from '@/api/http';
import { capabilitiesSchema } from '@/api/schemas';
import type { Capabilities, CapabilitiesApi } from '@/api/types';
import { SUPPORTED_LANGUAGES } from '@/domain/languages';

/** What a server without the REST API can do: live translation and nothing else. */
export const STREAMING_ONLY: Capabilities = {
  languages: [...SUPPORTED_LANGUAGES],
  features: { accounts: false, records: false, summary: false, speakers: false },
};

export const createLiveCapabilities = (http: HttpClient): CapabilitiesApi => ({
  async get() {
    try {
      return await http.request({ path: '/capabilities', schema: capabilitiesSchema });
    } catch (error) {
      if (error instanceof ApiError && error.kind === 'notFound') return STREAMING_ONLY;
      throw error;
    }
  },
});
