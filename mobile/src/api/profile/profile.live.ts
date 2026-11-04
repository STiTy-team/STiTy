import type { HttpClient } from '@/api/http';
import { profileSchema } from '@/api/schemas';
import type { ProfileApi } from '@/api/types';

export const createLiveProfile = (http: HttpClient): ProfileApi => ({
  get: () => http.request({ path: '/me', schema: profileSchema, auth: true }),
  update: (patch) =>
    http.request({ method: 'PATCH', path: '/me', body: patch, schema: profileSchema, auth: true }),
  deleteAccount: () => http.request({ method: 'DELETE', path: '/me', auth: true }),
});
