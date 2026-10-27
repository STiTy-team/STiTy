import { mockCall } from '@/api/mock/runtime';
import { mockStore } from '@/api/mock/store';
import { profileSchema } from '@/api/schemas';
import type { ProfileApi } from '@/api/types';

export const createMockProfile = (): ProfileApi => ({
  get: () => mockCall(profileSchema, async () => (await mockStore.read()).profile),
  update: (patch) =>
    mockCall(profileSchema, () =>
      mockStore.update((data) => {
        data.profile = { ...data.profile, ...patch };
        return data.profile;
      }),
    ),
  deleteAccount: () => mockCall(null, () => mockStore.reset()),
});
