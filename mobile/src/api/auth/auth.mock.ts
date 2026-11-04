import { ApiError } from '@/api/errors';
import { mockCall } from '@/api/mock/runtime';
import { mockStore } from '@/api/mock/store';
import { userSchema } from '@/api/schemas';
import type { AuthApi } from '@/api/types';

/** Accepted by the mock email flow, for any address. */
export const MOCK_EMAIL_CODE = '000000';

const signIn = () =>
  mockCall(userSchema, () =>
    mockStore.update((data) => {
      data.signedIn = true;
      return data.profile;
    }),
  );

/** Apple and Google return the seeded user; the email flow takes any address and 000000. */
export const createMockAuth = (): AuthApi => ({
  methods: ['apple', 'google', 'email'],
  restore: async () => {
    const data = await mockStore.read();
    return data.signedIn ? userSchema.parse(data.profile) : null;
  },
  signInWithApple: signIn,
  signInWithGoogle: signIn,
  requestEmailCode: (email) =>
    mockCall(null, () => {
      if (!email.includes('@')) throw new ApiError('invalid');
    }),
  verifyEmailCode: (_email, code) =>
    code === MOCK_EMAIL_CODE
      ? signIn()
      : mockCall(null, () => {
          throw new ApiError('invalid');
        }).then(signIn),
  signOut: () =>
    mockStore.update((data) => {
      data.signedIn = false;
    }),
  accessToken: async () => ((await mockStore.read()).signedIn ? 'mock-access-token' : null),
});
