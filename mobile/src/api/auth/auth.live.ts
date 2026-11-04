import type { TokenStore } from '@/api/auth/tokens';
import { ApiError } from '@/api/errors';
import type { HttpClient } from '@/api/http';
import { authResponseSchema, userSchema } from '@/api/schemas';
import type { AuthApi, User } from '@/api/types';

/**
 * Accounts on the STiTy server. Apple and Google hand a provider token to `/auth/apple` and
 * `/auth/google`; until the native sign-in SDKs are added, only the email code is offered.
 */
export const createLiveAuth = (http: HttpClient, tokens: TokenStore): AuthApi => {
  const exchange = async (path: string, body: unknown): Promise<User> => {
    const { user, accessToken, refreshToken } = await http.request({
      method: 'POST',
      path,
      body,
      schema: authResponseSchema,
    });
    await tokens.write({ accessToken, refreshToken });
    return user;
  };

  const nativeSdkMissing = async (): Promise<User> => {
    throw new ApiError('invalid', 'This sign-in method needs its native SDK');
  };

  return {
    methods: ['email'],
    async restore() {
      if (!(await tokens.read())) return null;
      try {
        return await http.request({ path: '/me', schema: userSchema, auth: true });
      } catch (error) {
        if (error instanceof ApiError && error.kind === 'unauthorized') return null;
        throw error;
      }
    },
    signInWithApple: nativeSdkMissing,
    signInWithGoogle: nativeSdkMissing,
    requestEmailCode: (email) =>
      http.request({ method: 'POST', path: '/auth/email/code', body: { email } }),
    verifyEmailCode: (email, code) => exchange('/auth/email/verify', { email, code }),
    async signOut() {
      const current = await tokens.read();
      await tokens.clear();
      if (current) {
        await http
          .request({
            method: 'POST',
            path: '/auth/logout',
            body: { refreshToken: current.refreshToken },
          })
          .catch(() => undefined);
      }
    },
    accessToken: () => http.accessToken(),
  };
};
