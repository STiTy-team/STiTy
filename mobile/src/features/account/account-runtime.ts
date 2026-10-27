import { apiMocks, getApi, onUnauthorized, type SignInMethod, type User } from '@/api';
import { createLogger } from '@/lib/logger';
import { useConversationPreferences } from '@/features/conversation/conversation-runtime';
import { useReadingPreferences } from '@/features/records/records-runtime';
import { usePreferencesStore } from '@/stores/preferences-store';
import { useSessionStore } from '@/stores/session-store';

import { useProfileStore } from './model/profile-store';

const log = createLogger('account');

/** Restores the signed-in user from secure storage. A failure leaves the app usable as a guest. */
export const restoreSession = async (): Promise<void> => {
  try {
    useSessionStore.getState().setUser(await getApi().auth.restore());
  } catch (error) {
    log.warn('session restore failed', error);
    useSessionStore.getState().setUser(null);
  }
};

const signedIn = (user: User) => {
  useSessionStore.getState().setUser(user);
  return user;
};

export const signIn = async (method: Exclude<SignInMethod, 'email'>): Promise<User> => {
  const { auth } = getApi();
  return signedIn(await (method === 'apple' ? auth.signInWithApple() : auth.signInWithGoogle()));
};

export const requestEmailCode = (email: string) => getApi().auth.requestEmailCode(email);

export const verifyEmailCode = async (email: string, code: string) =>
  signedIn(await getApi().auth.verifyEmailCode(email, code));

const clearSession = () => {
  useProfileStore.getState().clear();
  useSessionStore.getState().setUser(null);
};

export const signOut = async () => {
  await getApi()
    .auth.signOut()
    .catch((error: unknown) => log.warn('sign-out failed', error));
  clearSession();
};

/** Deletes the account and its records on the server, then everything STiTy keeps on the phone. */
export const deleteAccount = async () => {
  const api = getApi();
  await api.profile.deleteAccount();
  await api.auth.signOut().catch(() => undefined);
  await apiMocks.reset();
  for (const store of [usePreferencesStore, useConversationPreferences, useReadingPreferences]) {
    await store.persist.clearStorage();
    store.getState().reset();
  }
  clearSession();
};

/** Signs the app out when the server stops accepting this user's tokens. */
export const watchSession = (): (() => void) => onUnauthorized(clearSession);
