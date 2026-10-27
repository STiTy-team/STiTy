import { create } from 'zustand';

import type { User } from '@/api';

export type SessionStatus = 'restoring' | 'guest' | 'signedIn';

type SessionState = {
  status: SessionStatus;
  user: User | null;
  setUser: (user: User | null) => void;
};

/** Who is signed in. Live translation never waits on this; only saving does. */
export const useSessionStore = create<SessionState>()((set) => ({
  status: 'restoring',
  user: null,
  setUser: (user) => set({ user, status: user ? 'signedIn' : 'guest' }),
}));

export const selectSignedIn = ({ status }: SessionState) => status === 'signedIn';
