import { useEffect } from 'react';
import { create } from 'zustand';

import { apiErrorKind, getApi, type ApiErrorKind, type Profile, type ProfileUpdate } from '@/api';

type ProfileState = {
  profile: Profile | null;
  loading: boolean;
  loadError: ApiErrorKind | null;
  /** Set when an optimistic change was rolled back. */
  saveError: ApiErrorKind | null;
  load: () => Promise<void>;
  update: (patch: ProfileUpdate) => Promise<void>;
  dismissSaveError: () => void;
  clear: () => void;
};

export const useProfileStore = create<ProfileState>()((set, get) => ({
  profile: null,
  loading: false,
  loadError: null,
  saveError: null,
  async load() {
    set({ loading: true, loadError: null });
    try {
      set({ profile: await getApi().profile.get(), loading: false });
    } catch (error) {
      set({ loading: false, loadError: apiErrorKind(error) });
    }
  },
  /** Shows the change at once and puts the old value back if the server refuses it. */
  async update(patch) {
    const previous = get().profile;
    if (!previous) return;
    set({ profile: { ...previous, ...patch }, saveError: null });
    try {
      set({ profile: await getApi().profile.update(patch) });
    } catch (error) {
      set({ profile: previous, saveError: apiErrorKind(error) });
    }
  },
  dismissSaveError: () => set({ saveError: null }),
  clear: () => set({ profile: null, loading: false, loadError: null, saveError: null }),
}));

/** Loads once per mount. After a failure only an explicit retry (`load`) asks again. */
export const useLoadProfileOnMount = () => {
  useEffect(() => {
    const { profile, load } = useProfileStore.getState();
    if (!profile) void load();
  }, []);
};
