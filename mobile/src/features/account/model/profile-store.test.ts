import { renderHook, waitFor } from '@testing-library/react-native';

import { getApi, setMockFailure } from '@/api';

import { useLoadProfileOnMount, useProfileStore } from './profile-store';

jest.mock('@/api/mock/runtime', () => ({
  ...jest.requireActual('@/api/mock/runtime'),
  mockLatency: () => Promise.resolve(),
}));

afterEach(() => {
  setMockFailure(null);
  jest.restoreAllMocks();
  useProfileStore.getState().clear();
});

describe('loading the profile', () => {
  it('asks once per mount when the load fails, and again only on retry', async () => {
    const get = jest.spyOn(getApi().profile, 'get');
    setMockFailure('server');

    const { rerender } = await renderHook(() => useLoadProfileOnMount());
    await waitFor(() => expect(useProfileStore.getState().loadError).toBe('server'));
    await rerender({});
    await new Promise((resolve) => setTimeout(resolve, 20));
    expect(get).toHaveBeenCalledTimes(1);

    setMockFailure(null);
    await useProfileStore.getState().load();
    expect(get).toHaveBeenCalledTimes(2);
    expect(useProfileStore.getState().profile).not.toBeNull();
  });
});

describe('profile changes', () => {
  it('save at once and roll back with a message when the call fails', async () => {
    await useProfileStore.getState().load();
    const before = useProfileStore.getState().profile!;

    setMockFailure('server');
    const pending = useProfileStore.getState().update({ displayName: 'Renamed' });
    expect(useProfileStore.getState().profile?.displayName).toBe('Renamed');
    await pending;

    expect(useProfileStore.getState().profile?.displayName).toBe(before.displayName);
    expect(useProfileStore.getState().saveError).toBe('server');
  });
});
