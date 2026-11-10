import { reloadAppAsync } from 'expo';

import { getApi } from '@/api';
import { serverOverride } from '@/config/server-override';

import { changeServer } from './account-runtime';

jest.mock('expo', () => ({ ...jest.requireActual('expo'), reloadAppAsync: jest.fn() }));

describe('changeServer', () => {
  beforeEach(() => jest.clearAllMocks());

  it('signs out before it saves the new server and restarts', async () => {
    const order: string[] = [];
    jest.spyOn(getApi().auth, 'signOut').mockImplementation(async () => {
      order.push('signOut');
    });
    jest.spyOn(serverOverride, 'write').mockImplementation(() => {
      order.push('write');
    });
    jest.mocked(reloadAppAsync).mockImplementation(async () => {
      order.push('reload');
    });

    await changeServer('wss://next.example.com');

    expect(order).toEqual(['signOut', 'write', 'reload']);
    expect(serverOverride.write).toHaveBeenCalledWith('wss://next.example.com');
  });

  it('restarts without waiting for an old server that never answers the sign-out', async () => {
    jest.useFakeTimers();
    jest.spyOn(getApi().auth, 'signOut').mockReturnValue(new Promise(() => {}));
    const clear = jest.spyOn(serverOverride, 'clear').mockResolvedValue();

    const changing = changeServer(null);
    await jest.advanceTimersByTimeAsync(2_000);
    await changing;

    expect(clear).toHaveBeenCalled();
    expect(reloadAppAsync).toHaveBeenCalled();
    jest.useRealTimers();
  });
});
