import { create } from 'zustand';

import type { ServerSnapshot } from '@/api';

type ServerState = ServerSnapshot & {
  setSnapshot: (snapshot: ServerSnapshot) => void;
  setCapacity: (capacity: ServerSnapshot['capacity']) => void;
};

export const useServerStore = create<ServerState>()((set) => ({
  status: 'idle',
  capacity: null,
  setSnapshot: (snapshot) => set(snapshot),
  setCapacity: (capacity) => set({ capacity }),
}));

export const selectServerFull = ({ capacity }: ServerSnapshot): boolean =>
  capacity !== null && capacity.active >= capacity.max;
