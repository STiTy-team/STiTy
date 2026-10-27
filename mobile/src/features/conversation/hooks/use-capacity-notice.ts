import { useEffect, useRef, useState } from 'react';

import { selectServerFull, useServerStore } from '@/stores/server-store';

const FREED_NOTICE_MS = 2_500;

/** Server-full state plus a short "a slot just opened" flag after it stops being full. */
export const useCapacityNotice = () => {
  const capacity = useServerStore((state) => state.capacity);
  const full = useServerStore(selectServerFull);
  const [justFreed, setJustFreed] = useState(false);
  const wasFull = useRef(full);

  useEffect(() => {
    const freed = wasFull.current && !full;
    wasFull.current = full;
    if (!freed) return;
    setJustFreed(true);
    const timer = setTimeout(() => setJustFreed(false), FREED_NOTICE_MS);
    return () => clearTimeout(timer);
  }, [full]);

  return { capacity, full, justFreed };
};
