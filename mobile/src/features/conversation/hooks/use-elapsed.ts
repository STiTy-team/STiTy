import { useEffect, useState } from 'react';

import { useConversationStore } from '@/features/conversation/model/conversation-store';

/** Seconds spent recording in this conversation; pauses stop the clock. */
export const useElapsedSeconds = (): number => {
  const recordedMs = useConversationStore((state) => state.recordedMs);
  const recordingSince = useConversationStore((state) => state.recordingSince);
  const [now, setNow] = useState(Date.now);

  useEffect(() => {
    if (recordingSince === null) return;
    const timer = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(timer);
  }, [recordingSince]);

  const running = recordingSince === null ? 0 : Math.max(0, now - recordingSince);
  return Math.floor((recordedMs + running) / 1000);
};

export const formatElapsed = (seconds: number): string => {
  const minutes = String(Math.floor(seconds / 60)).padStart(2, '0');
  return `${minutes}:${String(seconds % 60).padStart(2, '0')}`;
};
