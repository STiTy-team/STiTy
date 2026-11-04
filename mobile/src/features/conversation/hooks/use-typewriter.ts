import { useEffect, useRef, useState } from 'react';

import { commonPrefixLength, typingInterval } from '@/features/conversation/model/typewriter';

/**
 * Reveals `target` one character at a time. When the model rewrites the tail of a partial,
 * the shown text rewinds to the common prefix and continues typing from there.
 */
export const useTypewriter = (target: string) => {
  const [shown, setShown] = useState('');
  const shownRef = useRef('');
  const targetRef = useRef(target);

  useEffect(() => {
    targetRef.current = target;

    const keep = commonPrefixLength(shownRef.current, target);
    if (keep < shownRef.current.length) {
      shownRef.current = target.slice(0, keep);
      setShown(shownRef.current);
    }

    const backlog = target.length - shownRef.current.length;
    if (backlog <= 0) return;

    const timer = setInterval(() => {
      if (shownRef.current.length >= targetRef.current.length) {
        clearInterval(timer);
        return;
      }
      shownRef.current = targetRef.current.slice(0, shownRef.current.length + 1);
      setShown(shownRef.current);
    }, typingInterval(backlog));
    return () => clearInterval(timer);
  }, [target]);

  return { shown, typing: shown.length < target.length };
};
