import { useCallback, useEffect, useRef, useState } from 'react';

import { apiErrorKind, getApi, type ApiErrorKind } from '@/api';
import type { RecordSummary } from '@/domain/records';

const SEARCH_DEBOUNCE_MS = 250;

type State = {
  items: RecordSummary[];
  cursor: string | null;
  status: 'loading' | 'ready' | 'refreshing' | 'loadingMore';
  error: ApiErrorKind | null;
};

const INITIAL: State = { items: [], cursor: null, status: 'loading', error: null };

/** One page at a time, newest first; a new search starts over. */
export const useRecords = ({ enabled, search }: { enabled: boolean; search: string }) => {
  const [state, setState] = useState<State>(INITIAL);
  const request = useRef(0);

  const fetchPage = useCallback(
    async (mode: 'reset' | 'refresh' | 'more', cursor?: string) => {
      const id = ++request.current;
      setState((current) => ({
        ...current,
        error: null,
        status: mode === 'more' ? 'loadingMore' : mode === 'refresh' ? 'refreshing' : 'loading',
      }));
      try {
        const page = await getApi().conversations.list({
          cursor,
          search: search || undefined,
        });
        if (id !== request.current) return;
        setState((current) => ({
          items: mode === 'more' ? [...current.items, ...page.items] : page.items,
          cursor: page.nextCursor,
          status: 'ready',
          error: null,
        }));
      } catch (error) {
        if (id !== request.current) return;
        setState((current) => ({ ...current, status: 'ready', error: apiErrorKind(error) }));
      }
    },
    [search],
  );

  useEffect(() => {
    if (!enabled) return;
    const timer = setTimeout(() => void fetchPage('reset'), search ? SEARCH_DEBOUNCE_MS : 0);
    return () => clearTimeout(timer);
  }, [enabled, fetchPage, search]);

  return {
    ...state,
    refresh: () => void fetchPage('refresh'),
    retry: () => void fetchPage('reset'),
    loadMore: () => {
      if (state.cursor && state.status === 'ready') void fetchPage('more', state.cursor);
    },
    removeLocally: (id: string) =>
      setState((current) => ({
        ...current,
        items: current.items.filter((item) => item.id !== id),
      })),
  };
};
