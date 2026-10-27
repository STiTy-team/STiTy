import type { FeedEvent, TranscriptEntry } from './transcript';

export type FeedItem =
  { kind: 'entry'; entry: TranscriptEntry } | { kind: 'event'; event: FeedEvent };

/** Interleaves setting changes with sentences, each change after the sentences before it. */
export const buildFeed = (entries: TranscriptEntry[], events: FeedEvent[]): FeedItem[] => {
  const items: FeedItem[] = [];
  let next = 0;
  entries.forEach((entry, index) => {
    while (next < events.length && events[next]!.after <= index) {
      items.push({ kind: 'event', event: events[next++]! });
    }
    items.push({ kind: 'entry', entry });
  });
  while (next < events.length) items.push({ kind: 'event', event: events[next++]! });
  return items;
};
