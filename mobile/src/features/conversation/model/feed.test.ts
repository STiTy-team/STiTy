import { buildFeed } from './feed';
import type { FeedEvent, TranscriptEntry } from './transcript';

const entry = (id: string): TranscriptEntry => ({
  id,
  language: 'en',
  text: id,
  translation: '',
  fromPartial: false,
  time: 0,
});

const event = (id: string, after: number): FeedEvent => ({
  id,
  after,
  time: 0,
  change: { kind: 'muted', muted: true },
});

const ids = (items: ReturnType<typeof buildFeed>) =>
  items.map((item) => (item.kind === 'entry' ? item.entry.id : item.event.id));

describe('buildFeed', () => {
  it('places each change after the sentences committed before it', () => {
    const feed = buildFeed([entry('a'), entry('b'), entry('c')], [event('x', 1), event('y', 3)]);
    expect(ids(feed)).toEqual(['a', 'x', 'b', 'c', 'y']);
  });

  it('shows changes made before anyone spoke', () => {
    expect(ids(buildFeed([], [event('x', 0)]))).toEqual(['x']);
    expect(ids(buildFeed([entry('a')], [event('x', 0)]))).toEqual(['x', 'a']);
  });
});
