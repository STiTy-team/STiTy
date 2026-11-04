import { useMemo, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import { FlatList, Text, View, type ListRenderItem } from 'react-native';

import { useTypewriter } from '@/features/conversation/hooks/use-typewriter';
import { useConversationStore } from '@/features/conversation/model/conversation-store';
import { buildFeed, type FeedItem } from '@/features/conversation/model/feed';
import { usePreferencesStore } from '@/stores/preferences-store';
import { makeStyles, spacing, typography } from '@/theme';

import { FeedEventRow } from './feed-event';
import { TranscriptItem } from './transcript-item';

export const TranscriptFeed = () => {
  const { t } = useTranslation();
  const styles = useStyles();
  const listRef = useRef<FlatList<FeedItem>>(null);
  const entries = useConversationStore((state) => state.entries);
  const events = useConversationStore((state) => state.events);
  const partial = useConversationStore((state) => state.partial);
  const speechPending = useConversationStore((state) => state.speechPending);
  const myLanguage = usePreferencesStore((state) => state.myLanguage);
  const { shown } = useTypewriter(partial?.text ?? '');

  const items = useMemo(() => buildFeed(entries, events), [entries, events]);

  const renderItem: ListRenderItem<FeedItem> = ({ item }) =>
    item.kind === 'event' ? (
      <FeedEventRow event={item.event} />
    ) : (
      <TranscriptItem
        language={item.entry.language}
        text={item.entry.text}
        translation={item.entry.translation}
        mine={item.entry.language === myLanguage}
        entrance={item.entry.fromPartial ? 'firm' : 'appear'}
      />
    );

  const live = partial ? (
    <TranscriptItem
      language={partial.language}
      text={shown}
      translation=""
      mine={partial.language === myLanguage}
      entrance="partial"
    />
  ) : speechPending ? (
    <View style={styles.pending} accessibilityLabel={t('conversation.transcribing')}>
      <Text style={styles.pendingText}>…</Text>
    </View>
  ) : null;

  return (
    <FlatList
      ref={listRef}
      data={items}
      keyExtractor={(item) => (item.kind === 'entry' ? item.entry.id : item.event.id)}
      renderItem={renderItem}
      style={styles.list}
      contentContainerStyle={styles.content}
      onContentSizeChange={() => listRef.current?.scrollToEnd({ animated: true })}
      ListEmptyComponent={
        live ? undefined : (
          <View style={styles.empty}>
            <Text style={styles.emptyTitle}>{t('conversation.emptyTitle')}</Text>
            <Text style={styles.emptyHint}>{t('conversation.emptyHint')}</Text>
          </View>
        )
      }
      ListFooterComponent={live ?? undefined}
    />
  );
};

const useStyles = makeStyles((colors) => ({
  list: { flex: 1 },
  content: { flexGrow: 1, justifyContent: 'flex-end', paddingTop: spacing.lg },
  empty: { flex: 1, alignItems: 'center', justifyContent: 'center', padding: spacing.xxxl },
  emptyTitle: { ...typography.title, color: colors.text, textAlign: 'center' },
  emptyHint: {
    ...typography.caption,
    color: colors.textTertiary,
    textAlign: 'center',
    marginTop: spacing.xs,
  },
  pending: { marginHorizontal: spacing.lg, marginBottom: spacing.xl },
  pendingText: { ...typography.translation, color: colors.textTertiary },
}));
