import * as Clipboard from 'expo-clipboard';
import { useLocalSearchParams } from 'expo-router';
import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import { useTranslation } from 'react-i18next';
import {
  FlatList,
  Pressable,
  ScrollView,
  Share,
  Text,
  View,
  type ListRenderItem,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { apiErrorKind, getApi, type ApiErrorKind } from '@/api';
import { BottomSheet } from '@/components/bottom-sheet';
import { Chip } from '@/components/chip';
import { showDialog, showPrompt } from '@/components/dialog';
import { IconButton } from '@/components/icon-button';
import { ListRow, ListSection } from '@/components/list';
import { goBack } from '@/components/go-back';
import { ScreenHeader } from '@/components/screen-header';
import { SegmentedControl } from '@/components/segmented-control';
import { LoadingView, StateView } from '@/components/state-view';
import { TextField } from '@/components/text-field';
import { formatOffset, type TranscriptDocument, type TranscriptParagraph } from '@/domain/records';
import { ParagraphItem } from '@/features/records/components/paragraph-item';
import { formatDuration } from '@/features/records/components/record-row';
import { SummaryCard } from '@/features/records/components/summary-card';
import {
  READING_SCALES,
  TRANSCRIPT_VIEWS,
  useReadingPreferences,
  type TranscriptView,
} from '@/features/records/model/reading-preferences';
import { findMatches, transcriptText } from '@/features/records/model/records-model';
import { track } from '@/services/telemetry';
import { makeStyles, radius, spacing, typography } from '@/theme';

const VIEWABILITY = { itemVisiblePercentThreshold: 10 };

type Load =
  | { status: 'loading' }
  | { status: 'ready'; doc: TranscriptDocument }
  | { status: 'failed'; error: ApiErrorKind };

/** What a paragraph shows in each view. My language falls back to the original when untranslated. */
const textsFor = (paragraph: TranscriptParagraph, myLanguage: string, view: TranscriptView) => {
  const inMine =
    paragraph.language === myLanguage
      ? paragraph.original
      : paragraph.translation || paragraph.original;
  switch (view) {
    case 'original':
      return { primary: paragraph.original, secondary: '' };
    case 'mine':
      return { primary: inMine, secondary: '' };
    case 'bilingual':
      return {
        primary: inMine,
        secondary: inMine === paragraph.original ? '' : paragraph.original,
      };
  }
};

/**
 * A saved conversation for reading afterwards: title, summary, then the transcript top-down as
 * paragraphs per speaker. Settings changes from the live feed are not shown; they are noise here.
 */
export const TranscriptScreen = () => {
  const { id } = useLocalSearchParams<{ id: string }>();
  const { t, i18n } = useTranslation();
  const styles = useStyles();
  const view = useReadingPreferences((state) => state.transcriptView);
  const scaleIndex = useReadingPreferences((state) => state.readingScale);
  const { setTranscriptView, setReadingScale } = useReadingPreferences.getState();
  const [load, setLoad] = useState<Load>({ status: 'loading' });
  const [menuOpen, setMenuOpen] = useState(false);
  const [searching, setSearching] = useState(false);
  const [query, setQuery] = useState('');
  const [matchIndex, setMatchIndex] = useState(0);
  const [copied, setCopied] = useState(false);
  const listRef = useRef<FlatList<TranscriptParagraph>>(null);
  const topVisible = useRef(0);

  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    let cancelled = false;
    getApi()
      .conversations.get(id)
      .then(
        (doc) => {
          if (cancelled) return;
          setLoad({ status: 'ready', doc });
          track({ name: 'record_opened' });
        },
        (error: unknown) => !cancelled && setLoad({ status: 'failed', error: apiErrorKind(error) }),
      );
    return () => {
      cancelled = true;
    };
  }, [id, attempt]);

  const retry = () => {
    setLoad({ status: 'loading' });
    setAttempt((count) => count + 1);
  };

  const doc = load.status === 'ready' ? load.doc : null;
  const paragraphs = useMemo(() => doc?.paragraphs ?? [], [doc]);
  const speakers = useMemo(() => new Map(doc?.speakers.map((s) => [s.id, s])), [doc]);
  const matches = useMemo(() => findMatches(paragraphs, query), [paragraphs, query]);
  const activeMatch = matches[Math.min(matchIndex, matches.length - 1)];
  const scale = READING_SCALES[scaleIndex] ?? 1;

  const labelOf = useCallback(
    (paragraph: TranscriptParagraph) => {
      const speaker = speakers.get(paragraph.speakerId);
      const who = speaker?.isMe
        ? t('viewer.me')
        : (speaker?.label ?? paragraph.language.toUpperCase());
      return `${who} · ${formatOffset(paragraph.startMs)}`;
    },
    [speakers, t],
  );

  const scrollToParagraph = useCallback(
    (paragraphId: string) => {
      const index = paragraphs.findIndex((paragraph) => paragraph.id === paragraphId);
      if (index >= 0) listRef.current?.scrollToIndex({ index, animated: true, viewPosition: 0.3 });
    },
    [paragraphs],
  );

  useEffect(() => {
    if (activeMatch) scrollToParagraph(activeMatch.paragraphId);
  }, [activeMatch, scrollToParagraph]);

  // A different view changes every paragraph's height; keep the paragraph at the top in place.
  const changeView = (next: TranscriptView) => {
    const index = topVisible.current;
    setTranscriptView(next);
    requestAnimationFrame(() =>
      listRef.current?.scrollToIndex({ index, animated: false, viewPosition: 0 }),
    );
  };

  const onViewableItemsChanged = useCallback(
    ({ viewableItems }: { viewableItems: { index?: number | null }[] }) => {
      const first = viewableItems.find((item) => item.index != null);
      if (first?.index != null) topVisible.current = first.index;
    },
    [],
  );

  const copyParagraph = useCallback(
    async (paragraphId: string) => {
      const paragraph = paragraphs.find((item) => item.id === paragraphId);
      if (!paragraph || !doc) return;
      const { primary, secondary } = textsFor(paragraph, doc.myLanguage, view);
      await Clipboard.setStringAsync([primary, secondary].filter(Boolean).join('\n'));
      setCopied(true);
      setTimeout(() => setCopied(false), 1_500);
    },
    [paragraphs, doc, view],
  );

  if (load.status === 'loading') {
    return (
      <Frame title="">
        <LoadingView label={t('viewer.loading')} />
      </Frame>
    );
  }
  if (load.status === 'failed') {
    return (
      <Frame title="">
        <StateView
          icon={load.error === 'notFound' ? 'trash-outline' : 'cloud-offline-outline'}
          title={t(load.error === 'notFound' ? 'viewer.notFound' : 'viewer.loadFailed')}
          message={load.error === 'notFound' ? undefined : t(`apiErrors.${load.error}`)}
          action={
            load.error === 'notFound'
              ? { label: t('common.back'), onPress: goBack }
              : { label: t('common.retry'), onPress: retry }
          }
        />
      </Frame>
    );
  }

  const { doc: document } = load;
  const showSummary = document.summary.status !== 'unavailable';
  const started = new Date(document.startedAt);
  const durationMs = Date.parse(document.endedAt) - started.getTime();
  const dateText = new Intl.DateTimeFormat(i18n.language, {
    dateStyle: 'medium',
    timeStyle: 'short',
  }).format(started);

  const rename = async () => {
    setMenuOpen(false);
    const title = await showPrompt({
      title: t('viewer.renameTitle'),
      value: document.title,
      confirmLabel: t('common.save'),
      cancelLabel: t('common.cancel'),
      maxLength: 80,
    });
    if (!title || title === document.title) return;
    setLoad({ status: 'ready', doc: { ...document, title } });
    try {
      await getApi().conversations.rename(document.id, title);
    } catch (error) {
      setLoad({ status: 'ready', doc: document });
      await showDialog({
        title: t('viewer.renameFailed'),
        message: t(`apiErrors.${apiErrorKind(error)}`),
        confirmLabel: t('common.ok'),
      });
    }
  };

  const share = async () => {
    setMenuOpen(false);
    const message = transcriptText(document.title, paragraphs, labelOf, (paragraph) => {
      const { primary, secondary } = textsFor(paragraph, document.myLanguage, view);
      return [primary, secondary].filter(Boolean).join('\n');
    });
    await Share.share({ title: document.title, message });
  };

  const remove = async () => {
    setMenuOpen(false);
    const confirmed = await showDialog({
      title: t('viewer.deleteConfirmTitle'),
      message: t('viewer.deleteConfirmMessage'),
      confirmLabel: t('viewer.delete'),
      cancelLabel: t('common.cancel'),
      destructive: true,
    });
    if (!confirmed) return;
    try {
      await getApi().conversations.remove(document.id);
      goBack();
    } catch (error) {
      await showDialog({
        title: t('viewer.deleteFailed'),
        message: t(`apiErrors.${apiErrorKind(error)}`),
        confirmLabel: t('common.ok'),
      });
    }
  };

  const jumpToTime = (startMs: number) => {
    const index = paragraphs.findIndex((paragraph) => paragraph.endMs >= startMs);
    if (index >= 0) listRef.current?.scrollToIndex({ index, animated: true, viewPosition: 0 });
  };

  const renderItem: ListRenderItem<TranscriptParagraph> = ({ item }) => {
    const { primary, secondary } = textsFor(item, document.myLanguage, view);
    return (
      <ParagraphItem
        id={item.id}
        label={labelOf(item)}
        isMe={speakers.get(item.speakerId)?.isMe ?? false}
        primary={primary}
        secondary={secondary}
        query={searching ? query : ''}
        active={searching && activeMatch?.paragraphId === item.id}
        scale={scale}
        copyLabel={t('viewer.copy')}
        onLongPress={(paragraphId) => void copyParagraph(paragraphId)}
      />
    );
  };

  const header = (
    <View style={styles.document}>
      <Text style={styles.title} accessibilityRole="header">
        {document.title || t('records.untitled')}
      </Text>
      <Text style={styles.meta}>
        {dateText} · {formatDuration(t, durationMs)}
      </Text>
      <Text style={styles.meta}>
        {document.languages.map((code) => code.toUpperCase()).join(' ↔ ')}
      </Text>
      {showSummary && <SummaryCard summary={document.summary} />}
      {!!document.chapters?.length && (
        <View style={styles.chapters}>
          <Text style={styles.sectionTitle} accessibilityRole="header">
            {t('viewer.chapters')}
          </Text>
          <ScrollView
            horizontal
            showsHorizontalScrollIndicator={false}
            contentContainerStyle={styles.chipRow}
          >
            {document.chapters.map((chapter) => (
              <Chip
                key={chapter.startMs}
                label={`${formatOffset(chapter.startMs)}  ${chapter.title}`}
                selected={false}
                onPress={() => jumpToTime(chapter.startMs)}
              />
            ))}
          </ScrollView>
        </View>
      )}
    </View>
  );

  const footer = document.glossary?.length ? (
    <View style={styles.glossary}>
      <Text style={styles.sectionTitle} accessibilityRole="header">
        {t('viewer.glossary')}
      </Text>
      {document.glossary.map((entry) => (
        <View key={entry.term} style={styles.glossaryRow}>
          <Text style={styles.term}>{entry.term}</Text>
          <Text style={styles.termTranslation}>{entry.translation}</Text>
        </View>
      ))}
    </View>
  ) : undefined;

  return (
    <Frame
      title={searching ? '' : document.title}
      right={
        <View style={styles.headerActions}>
          <IconButton icon="search" label={t('viewer.search')} onPress={() => setSearching(true)} />
          <IconButton
            icon="ellipsis-horizontal"
            label={t('viewer.more')}
            onPress={() => setMenuOpen(true)}
          />
        </View>
      }
    >
      {searching ? (
        <View style={styles.searchBar}>
          <View style={styles.searchField}>
            <TextField
              icon="search"
              value={query}
              onChangeText={(text) => {
                setQuery(text);
                setMatchIndex(0);
              }}
              placeholder={t('viewer.searchPlaceholder')}
              autoFocus
              returnKeyType="search"
              onSubmitEditing={() =>
                setMatchIndex((index) => (index + 1) % Math.max(1, matches.length))
              }
            />
          </View>
          <Text style={styles.matchCount} accessibilityLiveRegion="polite">
            {query.trim()
              ? matches.length
                ? t('viewer.matches', {
                    current: Math.min(matchIndex, matches.length - 1) + 1,
                    total: matches.length,
                  })
                : t('viewer.noMatches')
              : ''}
          </Text>
          <IconButton
            icon="chevron-up"
            label={t('viewer.previousMatch')}
            onPress={() =>
              setMatchIndex((index) => (index - 1 + matches.length) % Math.max(1, matches.length))
            }
          />
          <IconButton
            icon="chevron-down"
            label={t('viewer.nextMatch')}
            onPress={() => setMatchIndex((index) => (index + 1) % Math.max(1, matches.length))}
          />
          <IconButton
            icon="close"
            label={t('viewer.closeSearch')}
            onPress={() => {
              setSearching(false);
              setQuery('');
            }}
          />
        </View>
      ) : (
        <View style={styles.toolbar}>
          <View style={styles.views}>
            <SegmentedControl
              options={TRANSCRIPT_VIEWS}
              value={view}
              label={(option) => t(`viewer.views.${option}`)}
              onChange={changeView}
            />
          </View>
        </View>
      )}

      <FlatList
        ref={listRef}
        data={paragraphs}
        extraData={`${view}-${scale}-${query}-${activeMatch?.paragraphId}`}
        keyExtractor={(item) => item.id}
        renderItem={renderItem}
        ListHeaderComponent={header}
        ListFooterComponent={footer}
        contentContainerStyle={styles.content}
        initialNumToRender={12}
        maxToRenderPerBatch={16}
        windowSize={9}
        removeClippedSubviews
        keyboardDismissMode="on-drag"
        onViewableItemsChanged={onViewableItemsChanged}
        viewabilityConfig={VIEWABILITY}
        onScrollToIndexFailed={({ index, averageItemLength }) => {
          listRef.current?.scrollToOffset({ offset: index * averageItemLength, animated: false });
          setTimeout(() => listRef.current?.scrollToIndex({ index, animated: false }), 60);
        }}
      />

      {copied && (
        <View style={styles.toast} accessibilityLiveRegion="polite">
          <Text style={styles.toastText}>{t('viewer.copied')}</Text>
        </View>
      )}

      <BottomSheet visible={menuOpen} onClose={() => setMenuOpen(false)}>
        <ListSection>
          <ListRow title={t('viewer.textSize')}>
            <TextSizeButton
              label={t('viewer.textSmaller')}
              glyph="A"
              small
              disabled={scaleIndex <= 0}
              onPress={() => setReadingScale(scaleIndex - 1)}
            />
            <TextSizeButton
              label={t('viewer.textLarger')}
              glyph="A"
              disabled={scaleIndex >= READING_SCALES.length - 1}
              onPress={() => setReadingScale(scaleIndex + 1)}
            />
          </ListRow>
        </ListSection>
        <View style={styles.menuGap} />
        <ListSection>
          <ListRow title={t('viewer.rename')} onPress={() => void rename()} />
          <ListRow title={t('viewer.share')} onPress={() => void share()} />
        </ListSection>
        <View style={styles.menuGap} />
        <ListSection>
          <ListRow title={t('viewer.delete')} tone="destructive" onPress={() => void remove()} />
        </ListSection>
      </BottomSheet>
    </Frame>
  );
};

const Frame = ({
  title,
  right,
  children,
}: {
  title: string;
  right?: ReactNode;
  children: ReactNode;
}) => {
  const { t } = useTranslation();
  const styles = useStyles();
  return (
    <SafeAreaView style={styles.screen} edges={['top', 'left', 'right']}>
      <ScreenHeader
        title={title}
        left={<IconButton icon="chevron-back" label={t('common.back')} onPress={goBack} />}
        right={right}
      />
      {children}
    </SafeAreaView>
  );
};

const TextSizeButton = ({
  label,
  glyph,
  small = false,
  disabled,
  onPress,
}: {
  label: string;
  glyph: string;
  small?: boolean;
  disabled: boolean;
  onPress: () => void;
}) => {
  const styles = useStyles();
  return (
    <Pressable
      onPress={onPress}
      disabled={disabled}
      accessibilityRole="button"
      accessibilityLabel={label}
      accessibilityState={{ disabled }}
      style={({ pressed }) => [
        styles.sizeButton,
        pressed && styles.sizePressed,
        disabled && styles.sizeDisabled,
      ]}
    >
      <Text style={[styles.sizeGlyph, small && styles.sizeGlyphSmall]}>{glyph}</Text>
    </Pressable>
  );
};

const useStyles = makeStyles((colors) => ({
  screen: { flex: 1, backgroundColor: colors.background },
  headerActions: { flexDirection: 'row' },
  toolbar: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.xs,
    paddingHorizontal: spacing.lg,
    paddingBottom: spacing.sm,
    borderBottomWidth: 1,
    borderBottomColor: colors.border,
  },
  views: { flex: 1 },
  sizeButton: {
    width: 44,
    height: 44,
    borderRadius: radius.control,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: colors.surfaceStrong,
  },
  sizePressed: { opacity: 0.7 },
  sizeDisabled: { opacity: 0.4 },
  sizeGlyph: { ...typography.title, fontSize: 19, color: colors.textSecondary },
  sizeGlyphSmall: { fontSize: 14 },
  searchBar: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingLeft: spacing.lg,
    paddingRight: spacing.xs,
    paddingBottom: spacing.sm,
    borderBottomWidth: 1,
    borderBottomColor: colors.border,
  },
  searchField: { flex: 1 },
  matchCount: {
    ...typography.small,
    color: colors.textTertiary,
    minWidth: 52,
    textAlign: 'center',
    fontVariant: ['tabular-nums'],
  },
  content: { paddingBottom: spacing.xxxl * 2 },
  document: {
    paddingHorizontal: spacing.lg,
    paddingTop: spacing.lg,
    paddingBottom: spacing.md,
    gap: spacing.xs,
  },
  title: { ...typography.wordmark, fontSize: 24, lineHeight: 30, color: colors.text },
  meta: { ...typography.caption, color: colors.textTertiary },
  chapters: { marginTop: spacing.md, gap: spacing.sm },
  sectionTitle: { ...typography.small, color: colors.textTertiary, marginLeft: spacing.xs },
  chipRow: { gap: spacing.sm },
  glossary: {
    marginHorizontal: spacing.lg,
    marginTop: spacing.xl,
    gap: spacing.sm,
    paddingTop: spacing.lg,
    borderTopWidth: 1,
    borderTopColor: colors.border,
  },
  glossaryRow: { flexDirection: 'row', gap: spacing.md },
  term: { ...typography.bodyStrong, color: colors.text, minWidth: 96 },
  termTranslation: { ...typography.body, flex: 1, color: colors.textSecondary },
  toast: {
    position: 'absolute',
    bottom: spacing.xxxl,
    alignSelf: 'center',
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.sm,
    borderRadius: radius.pill,
    backgroundColor: colors.text,
  },
  toastText: { ...typography.label, color: colors.background },
  menuGap: { height: spacing.md },
}));
