import { router, useFocusEffect } from 'expo-router';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { RefreshControl, ScrollView, SectionList, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { LoadingView, StateView } from '@/components/state-view';
import { TabHeader } from '@/components/tab-header';
import { TextField } from '@/components/text-field';
import { RecordRow } from '@/features/records/components/record-row';
import { daysAgo, groupByDay } from '@/features/records/model/records-model';
import { useRecords } from '@/features/records/model/use-records';
import { useCapabilitiesStore, useFeatures } from '@/stores/capabilities-store';
import { selectSignedIn, useSessionStore } from '@/stores/session-store';
import { makeStyles, spacing, typography, useColors } from '@/theme';

export const RecordsScreen = () => {
  const { t, i18n } = useTranslation();
  const styles = useStyles();
  const colors = useColors();
  const signedIn = useSessionStore(selectSignedIn);
  const restoring = useSessionStore((state) => state.status === 'restoring');
  const features = useFeatures();
  const capabilitiesLoaded = useCapabilitiesStore((state) => state.loaded || state.failed);
  const [search, setSearch] = useState('');
  const enabled = signedIn && features.records;
  const records = useRecords({ enabled, search: search.trim() });

  // A conversation that just ended should be at the top when the tab comes back into view.
  const focusedBefore = useRef(false);
  const refresh = useRef(records.refresh);
  useEffect(() => {
    refresh.current = records.refresh;
  });
  useFocusEffect(
    useCallback(() => {
      if (focusedBefore.current && enabled) refresh.current();
      focusedBefore.current = true;
    }, [enabled]),
  );

  const sections = useMemo(() => groupByDay(records.items), [records.items]);
  const dayLabel = (date: Date) => {
    const ago = daysAgo(date);
    if (ago === 0) return t('records.today');
    if (ago === 1) return t('records.yesterday');
    return new Intl.DateTimeFormat(i18n.language, {
      weekday: 'short',
      month: 'short',
      day: 'numeric',
    }).format(date);
  };

  const body = (() => {
    if (restoring || !capabilitiesLoaded) {
      return <LoadingView label={t('records.loading')} />;
    }
    if (!features.records || !features.accounts) {
      return (
        <StateView
          icon="time-outline"
          title={t('records.unavailableTitle')}
          message={t('records.unavailableHint')}
        />
      );
    }
    if (!signedIn) {
      return (
        <StateView
          icon="bookmark-outline"
          title={t('records.guestTitle')}
          message={t('records.guestHint')}
          action={{ label: t('records.signIn'), onPress: () => router.push('/sign-in') }}
        />
      );
    }
    return null;
  })();

  if (body) {
    return (
      <SafeAreaView style={styles.screen} edges={['top', 'left', 'right']}>
        <TabHeader title={t('tabs.records')} />
        <ScrollView contentContainerStyle={styles.fill}>{body}</ScrollView>
      </SafeAreaView>
    );
  }

  const filtering = search.trim() !== '';

  return (
    <SafeAreaView style={styles.screen} edges={['top', 'left', 'right']}>
      <TabHeader title={t('tabs.records')} />
      <View style={styles.search}>
        <TextField
          icon="search"
          value={search}
          onChangeText={setSearch}
          placeholder={t('records.searchPlaceholder')}
          returnKeyType="search"
          autoCorrect={false}
          clearButtonMode="while-editing"
        />
      </View>

      {records.status === 'loading' ? (
        <LoadingView label={t('records.loading')} />
      ) : (
        <SectionList
          sections={sections}
          keyExtractor={(item) => item.id}
          renderItem={({ item }) => (
            <RecordRow
              record={item}
              onPress={(id) => router.push({ pathname: '/records/[id]', params: { id } })}
            />
          )}
          renderSectionHeader={({ section }) => (
            <Text style={styles.day} accessibilityRole="header">
              {dayLabel(section.date)}
            </Text>
          )}
          stickySectionHeadersEnabled
          contentContainerStyle={styles.fill}
          keyboardDismissMode="on-drag"
          onEndReached={records.loadMore}
          onEndReachedThreshold={0.5}
          refreshControl={
            <RefreshControl
              refreshing={records.status === 'refreshing'}
              onRefresh={records.refresh}
              tintColor={colors.textTertiary}
            />
          }
          ListEmptyComponent={
            records.error ? (
              <StateView
                icon="cloud-offline-outline"
                title={t('records.loadFailed')}
                message={t(`apiErrors.${records.error}`)}
                action={{ label: t('common.retry'), onPress: records.retry }}
              />
            ) : filtering ? (
              <StateView icon="search" title={t('records.noResults')} />
            ) : (
              <StateView
                icon="chatbubbles-outline"
                title={t('records.emptyTitle')}
                message={t('records.emptyHint')}
              />
            )
          }
          ListFooterComponent={
            records.status === 'loadingMore' ? (
              <LoadingView label={t('records.loading')} />
            ) : undefined
          }
        />
      )}
    </SafeAreaView>
  );
};

const useStyles = makeStyles((colors) => ({
  screen: { flex: 1, backgroundColor: colors.background },
  fill: { flexGrow: 1 },
  search: { paddingHorizontal: spacing.lg, paddingBottom: spacing.sm },
  day: {
    ...typography.small,
    color: colors.textTertiary,
    paddingHorizontal: spacing.lg,
    paddingTop: spacing.lg,
    paddingBottom: spacing.xs,
    backgroundColor: colors.background,
  },
}));
