import { useTranslation } from 'react-i18next';
import { ActivityIndicator, Text, View } from 'react-native';

import type { TranscriptSummary } from '@/domain/records';
import { makeStyles, radius, spacing, typography, useColors } from '@/theme';

const List = ({ title, items }: { title: string; items?: string[] }) => {
  const styles = useStyles();
  if (!items?.length) return null;
  return (
    <View style={styles.group}>
      <Text style={styles.groupTitle} accessibilityRole="header">
        {title}
      </Text>
      {items.map((item, index) => (
        <View key={index} style={styles.item}>
          <View style={styles.bullet} />
          <Text style={styles.itemText}>{item}</Text>
        </View>
      ))}
    </View>
  );
};

/**
 * The server's summary. Pending and unavailable take a single quiet line so the transcript
 * below never jumps when the summary arrives.
 */
export const SummaryCard = ({ summary }: { summary: TranscriptSummary }) => {
  const { t } = useTranslation();
  const styles = useStyles();
  const colors = useColors();

  if (summary.status !== 'ready') {
    return (
      <View style={[styles.card, styles.quiet]} accessibilityLiveRegion="polite">
        {summary.status === 'pending' && (
          <ActivityIndicator size="small" color={colors.textTertiary} />
        )}
        <Text style={styles.quietText}>
          {t(summary.status === 'pending' ? 'viewer.summaryPending' : 'viewer.summaryUnavailable')}
        </Text>
      </View>
    );
  }

  const actionItems = summary.actionItems?.map((item) =>
    item.owner ? `${item.text} — ${item.owner}` : item.text,
  );

  return (
    <View style={styles.card}>
      <Text style={styles.title} accessibilityRole="header">
        {t('viewer.summary')}
      </Text>
      {!!summary.overview && <Text style={styles.overview}>{summary.overview}</Text>}
      <List title={t('viewer.keyPoints')} items={summary.keyPoints} />
      <List title={t('viewer.decisions')} items={summary.decisions} />
      <List title={t('viewer.actionItems')} items={actionItems} />
    </View>
  );
};

const useStyles = makeStyles((colors) => ({
  card: {
    padding: spacing.lg,
    gap: spacing.md,
    borderRadius: radius.card,
    backgroundColor: colors.surface,
  },
  quiet: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm,
    paddingVertical: spacing.md,
  },
  quietText: { ...typography.caption, flex: 1, color: colors.textTertiary },
  title: { ...typography.label, color: colors.textSecondary },
  overview: { ...typography.body, color: colors.text },
  group: { gap: spacing.xs },
  groupTitle: { ...typography.small, color: colors.textTertiary },
  item: { flexDirection: 'row', gap: spacing.sm, alignItems: 'flex-start' },
  bullet: {
    width: 5,
    height: 5,
    borderRadius: 3,
    marginTop: 9,
    backgroundColor: colors.textTertiary,
  },
  itemText: { ...typography.body, flex: 1, color: colors.text },
}));
