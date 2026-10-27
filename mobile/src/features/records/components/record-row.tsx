import type { TFunction } from 'i18next';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { Pressable, Text, View } from 'react-native';

import type { RecordSummary } from '@/domain/records';
import { makeStyles, spacing, typography } from '@/theme';

export const formatDuration = (t: TFunction, ms: number): string => {
  const minutes = Math.round(ms / 60_000);
  if (minutes < 1) return t('records.underMinute');
  if (minutes < 60) return t('records.minutes', { count: minutes });
  return t('records.hoursMinutes', { hours: Math.floor(minutes / 60), minutes: minutes % 60 });
};

type Props = { record: RecordSummary; onPress: (id: string) => void };

/** Title and time, then languages and length, then how it began. */
export const RecordRow = memo(function RecordRow({ record, onPress }: Props) {
  const { t, i18n } = useTranslation();
  const styles = useStyles();
  const time = new Intl.DateTimeFormat(i18n.language, {
    hour: 'numeric',
    minute: '2-digit',
  }).format(new Date(record.startedAt));
  const title = record.title || t('records.untitled');
  const languages = record.languages.map((code) => code.toUpperCase()).join(' ↔ ');
  const meta = `${languages} · ${formatDuration(t, record.durationMs)}`;

  return (
    <Pressable
      onPress={() => onPress(record.id)}
      accessibilityRole="button"
      accessibilityLabel={`${title}. ${time}. ${meta}`}
      style={({ pressed }) => [styles.row, pressed && styles.pressed]}
    >
      <View style={styles.top}>
        <Text style={styles.title} numberOfLines={1}>
          {title}
        </Text>
        <Text style={styles.time}>{time}</Text>
      </View>
      <Text style={styles.meta} numberOfLines={1}>
        {meta}
      </Text>
      {!!record.firstLine && (
        <Text style={styles.firstLine} numberOfLines={1}>
          {record.firstLine}
        </Text>
      )}
    </Pressable>
  );
});

const useStyles = makeStyles((colors) => ({
  row: {
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.md,
    gap: 3,
    borderBottomWidth: 1,
    borderBottomColor: colors.border,
  },
  pressed: { backgroundColor: colors.surface },
  top: { flexDirection: 'row', alignItems: 'baseline', gap: spacing.sm },
  title: { ...typography.bodyStrong, flex: 1, color: colors.text },
  time: { ...typography.small, color: colors.textTertiary, fontVariant: ['tabular-nums'] },
  meta: { ...typography.small, color: colors.textTertiary },
  firstLine: { ...typography.caption, color: colors.textSecondary },
}));
