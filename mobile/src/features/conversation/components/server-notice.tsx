import { Ionicons } from '@expo/vector-icons';
import { useTranslation } from 'react-i18next';
import { Text, View } from 'react-native';

import { useCapacityNotice } from '@/features/conversation/hooks/use-capacity-notice';
import { makeStyles, radius, spacing, typography, useColors } from '@/theme';

/**
 * Says why Start is disabled while the server is full, and when it opens up again. Connection
 * problems are shown by the header's status icon instead.
 */
export const ServerNotice = () => {
  const { t } = useTranslation();
  const styles = useStyles();
  const colors = useColors();
  const { full, justFreed } = useCapacityNotice();

  const notice = full
    ? { text: t('server.full'), icon: 'hourglass-outline' as const, warn: true }
    : justFreed
      ? { text: t('server.freed'), icon: 'checkmark-circle-outline' as const, warn: false }
      : null;

  if (!notice) return null;
  return (
    <View
      style={[styles.notice, notice.warn && styles.warn]}
      accessibilityLiveRegion="polite"
      accessibilityRole={notice.warn ? 'alert' : 'text'}
    >
      <Ionicons
        name={notice.icon}
        size={18}
        color={notice.warn ? colors.warning : colors.textSecondary}
      />
      <Text style={[styles.text, notice.warn && styles.warnText]}>{notice.text}</Text>
    </View>
  );
};

const useStyles = makeStyles((colors) => ({
  notice: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm,
    paddingVertical: spacing.md,
    paddingHorizontal: spacing.md,
    borderRadius: radius.control,
    backgroundColor: colors.surface,
  },
  warn: { backgroundColor: colors.warningSoft },
  text: { ...typography.caption, flex: 1, color: colors.textSecondary },
  warnText: { color: colors.warning },
}));
