import { Ionicons } from '@expo/vector-icons';
import { useTranslation } from 'react-i18next';
import { Text, View } from 'react-native';

import type { ConversationErrorCode } from '@/features/conversation/model/conversation-error';
import { makeStyles, radius, spacing, typography, useColors } from '@/theme';

export const ErrorBanner = ({ code }: { code: ConversationErrorCode }) => {
  const { t } = useTranslation();
  const styles = useStyles();
  const colors = useColors();
  return (
    <View style={styles.banner} accessibilityRole="alert">
      <Ionicons name="alert-circle" size={18} color={colors.danger} />
      <Text style={styles.text}>{t(`errors.${code}`)}</Text>
    </View>
  );
};

const useStyles = makeStyles((colors) => ({
  banner: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    gap: spacing.sm,
    marginHorizontal: spacing.lg,
    marginBottom: spacing.sm,
    paddingVertical: spacing.md,
    paddingHorizontal: spacing.md,
    borderRadius: radius.control,
    backgroundColor: colors.dangerSoft,
  },
  text: { ...typography.caption, flex: 1, color: colors.text },
}));
