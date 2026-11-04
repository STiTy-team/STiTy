import { Ionicons } from '@expo/vector-icons';
import type { ComponentProps } from 'react';
import { ActivityIndicator, Pressable, Text, View } from 'react-native';

import { makeStyles, radius, spacing, typography, useColors } from '@/theme';

type Props = {
  icon?: ComponentProps<typeof Ionicons>['name'];
  title: string;
  message?: string;
  action?: { label: string; onPress: () => void };
};

/** The centred block a screen shows instead of content: empty, failed or not available. */
export const StateView = ({ icon, title, message, action }: Props) => {
  const styles = useStyles();
  const colors = useColors();
  return (
    <View style={styles.block} accessibilityLiveRegion="polite">
      {icon && <Ionicons name={icon} size={28} color={colors.textTertiary} />}
      <Text style={styles.title} accessibilityRole="header">
        {title}
      </Text>
      {message && <Text style={styles.message}>{message}</Text>}
      {action && (
        <Pressable
          onPress={action.onPress}
          accessibilityRole="button"
          style={({ pressed }) => [styles.action, pressed && styles.pressed]}
        >
          <Text style={styles.actionText}>{action.label}</Text>
        </Pressable>
      )}
    </View>
  );
};

export const LoadingView = ({ label }: { label: string }) => {
  const styles = useStyles();
  const colors = useColors();
  return (
    <View style={styles.block} accessibilityLabel={label} accessibilityRole="progressbar">
      <ActivityIndicator color={colors.textTertiary} />
    </View>
  );
};

const useStyles = makeStyles((colors) => ({
  block: {
    flexGrow: 1,
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: spacing.xxxl,
    paddingVertical: spacing.xxxl,
    gap: spacing.sm,
  },
  title: { ...typography.title, color: colors.text, textAlign: 'center', marginTop: spacing.xs },
  message: { ...typography.caption, color: colors.textTertiary, textAlign: 'center' },
  action: {
    marginTop: spacing.md,
    minHeight: 44,
    paddingHorizontal: spacing.xl,
    borderRadius: radius.control,
    backgroundColor: colors.accentSoft,
    alignItems: 'center',
    justifyContent: 'center',
  },
  actionText: { ...typography.label, color: colors.accent },
  pressed: { opacity: 0.8 },
}));
