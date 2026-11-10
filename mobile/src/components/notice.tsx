import { Ionicons } from '@expo/vector-icons';
import type { ComponentProps } from 'react';
import { type StyleProp, Text, View, type ViewStyle } from 'react-native';

import { makeStyles, radius, spacing, typography, useColors } from '@/theme';

type Props = {
  icon: ComponentProps<typeof Ionicons>['name'];
  text: string;
  warn?: boolean;
  style?: StyleProp<ViewStyle>;
};

export const Notice = ({ icon, text, warn = false, style }: Props) => {
  const styles = useStyles();
  const colors = useColors();
  return (
    <View
      style={[styles.notice, warn && styles.warn, style]}
      accessibilityLiveRegion="polite"
      accessibilityRole={warn ? 'alert' : 'text'}
    >
      <Ionicons name={icon} size={18} color={warn ? colors.warning : colors.textSecondary} />
      <Text style={[styles.text, warn && styles.warnText]}>{text}</Text>
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
