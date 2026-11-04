import type { ReactNode } from 'react';
import { Text, View } from 'react-native';

import { makeStyles, spacing, typography } from '@/theme';

type Props = {
  title: string;
  subtitle?: ReactNode;
  left?: ReactNode;
  right?: ReactNode;
};

/** Centered title with one action slot on each side; empty slots keep the title centered. */
export const ScreenHeader = ({ title, subtitle, left, right }: Props) => {
  const styles = useStyles();
  return (
    <View style={styles.header}>
      <View style={styles.side}>{left}</View>
      <View style={styles.center}>
        <Text style={styles.title} numberOfLines={1} accessibilityRole="header">
          {title}
        </Text>
        {subtitle}
      </View>
      <View style={[styles.side, styles.right]}>{right}</View>
    </View>
  );
};

const useStyles = makeStyles((colors) => ({
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    minHeight: 56,
    paddingHorizontal: spacing.sm,
  },
  side: { width: 52, alignItems: 'flex-start' },
  right: { alignItems: 'flex-end' },
  center: { flex: 1, alignItems: 'center', minWidth: 0 },
  title: { ...typography.title, color: colors.text },
}));
