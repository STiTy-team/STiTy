import type { ReactNode } from 'react';
import { Text, View } from 'react-native';

import { makeStyles, spacing, typography } from '@/theme';

const SIDE_WIDTH = 52;

type Props = {
  title: string;
  subtitle?: ReactNode;
  left?: ReactNode;
  right?: ReactNode;
  /** Room for several actions on the right; the title then sits slightly left of center. */
  rightWidth?: number;
};

/** Centered title with one action slot on each side; empty slots keep the title centered. */
export const ScreenHeader = ({ title, subtitle, left, right, rightWidth = SIDE_WIDTH }: Props) => {
  const styles = useStyles();
  return (
    <View style={styles.header}>
      <View style={[styles.side, { width: SIDE_WIDTH }]}>{left}</View>
      <View style={styles.center}>
        <Text style={styles.title} numberOfLines={1} accessibilityRole="header">
          {title}
        </Text>
        {subtitle}
      </View>
      <View style={[styles.side, styles.right, { width: rightWidth }]}>{right}</View>
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
  side: { alignItems: 'flex-start' },
  right: { alignItems: 'flex-end' },
  center: { flex: 1, alignItems: 'center', minWidth: 0 },
  title: { ...typography.title, color: colors.text },
}));
