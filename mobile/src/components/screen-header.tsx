import type { ReactNode } from 'react';
import { Text, View } from 'react-native';

import { makeStyles, spacing, typography } from '@/theme';

type Props = {
  title: string;
  subtitle?: ReactNode;
  left?: ReactNode;
  right?: ReactNode;
  /** Room for more than one action per side; both sides match so the title stays centered. */
  sideWidth?: number;
};

/** Centered title with one action slot on each side; empty slots keep the title centered. */
export const ScreenHeader = ({ title, subtitle, left, right, sideWidth = 52 }: Props) => {
  const styles = useStyles();
  return (
    <View style={styles.header}>
      <View style={[styles.side, { width: sideWidth }]}>{left}</View>
      <View style={styles.center}>
        <Text style={styles.title} numberOfLines={1} accessibilityRole="header">
          {title}
        </Text>
        {subtitle}
      </View>
      <View style={[styles.side, styles.right, { width: sideWidth }]}>{right}</View>
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
