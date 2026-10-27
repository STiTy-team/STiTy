import type { ReactNode } from 'react';
import { Text, View } from 'react-native';

import { makeStyles, spacing, typography } from '@/theme';

/** A tab's own title on the left, with optional actions on the right. */
export const TabHeader = ({ title, right }: { title: string; right?: ReactNode }) => {
  const styles = useStyles();
  return (
    <View style={styles.header}>
      <Text style={styles.title} accessibilityRole="header" numberOfLines={1}>
        {title}
      </Text>
      {right}
    </View>
  );
};

const useStyles = makeStyles((colors) => ({
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    minHeight: 56,
    paddingLeft: spacing.xl,
    paddingRight: spacing.sm,
  },
  title: { ...typography.wordmark, flex: 1, color: colors.text },
}));
