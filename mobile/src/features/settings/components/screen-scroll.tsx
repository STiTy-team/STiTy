import type { ReactNode } from 'react';
import { ScrollView } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';

import { makeStyles, spacing } from '@/theme';

export const ScreenScroll = ({ children }: { children: ReactNode }) => {
  const styles = useStyles();
  const { bottom } = useSafeAreaInsets();
  return (
    <ScrollView
      style={styles.scroll}
      contentContainerStyle={[styles.content, { paddingBottom: bottom + spacing.xxl }]}
      showsVerticalScrollIndicator={false}
    >
      {children}
    </ScrollView>
  );
};

const useStyles = makeStyles((colors) => ({
  scroll: { flex: 1, backgroundColor: colors.background },
  content: { paddingHorizontal: spacing.lg, paddingTop: spacing.sm, gap: spacing.xxl },
}));
