import { Text } from 'react-native';

import { makeStyles, typography } from '@/theme';

export const Logo = () => {
  const styles = useStyles();
  return (
    <Text style={styles.logo} accessibilityRole="header" accessibilityLabel="STiTy">
      STiTy
    </Text>
  );
};

const useStyles = makeStyles((colors) => ({
  logo: { ...typography.wordmark, color: colors.text },
}));
