import { Text } from 'react-native';

import { makeStyles, typography, type ColorScheme } from '@/theme';

export const Logo = ({ scheme }: { scheme?: ColorScheme }) => {
  const styles = useStyles(scheme);
  return (
    <Text style={styles.logo} accessibilityRole="header" accessibilityLabel="STiTy">
      STiTy
    </Text>
  );
};

const useStyles = makeStyles((colors) => ({
  logo: { ...typography.wordmark, color: colors.text },
}));
