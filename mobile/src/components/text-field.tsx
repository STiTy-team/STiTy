import { Ionicons } from '@expo/vector-icons';
import type { ComponentProps } from 'react';
import { TextInput, View, type TextInputProps } from 'react-native';

import { makeStyles, radius, spacing, typography, useColors } from '@/theme';

type Props = Omit<TextInputProps, 'style' | 'placeholderTextColor'> & {
  icon?: ComponentProps<typeof Ionicons>['name'];
};

export const TextField = ({ icon, ...input }: Props) => {
  const styles = useStyles();
  const colors = useColors();
  return (
    <View style={styles.field}>
      {icon && <Ionicons name={icon} size={18} color={colors.textTertiary} />}
      <TextInput
        {...input}
        placeholderTextColor={colors.textTertiary}
        style={styles.input}
        accessibilityLabel={input.accessibilityLabel ?? input.placeholder}
      />
    </View>
  );
};

const useStyles = makeStyles((colors) => ({
  field: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm,
    minHeight: 48,
    paddingHorizontal: spacing.md,
    borderRadius: radius.control,
    backgroundColor: colors.surface,
  },
  input: { ...typography.body, flex: 1, color: colors.text, paddingVertical: spacing.sm },
}));
