import { useTranslation } from 'react-i18next';
import { Text, View } from 'react-native';

import { SegmentedControl } from '@/components/segmented-control';
import { OUTPUT_MODES } from '@/domain/output-modes';
import { useConversationPreferences } from '@/features/conversation/model/conversation-preferences';
import { makeStyles, spacing, typography } from '@/theme';

export const OutputModeControl = () => {
  const { t } = useTranslation();
  const styles = useStyles();
  const outputMode = useConversationPreferences((state) => state.outputMode);

  return (
    <View>
      <Text style={styles.title} accessibilityRole="header">
        {t('home.outputTitle')}
      </Text>
      <SegmentedControl
        options={OUTPUT_MODES}
        value={outputMode}
        label={(mode) => t(`outputModes.${mode}`)}
        onChange={useConversationPreferences.getState().setOutputMode}
      />
      <Text style={styles.hint}>{t(`outputModes.${outputMode}Hint`)}</Text>
    </View>
  );
};

const useStyles = makeStyles((colors) => ({
  title: {
    ...typography.small,
    color: colors.textTertiary,
    marginBottom: spacing.sm,
    marginLeft: spacing.xs,
  },
  hint: {
    ...typography.caption,
    color: colors.textTertiary,
    marginTop: spacing.sm,
    marginHorizontal: spacing.xs,
  },
}));
