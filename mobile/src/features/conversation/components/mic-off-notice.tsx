import { useTranslation } from 'react-i18next';
import { StyleSheet } from 'react-native';

import { Notice } from '@/components/notice';
import { spacing } from '@/theme';

export const MicOffNotice = () => {
  const { t } = useTranslation();
  return <Notice icon="mic-off" text={t('conversation.micOffNotice')} style={styles.notice} />;
};

const styles = StyleSheet.create({
  notice: { marginHorizontal: spacing.lg, marginBottom: spacing.sm },
});
