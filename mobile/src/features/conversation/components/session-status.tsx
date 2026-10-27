import { useTranslation } from 'react-i18next';
import { Text, View } from 'react-native';

import { formatElapsed, useElapsedSeconds } from '@/features/conversation/hooks/use-elapsed';
import { isConnecting } from '@/features/conversation/model/conversation-machine';
import { useConversationStore } from '@/features/conversation/model/conversation-store';
import { makeStyles, typography } from '@/theme';

/**
 * "● Listening 04:12 · Saving", "Paused 04:40" or "Connecting…", with the red dot only while
 * recording and "Saving" while the server writes a record.
 */
export const SessionStatusLine = ({ prefix }: { prefix?: string }) => {
  const { t } = useTranslation();
  const styles = useStyles();
  const phase = useConversationStore((state) => state.phase);
  const demo = useConversationStore((state) => state.demo);
  const saving = useConversationStore((state) => state.conversationId !== null);
  const seconds = useElapsedSeconds();

  if (isConnecting(phase)) return <Text style={styles.text}>{t('conversation.connecting')}</Text>;
  if (phase !== 'recording' && phase !== 'paused') return null;

  const recording = phase === 'recording';
  return (
    <View style={styles.row} accessibilityLiveRegion="polite">
      {recording && <View style={styles.dot} />}
      <Text style={styles.text} numberOfLines={1}>
        {prefix ??
          (demo
            ? t('conversation.demo')
            : t(recording ? 'conversation.listening' : 'conversation.paused'))}{' '}
        <Text style={styles.time}>{formatElapsed(seconds)}</Text>
        {saving && <Text style={styles.saving}> · {t('conversation.saving')}</Text>}
      </Text>
    </View>
  );
};

const useStyles = makeStyles((colors) => ({
  row: { flexDirection: 'row', alignItems: 'center', gap: 5 },
  dot: { width: 6, height: 6, borderRadius: 3, backgroundColor: colors.recording },
  text: { ...typography.small, color: colors.textTertiary },
  time: { fontVariant: ['tabular-nums'] },
  saving: { color: colors.accent },
}));
