import { useTranslation } from 'react-i18next';
import { Text, View } from 'react-native';

import { languageName } from '@/features/conversation/language-name';
import type { FeedEvent } from '@/features/conversation/model/transcript';
import { makeStyles, spacing, typography } from '@/theme';

const timeFormat = (locale: string) =>
  new Intl.DateTimeFormat(locale, { hour: 'numeric', minute: '2-digit' });

/** A quiet divider that marks when a setting changed mid-conversation. */
export const FeedEventRow = ({ event }: { event: FeedEvent }) => {
  const { t, i18n } = useTranslation();
  const styles = useStyles();
  const { change } = event;

  const message = (() => {
    switch (change.kind) {
      case 'outputMode':
        return t('conversation.events.outputMode', { mode: t(`outputModes.${change.mode}`) });
      case 'muted':
        return t(change.muted ? 'conversation.events.muted' : 'conversation.events.unmuted');
      case 'myLanguage':
        return t('conversation.events.myLanguage', { language: languageName(t, change.language) });
      case 'languageAdded':
        return t('conversation.events.languageAdded', {
          language: languageName(t, change.language),
        });
      case 'languageRemoved':
        return t('conversation.events.languageRemoved', {
          language: languageName(t, change.language),
        });
    }
  })();

  return (
    <View style={styles.row} accessibilityRole="text">
      <View style={styles.rule} />
      <Text style={styles.text}>
        {timeFormat(i18n.language).format(event.time)} {message}
      </Text>
      <View style={styles.rule} />
    </View>
  );
};

const useStyles = makeStyles((colors) => ({
  row: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.md,
    marginHorizontal: spacing.lg,
    marginBottom: spacing.xl,
  },
  rule: { flex: 1, height: 1, backgroundColor: colors.border },
  text: {
    ...typography.small,
    color: colors.textTertiary,
    flexShrink: 1,
    textAlign: 'center',
    fontVariant: ['tabular-nums'],
  },
}));
