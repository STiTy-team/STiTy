import { useTranslation } from 'react-i18next';
import { ScrollView, Text, View } from 'react-native';

import { languageName } from '@/features/conversation/language-name';
import { useConversationPreferences } from '@/features/conversation/model/conversation-preferences';
import { useConversationStore } from '@/features/conversation/model/conversation-store';
import { usePreferencesStore } from '@/stores/preferences-store';
import { makeStyles, spacing, typography } from '@/theme';

/**
 * One sentence at a time, as large as the screen allows, for when the phone is held up between
 * two people in a loud place. The live partial takes over while someone is speaking.
 */
export const LargeFeed = () => {
  const { t } = useTranslation();
  const styles = useStyles();
  const last = useConversationStore((state) => state.entries[state.entries.length - 1]);
  const partial = useConversationStore((state) => state.partial);
  const talking = useConversationStore((state) => state.talking);
  const myLanguage = usePreferencesStore((state) => state.myLanguage);
  const holdToTalk = useConversationPreferences((state) => state.holdToTalk);

  if (partial) {
    return (
      <View style={styles.stage}>
        <Text style={styles.who}>{t('conversation.transcribing')}</Text>
        <Text style={[styles.headline, styles.live]} accessibilityLiveRegion="polite">
          {partial.text} …
        </Text>
      </View>
    );
  }

  if (!last) {
    return (
      <View style={styles.stage}>
        <Text style={styles.prompt}>
          {t(
            talking
              ? 'conversation.talking'
              : holdToTalk
                ? 'conversation.holdPrompt'
                : 'conversation.emptyTitle',
          )}
        </Text>
      </View>
    );
  }

  const mine = last.language === myLanguage;
  const headline = mine ? last.text : last.translation || last.text;
  const secondary = mine ? last.translation : last.translation ? last.text : '';

  return (
    <ScrollView contentContainerStyle={styles.stage} accessibilityLiveRegion="polite">
      <Text style={styles.who}>{mine ? t('conversation.me') : languageName(t, last.language)}</Text>
      <Text style={styles.headline}>{headline}</Text>
      {!!secondary && <Text style={styles.secondary}>{secondary}</Text>}
    </ScrollView>
  );
};

const useStyles = makeStyles((colors) => ({
  stage: {
    flexGrow: 1,
    justifyContent: 'center',
    paddingHorizontal: spacing.xxl,
    paddingVertical: spacing.xxl,
    gap: spacing.md,
  },
  who: { ...typography.label, color: colors.textSecondary },
  headline: { ...typography.display, color: colors.text },
  live: { color: colors.textSecondary },
  secondary: { ...typography.body, fontSize: 17, lineHeight: 25, color: colors.textSecondary },
  prompt: {
    ...typography.title,
    fontSize: 22,
    lineHeight: 30,
    color: colors.textTertiary,
    textAlign: 'center',
  },
}));
