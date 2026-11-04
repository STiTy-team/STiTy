import { router } from 'expo-router';
import { useTranslation } from 'react-i18next';
import { Switch } from 'react-native';

import { ListRow, ListSection } from '@/components/list';
import { canStart } from '@/features/conversation/model/conversation-machine';
import { useConversationPreferences } from '@/features/conversation/model/conversation-preferences';
import { useConversationStore } from '@/features/conversation/model/conversation-store';
import { useFeatures } from '@/stores/capabilities-store';
import { selectSignedIn, useSessionStore } from '@/stores/session-store';
import { useColors } from '@/theme';

/**
 * Whether the server saves the next conversation. It is decided before Start, because the
 * server opens the record when the session starts.
 */
export const SaveToggle = () => {
  const { t } = useTranslation();
  const colors = useColors();
  const { records, accounts } = useFeatures();
  const signedIn = useSessionStore(selectSignedIn);
  const saveRequested = useConversationPreferences((state) => state.saveRequested);
  const phase = useConversationStore((state) => state.phase);

  if (!records || !accounts) return null;

  if (!signedIn) {
    return (
      <ListSection footer={t('home.saveGuestHint')}>
        <ListRow
          title={t('home.signInToSave')}
          tone="action"
          chevron
          onPress={() => router.push('/sign-in')}
        />
      </ListSection>
    );
  }

  const locked = !canStart(phase);
  return (
    <ListSection footer={t(locked ? 'home.saveLocked' : 'home.saveHint')}>
      <ListRow title={t('home.save')}>
        <Switch
          value={saveRequested}
          disabled={locked}
          onValueChange={useConversationPreferences.getState().setSaveRequested}
          trackColor={{ true: colors.accent, false: colors.surfaceStrong }}
          thumbColor="#FFFFFF"
          accessibilityLabel={t('home.save')}
        />
      </ListRow>
    </ListSection>
  );
};
