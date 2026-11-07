import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Text, View } from 'react-native';

import { BottomSheet } from '@/components/bottom-sheet';
import { PrimaryButton } from '@/components/button';
import { TextField } from '@/components/text-field';
import { isRoomCode, normalizeRoomCode, ROOM_CODE_LENGTH } from '@/domain/rooms';
import { joinConversation } from '@/features/conversation/conversation-runtime';
import type { ConversationErrorCode } from '@/features/conversation/model/conversation-error';
import { makeStyles, spacing, typography } from '@/theme';

type Props = {
  visible: boolean;
  onClose: () => void;
};

export const JoinRoomSheet = ({ visible, onClose }: Props) => {
  const { t } = useTranslation();
  const styles = useStyles();
  const [code, setCode] = useState('');
  const [checking, setChecking] = useState(false);
  const [error, setError] = useState<ConversationErrorCode | null>(null);

  const close = () => {
    if (checking) return;
    setCode('');
    setError(null);
    onClose();
  };
  const join = async () => {
    if (!isRoomCode(code) || checking) return;
    setChecking(true);
    setError(null);
    const failure = await joinConversation(code);
    setChecking(false);
    if (failure) {
      setError(failure);
      return;
    }
    setCode('');
    onClose();
  };

  return (
    <BottomSheet visible={visible} title={t('home.joinRoom')} onClose={close}>
      <View style={styles.body}>
        <Text style={styles.hint}>{t('home.joinRoomHint')}</Text>
        <TextField
          icon="people-outline"
          value={code}
          onChangeText={(text) => {
            setCode(normalizeRoomCode(text));
            setError(null);
          }}
          onSubmitEditing={() => void join()}
          placeholder={t('home.roomCode')}
          autoCapitalize="characters"
          autoCorrect={false}
          autoFocus
          maxLength={ROOM_CODE_LENGTH}
        />
        {error && (
          <Text style={styles.error} accessibilityLiveRegion="polite">
            {t(`errors.${error}`)}
          </Text>
        )}
        <PrimaryButton
          label={t('home.join')}
          disabled={!isRoomCode(code)}
          loading={checking}
          onPress={() => void join()}
        />
      </View>
    </BottomSheet>
  );
};

const useStyles = makeStyles((colors) => ({
  body: { gap: spacing.md, paddingHorizontal: spacing.lg },
  hint: { ...typography.small, color: colors.textSecondary },
  error: { ...typography.small, color: colors.danger },
}));
