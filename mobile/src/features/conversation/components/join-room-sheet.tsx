import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Text, View } from 'react-native';

import { BottomSheet } from '@/components/bottom-sheet';
import { PrimaryButton } from '@/components/button';
import { TextField } from '@/components/text-field';
import { isRoomCode, normalizeRoomCode, ROOM_CODE_LENGTH } from '@/domain/rooms';
import { useJoinRoom } from '@/features/conversation/hooks/use-join-room';
import { makeStyles, spacing, typography } from '@/theme';

type Props = {
  visible: boolean;
  onClose: () => void;
};

export const JoinRoomSheet = ({ visible, onClose }: Props) => {
  const { t } = useTranslation();
  const styles = useStyles();
  const [code, setCode] = useState('');
  const { checking, error, join, clearError } = useJoinRoom();

  const close = () => {
    if (checking) return;
    setCode('');
    clearError();
    onClose();
  };
  const submit = async () => {
    if (!isRoomCode(code) || checking) return;
    if (await join(code)) {
      setCode('');
      onClose();
    }
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
            clearError();
          }}
          onSubmitEditing={() => void submit()}
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
          onPress={() => void submit()}
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
