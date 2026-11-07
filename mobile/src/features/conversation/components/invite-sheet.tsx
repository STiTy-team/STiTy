import * as Clipboard from 'expo-clipboard';
import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Platform, Share, Text, View } from 'react-native';

import { BottomSheet } from '@/components/bottom-sheet';
import { PrimaryButton } from '@/components/button';
import { QrCode } from '@/components/qr-code';
import { useConversationStore } from '@/features/conversation/model/conversation-store';
import { roomLink } from '@/features/conversation/room-link';
import { makeStyles, spacing, typography } from '@/theme';

const QR_SIZE = 200;

const canShare = Platform.OS !== 'web' || typeof navigator.share === 'function';

type Props = {
  visible: boolean;
  onClose: () => void;
};

export const InviteSheet = ({ visible, onClose }: Props) => {
  const { t } = useTranslation();
  const styles = useStyles();
  const code = useConversationStore((state) => state.room?.code);
  const [copied, setCopied] = useState(false);
  const link = code ? roomLink(code) : '';

  const copy = async () => {
    await Clipboard.setStringAsync(link);
    setCopied(true);
  };
  const close = () => {
    setCopied(false);
    onClose();
  };

  return (
    <BottomSheet visible={visible} title={t('conversation.invite')} onClose={close}>
      <View style={styles.body}>
        {link !== '' && (
          <View style={styles.qr}>
            <QrCode value={link} size={QR_SIZE} label={t('conversation.inviteQr')} />
          </View>
        )}
        <Text style={styles.code} selectable accessibilityLabel={t('home.roomCode')}>
          {code}
        </Text>
        <Text style={styles.link} selectable numberOfLines={2}>
          {link}
        </Text>
        <Text style={styles.hint}>{t('conversation.inviteHint')}</Text>
        <View style={styles.actions}>
          {canShare && (
            <View style={styles.action}>
              <PrimaryButton
                label={t('conversation.share')}
                icon="share-outline"
                onPress={() => void Share.share({ message: link })}
              />
            </View>
          )}
          <View style={styles.action}>
            <PrimaryButton
              label={t(copied ? 'conversation.linkCopied' : 'conversation.copyLink')}
              icon={copied ? 'checkmark' : 'link-outline'}
              variant="secondary"
              onPress={() => void copy()}
            />
          </View>
        </View>
        <PrimaryButton label={t('common.close')} variant="secondary" onPress={close} />
      </View>
    </BottomSheet>
  );
};

const useStyles = makeStyles((colors) => ({
  body: { gap: spacing.md, paddingHorizontal: spacing.lg },
  qr: { alignItems: 'center' },
  code: {
    ...typography.title,
    color: colors.text,
    letterSpacing: 4,
    textAlign: 'center',
    fontVariant: ['tabular-nums'],
  },
  link: { ...typography.small, color: colors.textTertiary, textAlign: 'center' },
  hint: { ...typography.small, color: colors.textSecondary },
  actions: { flexDirection: 'row', gap: spacing.sm },
  action: { flex: 1 },
}));
