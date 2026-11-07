import { useTranslation } from 'react-i18next';
import { View } from 'react-native';

import { showDialog } from '@/components/dialog';
import { IconButton } from '@/components/icon-button';
import { ListRow, ListSection } from '@/components/list';
import { ToggleSwitch } from '@/components/toggle-switch';
import type { RoomParticipant } from '@/domain/rooms';
import { getConversationController } from '@/features/conversation/conversation-runtime';
import {
  selectInRoom,
  selectIsHost,
  useConversationStore,
} from '@/features/conversation/model/conversation-store';
import { makeStyles, spacing } from '@/theme';

/** Who is in the room, and for the host, who may speak and who stays. */
export const RoomSection = () => {
  const { t } = useTranslation();
  const styles = useStyles();
  const controller = getConversationController();
  const inRoom = useConversationStore(selectInRoom);
  const roster = useConversationStore((state) => state.roster);
  const me = useConversationStore((state) => state.participantId);
  const isHost = useConversationStore(selectIsHost);

  if (!inRoom || !roster) return null;

  const nameOf = ({ number }: RoomParticipant) => t('room.participant', { number });
  const remove = async (participant: RoomParticipant) => {
    const name = nameOf(participant);
    const confirmed = await showDialog({
      title: t('room.removeTitle', { name }),
      message: t('room.removeMessage'),
      confirmLabel: t('room.removeAction'),
      cancelLabel: t('common.cancel'),
    });
    if (confirmed) controller.removeParticipant(participant.id);
  };
  const details = (participant: RoomParticipant) =>
    [
      participant.id === roster.hostId && t('room.host'),
      participant.id === me && t('room.you'),
      t(participant.canSpeak ? 'room.micOn' : 'room.micOff'),
    ]
      .filter(Boolean)
      .join(' · ');

  return (
    <ListSection title={t('room.title')} footer={isHost ? t('room.hostHint') : undefined}>
      {isHost && (
        <ListRow
          title={t('room.allMicsOff')}
          tone="action"
          onPress={() => controller.setAllMics(false)}
        />
      )}
      {isHost && (
        <ListRow
          title={t('room.allMicsOn')}
          tone="action"
          onPress={() => controller.setAllMics(true)}
        />
      )}
      {roster.participants.map((participant) => (
        <ListRow key={participant.id} title={nameOf(participant)} subtitle={details(participant)}>
          {isHost && (
            <View style={styles.controls}>
              {participant.id !== me && (
                <IconButton
                  icon="close"
                  label={t('room.remove', { name: nameOf(participant) })}
                  onPress={() => void remove(participant)}
                />
              )}
              <ToggleSwitch
                value={participant.canSpeak}
                onValueChange={(on) => controller.setMic(participant.id, on)}
                accessibilityLabel={`${nameOf(participant)}: ${t('room.micOn')}`}
              />
            </View>
          )}
        </ListRow>
      ))}
    </ListSection>
  );
};

const useStyles = makeStyles(() => ({
  controls: { flexDirection: 'row', alignItems: 'center', gap: spacing.xs },
}));
