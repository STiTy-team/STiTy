import { Ionicons } from '@expo/vector-icons';
import type { ComponentProps } from 'react';
import { useTranslation } from 'react-i18next';
import { ActivityIndicator, Pressable, Text, View } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';

import { OUTPUT_MODES, type OutputMode } from '@/domain/output-modes';
import { getConversationController } from '@/features/conversation/conversation-runtime';
import { isConnecting } from '@/features/conversation/model/conversation-machine';
import { useConversationPreferences } from '@/features/conversation/model/conversation-preferences';
import { useConversationStore } from '@/features/conversation/model/conversation-store';
import { haptics } from '@/services/haptics';
import { makeStyles, spacing, typography, useColors, type ColorScheme } from '@/theme';

const OUTPUT_ICONS: Record<OutputMode, ComponentProps<typeof Ionicons>['name']> = {
  speaker: 'phone-portrait-outline',
  earphone: 'headset-outline',
  both: 'swap-horizontal-outline',
};

type Props = {
  /** Push-to-talk replaces the pause button: audio is sent only while it is held. */
  holdToTalk?: boolean;
  /** Present mode: a floating bar on a fixed scheme, with Exit in place of the output mode. */
  present?: { scheme: ColorScheme; onExit: () => void };
  /** Any press, so Present can keep its controls on screen. */
  onInteract?: () => void;
};

/**
 * Sound and output on either side of the microphone, for the live screen and Present alike.
 * The microphone is red only while it records; paused, it becomes the accent "resume" button,
 * and after a failed start it retries.
 */
export const SessionControls = ({ holdToTalk = false, present, onInteract }: Props) => {
  const { t } = useTranslation();
  const scheme = present?.scheme;
  const styles = useStyles(scheme);
  const colors = useColors(scheme);
  const { bottom } = useSafeAreaInsets();
  const controller = getConversationController();
  const phase = useConversationStore((state) => state.phase);
  const talking = useConversationStore((state) => state.talking);
  const muted = useConversationPreferences((state) => state.muted);
  const outputMode = useConversationPreferences((state) => state.outputMode);
  const { setMuted, setOutputMode } = useConversationPreferences.getState();

  const press = (action: () => void) => () => {
    onInteract?.();
    action();
  };
  const nextOutput = OUTPUT_MODES[(OUTPUT_MODES.indexOf(outputMode) + 1) % OUTPUT_MODES.length]!;
  const busy = isConnecting(phase) || phase === 'ending';
  const recording = phase === 'recording';

  const mic = busy
    ? { label: t('conversation.connecting'), onPress: () => {}, style: styles.micIdle }
    : recording
      ? {
          label: t('conversation.pause'),
          onPress: () => controller.pause(),
          style: styles.micRecording,
        }
      : phase === 'paused'
        ? {
            label: t('conversation.resume'),
            onPress: () => void controller.resume(),
            style: styles.micIdle,
          }
        : {
            label: t('conversation.retry'),
            onPress: () => void controller.start(),
            style: styles.micIdle,
          };

  return (
    <View
      style={
        present
          ? styles.floating
          : [styles.bar, { paddingBottom: Math.max(bottom, spacing.lg) + spacing.sm }]
      }
    >
      <SideButton
        scheme={scheme}
        icon={muted ? 'volume-mute-outline' : 'volume-high-outline'}
        caption={muted ? t('conversation.muted') : t('conversation.sound')}
        accessibilityRole="switch"
        checked={!muted}
        onPress={press(() => setMuted(!muted))}
      />

      {holdToTalk && recording ? (
        <View style={styles.holdColumn}>
          <View style={[styles.halo, talking && styles.haloOn]}>
            <Pressable
              onPressIn={() => {
                haptics.press();
                controller.setTalking(true);
              }}
              onPressOut={() => controller.setTalking(false)}
              accessibilityRole="button"
              accessibilityLabel={t('conversation.holdToTalk')}
              accessibilityHint={t('conversation.holdHint')}
              accessibilityState={{ selected: talking }}
              style={[styles.hold, talking ? styles.micRecording : styles.micIdle]}
            >
              <Ionicons name="mic" size={36} color={colors.onAccent} />
            </Pressable>
          </View>
          <Text style={styles.holdCaption}>
            {t(talking ? 'conversation.talking' : 'conversation.holdToTalk')}
          </Text>
        </View>
      ) : (
        <View style={[styles.halo, recording && styles.haloOn]}>
          <Pressable
            onPress={press(mic.onPress)}
            disabled={busy}
            accessibilityRole="button"
            accessibilityLabel={mic.label}
            accessibilityState={{ busy }}
            style={({ pressed }) => [styles.mic, mic.style, pressed && styles.pressed]}
          >
            {busy ? (
              <ActivityIndicator color={colors.onAccent} />
            ) : recording ? (
              <View style={styles.stop} />
            ) : (
              <Ionicons
                name={phase === 'paused' ? 'mic' : 'refresh'}
                size={28}
                color={colors.onAccent}
              />
            )}
          </Pressable>
        </View>
      )}

      {present ? (
        <SideButton
          scheme={scheme}
          icon="contract-outline"
          caption={t('present.exit')}
          onPress={present.onExit}
        />
      ) : (
        <SideButton
          icon={OUTPUT_ICONS[outputMode]}
          caption={t(`outputModes.${outputMode}`)}
          accessibilityLabel={`${t('home.outputTitle')}: ${t(`outputModes.${outputMode}`)}`}
          accessibilityHint={t(`outputModes.${nextOutput}`)}
          onPress={press(() => setOutputMode(nextOutput))}
        />
      )}
    </View>
  );
};

type SideButtonProps = {
  scheme?: ColorScheme;
  icon: ComponentProps<typeof Ionicons>['name'];
  caption: string;
  onPress: () => void;
  accessibilityRole?: 'button' | 'switch';
  accessibilityLabel?: string;
  accessibilityHint?: string;
  checked?: boolean;
};

const SideButton = ({
  scheme,
  icon,
  caption,
  onPress,
  accessibilityRole = 'button',
  accessibilityLabel,
  accessibilityHint,
  checked,
}: SideButtonProps) => {
  const styles = useStyles(scheme);
  const colors = useColors(scheme);
  return (
    <Pressable
      onPress={onPress}
      accessibilityRole={accessibilityRole}
      accessibilityLabel={accessibilityLabel ?? caption}
      accessibilityHint={accessibilityHint}
      accessibilityState={checked === undefined ? undefined : { checked }}
      style={styles.side}
    >
      {({ pressed }) => (
        <>
          <View style={[styles.sideCircle, pressed && styles.sidePressed]}>
            <Ionicons name={icon} size={22} color={colors.textSecondary} />
          </View>
          <Text style={styles.caption} numberOfLines={1}>
            {caption}
          </Text>
        </>
      )}
    </Pressable>
  );
};

const MIC = 72;
const HOLD = 96;
const HALO = 6;

const useStyles = makeStyles((colors) => ({
  bar: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    justifyContent: 'space-between',
    paddingTop: spacing.md,
    paddingHorizontal: spacing.xxl,
    borderTopWidth: 1,
    borderTopColor: colors.border,
    backgroundColor: colors.background,
  },
  floating: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    gap: spacing.lg,
    paddingTop: spacing.md,
    paddingBottom: spacing.sm,
    paddingHorizontal: spacing.lg,
    borderRadius: 28,
    backgroundColor: colors.surfaceStrong,
  },
  side: { width: 76, alignItems: 'center', gap: 6, paddingTop: 6 },
  sideCircle: {
    width: 52,
    height: 52,
    borderRadius: 26,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: colors.surface,
  },
  sidePressed: { backgroundColor: colors.surfaceStrong },
  caption: { ...typography.small, color: colors.textTertiary },
  halo: { padding: HALO, borderRadius: (MIC + HALO * 2) / 2 },
  haloOn: { backgroundColor: colors.recordingHalo },
  mic: {
    width: MIC,
    height: MIC,
    borderRadius: MIC / 2,
    alignItems: 'center',
    justifyContent: 'center',
  },
  micRecording: { backgroundColor: colors.recording },
  micIdle: { backgroundColor: colors.accent },
  pressed: { opacity: 0.85 },
  stop: { width: 22, height: 22, borderRadius: 6, backgroundColor: colors.onRecording },
  holdColumn: { alignItems: 'center', gap: spacing.xs, marginTop: -spacing.lg },
  hold: {
    width: HOLD,
    height: HOLD,
    borderRadius: HOLD / 2,
    alignItems: 'center',
    justifyContent: 'center',
  },
  holdCaption: { ...typography.label, color: colors.textSecondary },
}));
