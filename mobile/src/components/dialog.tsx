import {
  KeyboardAvoidingView,
  Modal,
  Platform,
  Pressable,
  Text,
  TextInput,
  View,
} from 'react-native';
import { create } from 'zustand';

import { makeStyles, radius, spacing, typography, useColors } from '@/theme';

type DialogOptions = {
  title: string;
  message?: string;
  confirmLabel: string;
  /** Leave out for a dialog that only informs. */
  cancelLabel?: string;
  destructive?: boolean;
};

type PromptOptions = Omit<DialogOptions, 'destructive'> & {
  value: string;
  placeholder?: string;
  maxLength?: number;
};

type OpenDialog = DialogOptions & {
  input?: Pick<PromptOptions, 'value' | 'placeholder' | 'maxLength'>;
  resolve: (result: { confirmed: boolean; value: string }) => void;
};

const useDialogStore = create<{ current: OpenDialog | null; text: string }>(() => ({
  current: null,
  text: '',
}));

const open = (dialog: Omit<OpenDialog, 'resolve'>) =>
  new Promise<{ confirmed: boolean; value: string }>((resolve) => {
    useDialogStore.getState().current?.resolve({ confirmed: false, value: '' });
    useDialogStore.setState({ current: { ...dialog, resolve }, text: dialog.input?.value ?? '' });
  });

/**
 * Shows a dialog and resolves true when confirmed. Drawn by the app rather than `Alert`, which
 * does nothing on the web and can't follow the app's theme.
 */
export const showDialog = async (options: DialogOptions): Promise<boolean> =>
  (await open(options)).confirmed;

/** Asks for one line of text; resolves null when cancelled or left empty. */
export const showPrompt = async ({
  value,
  placeholder,
  maxLength,
  ...options
}: PromptOptions): Promise<string | null> => {
  const result = await open({ ...options, input: { value, placeholder, maxLength } });
  const text = result.value.trim();
  return result.confirmed && text ? text : null;
};

const close = (confirmed: boolean, value = '') => {
  const { current } = useDialogStore.getState();
  useDialogStore.setState({ current: null });
  current?.resolve({ confirmed, value });
};

/** Rendered once at the root of the app. */
export const DialogHost = () => {
  const styles = useStyles();
  const colors = useColors();
  const dialog = useDialogStore((state) => state.current);
  const text = useDialogStore((state) => state.text);
  const setText = (next: string) => useDialogStore.setState({ text: next });

  return (
    <Modal
      visible={dialog !== null}
      transparent
      animationType="fade"
      statusBarTranslucent
      navigationBarTranslucent
      onRequestClose={() => close(false)}
    >
      <KeyboardAvoidingView
        style={styles.scrim}
        behavior={Platform.OS === 'ios' ? 'padding' : undefined}
      >
        {dialog && (
          <View style={styles.card} accessibilityRole="alert" accessibilityViewIsModal>
            <Text style={styles.title}>{dialog.title}</Text>
            {dialog.message && <Text style={styles.message}>{dialog.message}</Text>}
            {dialog.input && (
              <TextInput
                value={text}
                onChangeText={setText}
                placeholder={dialog.input.placeholder}
                placeholderTextColor={colors.textTertiary}
                maxLength={dialog.input.maxLength}
                autoFocus
                selectTextOnFocus
                returnKeyType="done"
                onSubmitEditing={() => close(true, text)}
                accessibilityLabel={dialog.title}
                style={styles.input}
              />
            )}
            <View style={styles.actions}>
              {dialog.cancelLabel && (
                <Pressable
                  onPress={() => close(false)}
                  accessibilityRole="button"
                  style={({ pressed }) => [styles.button, styles.cancel, pressed && styles.pressed]}
                >
                  <Text style={styles.cancelText}>{dialog.cancelLabel}</Text>
                </Pressable>
              )}
              <Pressable
                onPress={() => close(true, text)}
                accessibilityRole="button"
                style={({ pressed }) => [
                  styles.button,
                  dialog.destructive ? styles.destructive : styles.confirm,
                  pressed && styles.pressed,
                ]}
              >
                <Text style={styles.confirmText}>{dialog.confirmLabel}</Text>
              </Pressable>
            </View>
          </View>
        )}
      </KeyboardAvoidingView>
    </Modal>
  );
};

const useStyles = makeStyles((colors) => ({
  scrim: {
    flex: 1,
    backgroundColor: colors.scrim,
    alignItems: 'center',
    justifyContent: 'center',
    padding: spacing.xxl,
  },
  card: {
    width: '100%',
    maxWidth: 340,
    backgroundColor: colors.background,
    borderRadius: radius.sheet,
    padding: spacing.xxl,
    gap: spacing.sm,
  },
  title: { ...typography.title, color: colors.text },
  message: { ...typography.body, color: colors.textSecondary },
  input: {
    ...typography.body,
    color: colors.text,
    minHeight: 48,
    marginTop: spacing.sm,
    paddingHorizontal: spacing.md,
    borderRadius: radius.control,
    backgroundColor: colors.surface,
  },
  actions: { flexDirection: 'row', gap: spacing.sm, marginTop: spacing.lg },
  button: {
    flex: 1,
    minHeight: 48,
    borderRadius: radius.control,
    alignItems: 'center',
    justifyContent: 'center',
  },
  cancel: { backgroundColor: colors.surface },
  confirm: { backgroundColor: colors.accent },
  destructive: { backgroundColor: colors.danger },
  pressed: { opacity: 0.85 },
  cancelText: { ...typography.button, color: colors.text },
  confirmText: { ...typography.button, color: colors.onAccent },
}));
