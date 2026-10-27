import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { KeyboardAvoidingView, Platform, ScrollView, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { apiErrorKind, getApi, type ApiErrorKind } from '@/api';
import { PrimaryButton } from '@/components/button';
import { IconButton } from '@/components/icon-button';
import { goBack } from '@/components/go-back';
import { ScreenHeader } from '@/components/screen-header';
import { TextField } from '@/components/text-field';
import { requestEmailCode, signIn, verifyEmailCode } from '@/features/account/account-runtime';
import { makeStyles, spacing, typography } from '@/theme';

const CODE_LENGTH = 6;

type Step = { kind: 'choose' } | { kind: 'email' } | { kind: 'code'; email: string };

/**
 * Signing in only unlocks saving; translation never asks for it. Apple is offered wherever
 * Google is, as the App Store requires.
 */
export const SignInScreen = () => {
  const { t } = useTranslation();
  const styles = useStyles();
  const { methods } = getApi().auth;
  const [step, setStep] = useState<Step>({ kind: 'choose' });
  const [email, setEmail] = useState('');
  const [code, setCode] = useState('');
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<ApiErrorKind | null>(null);

  const run = async (key: string, action: () => Promise<unknown>, onDone?: () => void) => {
    setBusy(key);
    setError(null);
    try {
      await action();
      if (onDone) onDone();
      else goBack();
    } catch (failure) {
      setError(apiErrorKind(failure));
    } finally {
      setBusy(null);
    }
  };

  const errorText = error && (
    <Text style={styles.error} accessibilityRole="alert">
      {t(`signIn.errors.${error}`)}
    </Text>
  );

  return (
    <SafeAreaView style={styles.screen} edges={['top', 'left', 'right', 'bottom']}>
      <ScreenHeader
        title=""
        left={
          step.kind === 'choose' ? (
            <IconButton icon="close" label={t('common.close')} onPress={goBack} />
          ) : (
            <IconButton
              icon="chevron-back"
              label={t('common.back')}
              onPress={() => {
                setError(null);
                setStep(step.kind === 'code' ? { kind: 'email' } : { kind: 'choose' });
              }}
            />
          )
        }
      />
      <KeyboardAvoidingView
        style={styles.fill}
        behavior={Platform.OS === 'ios' ? 'padding' : undefined}
      >
        <ScrollView contentContainerStyle={styles.content} keyboardShouldPersistTaps="handled">
          {step.kind === 'choose' && (
            <>
              <View style={styles.intro}>
                <Text style={styles.title} accessibilityRole="header">
                  {t('signIn.title')}
                </Text>
                <Text style={styles.subtitle}>{t('signIn.subtitle')}</Text>
              </View>
              <View style={styles.buttons}>
                {methods.includes('apple') && (
                  <PrimaryButton
                    variant="contrast"
                    label={t('signIn.apple')}
                    icon="logo-apple"
                    loading={busy === 'apple'}
                    disabled={busy !== null}
                    onPress={() => void run('apple', () => signIn('apple'))}
                  />
                )}
                {methods.includes('google') && (
                  <PrimaryButton
                    variant="secondary"
                    label={t('signIn.google')}
                    icon="logo-google"
                    loading={busy === 'google'}
                    disabled={busy !== null}
                    onPress={() => void run('google', () => signIn('google'))}
                  />
                )}
                {methods.includes('email') && (
                  <PrimaryButton
                    variant={methods.length > 1 ? 'secondary' : 'primary'}
                    label={t('signIn.email')}
                    icon="mail-outline"
                    disabled={busy !== null}
                    onPress={() => setStep({ kind: 'email' })}
                  />
                )}
              </View>
              {errorText}
              <Text style={styles.note}>{t('signIn.guestNote')}</Text>
            </>
          )}

          {step.kind === 'email' && (
            <>
              <View style={styles.intro}>
                <Text style={styles.title} accessibilityRole="header">
                  {t('signIn.emailTitle')}
                </Text>
              </View>
              <TextField
                icon="mail-outline"
                value={email}
                onChangeText={setEmail}
                placeholder={t('signIn.emailPlaceholder')}
                accessibilityLabel={t('signIn.emailLabel')}
                keyboardType="email-address"
                autoCapitalize="none"
                autoComplete="email"
                textContentType="emailAddress"
                autoFocus
                returnKeyType="send"
              />
              {errorText}
              <PrimaryButton
                label={t('signIn.sendCode')}
                loading={busy === 'send'}
                disabled={!email.includes('@')}
                onPress={() =>
                  void run(
                    'send',
                    () => requestEmailCode(email.trim()),
                    () => setStep({ kind: 'code', email: email.trim() }),
                  )
                }
              />
            </>
          )}

          {step.kind === 'code' && (
            <>
              <View style={styles.intro}>
                <Text style={styles.title} accessibilityRole="header">
                  {t('signIn.codeTitle')}
                </Text>
                <Text style={styles.subtitle}>{t('signIn.codeSent', { email: step.email })}</Text>
              </View>
              <TextField
                icon="keypad-outline"
                value={code}
                onChangeText={(text) => setCode(text.replace(/\D/g, '').slice(0, CODE_LENGTH))}
                placeholder="000000"
                accessibilityLabel={t('signIn.codeLabel')}
                keyboardType="number-pad"
                autoComplete="one-time-code"
                textContentType="oneTimeCode"
                autoFocus
              />
              {errorText}
              <PrimaryButton
                label={t('signIn.verify')}
                loading={busy === 'verify'}
                disabled={code.length !== CODE_LENGTH}
                onPress={() => void run('verify', () => verifyEmailCode(step.email, code))}
              />
            </>
          )}
        </ScrollView>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
};

const useStyles = makeStyles((colors) => ({
  screen: { flex: 1, backgroundColor: colors.background },
  fill: { flex: 1 },
  content: {
    flexGrow: 1,
    paddingHorizontal: spacing.xl,
    paddingBottom: spacing.xxl,
    gap: spacing.lg,
  },
  intro: { gap: spacing.sm, marginTop: spacing.lg, marginBottom: spacing.md },
  title: { ...typography.wordmark, fontSize: 26, lineHeight: 32, color: colors.text },
  subtitle: { ...typography.body, color: colors.textSecondary },
  buttons: { gap: spacing.sm },
  error: { ...typography.caption, color: colors.danger },
  note: {
    ...typography.caption,
    color: colors.textTertiary,
    textAlign: 'center',
    marginTop: spacing.sm,
  },
}));
