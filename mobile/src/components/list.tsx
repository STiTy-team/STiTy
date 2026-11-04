import { Ionicons } from '@expo/vector-icons';
import { Children, Fragment, type ReactNode } from 'react';
import { Pressable, Text, View } from 'react-native';

import { makeStyles, radius, spacing, typography, useColors } from '@/theme';

type SectionProps = {
  title?: string;
  footer?: string;
  children: ReactNode;
};

/** A titled group of rows on one surface, with hairlines between rows. */
export const ListSection = ({ title, footer, children }: SectionProps) => {
  const styles = useStyles();
  const rows = Children.toArray(children);
  return (
    <View>
      {title && (
        <Text style={styles.title} accessibilityRole="header">
          {title}
        </Text>
      )}
      <View style={styles.group}>
        {rows.map((row, index) => (
          <Fragment key={index}>
            {index > 0 && <View style={styles.separator} />}
            {row}
          </Fragment>
        ))}
      </View>
      {footer && <Text style={styles.footer}>{footer}</Text>}
    </View>
  );
};

type Tone = 'default' | 'action' | 'disabled' | 'destructive';

type RowProps = {
  title: string;
  /** A quiet line under the title that says what the row does. */
  subtitle?: string;
  onPress?: () => void;
  tone?: Tone;
  icon?: 'add';
  chevron?: boolean;
  accessibilityLabel?: string;
  accessibilityState?: { selected?: boolean; disabled?: boolean };
  /** Content after the title: a value, a badge, a switch. */
  children?: ReactNode;
};

export const ListRow = ({
  title,
  subtitle,
  onPress,
  tone = 'default',
  icon,
  chevron = false,
  accessibilityLabel,
  accessibilityState,
  children,
}: RowProps) => {
  const styles = useStyles();
  const colors = useColors();
  const titleColor = {
    default: colors.text,
    action: colors.accent,
    disabled: colors.textTertiary,
    destructive: colors.danger,
  }[tone];

  const titleText = (
    <Text
      style={[
        styles.rowTitle,
        { color: titleColor },
        (tone === 'action' || tone === 'destructive') && styles.strong,
        tone === 'destructive' && styles.centered,
      ]}
      numberOfLines={1}
    >
      {title}
    </Text>
  );

  const content = (
    <>
      {icon && <Ionicons name={icon} size={20} color={titleColor} />}
      {subtitle ? (
        <View style={styles.titles}>
          {titleText}
          <Text style={styles.subtitle}>{subtitle}</Text>
        </View>
      ) : (
        titleText
      )}
      {children}
      {chevron && <Ionicons name="chevron-forward" size={16} color={colors.textTertiary} />}
    </>
  );

  if (!onPress) {
    return (
      <View style={styles.row} accessibilityState={accessibilityState}>
        {content}
      </View>
    );
  }
  return (
    <Pressable
      onPress={onPress}
      disabled={tone === 'disabled'}
      accessibilityRole="button"
      accessibilityLabel={accessibilityLabel}
      accessibilityState={accessibilityState}
      style={({ pressed }) => [styles.row, pressed && styles.pressed]}
    >
      {content}
    </Pressable>
  );
};

export const ListValue = ({ children }: { children: string }) => {
  const styles = useStyles();
  return <Text style={styles.value}>{children}</Text>;
};

export const ListBadge = ({ children, muted = false }: { children: string; muted?: boolean }) => {
  const styles = useStyles();
  return (
    <View style={[styles.badge, muted && styles.badgeMuted]}>
      <Text style={[styles.badgeText, muted && styles.badgeTextMuted]}>{children}</Text>
    </View>
  );
};

const useStyles = makeStyles((colors) => ({
  title: {
    ...typography.small,
    color: colors.textTertiary,
    marginBottom: spacing.sm,
    marginLeft: spacing.xs,
  },
  group: { backgroundColor: colors.surface, borderRadius: radius.card, overflow: 'hidden' },
  separator: { height: 1, backgroundColor: colors.border, marginLeft: spacing.lg },
  footer: {
    ...typography.caption,
    color: colors.textTertiary,
    marginTop: spacing.sm,
    marginHorizontal: spacing.xs,
  },
  row: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
    minHeight: 52,
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.sm,
  },
  pressed: { backgroundColor: colors.surfaceStrong },
  rowTitle: { ...typography.body, flex: 1 },
  titles: { flex: 1, gap: 2, paddingVertical: spacing.xs },
  subtitle: { ...typography.caption, color: colors.textTertiary },
  strong: { fontFamily: typography.label.fontFamily },
  centered: { textAlign: 'center' },
  value: { ...typography.body, color: colors.textTertiary },
  badge: {
    backgroundColor: colors.accentSoft,
    borderRadius: 6,
    paddingHorizontal: 7,
    paddingVertical: 3,
  },
  badgeMuted: { backgroundColor: 'transparent', borderWidth: 1, borderColor: colors.border },
  badgeText: { ...typography.badge, color: colors.accent },
  badgeTextMuted: { color: colors.textTertiary },
}));
