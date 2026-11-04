import { memo } from 'react';
import { Pressable, Text, View } from 'react-native';

import { highlight } from '@/features/records/model/records-model';
import { makeStyles, radius, spacing, typography } from '@/theme';

type Props = {
  id: string;
  label: string;
  isMe: boolean;
  primary: string;
  secondary: string;
  query: string;
  active: boolean;
  scale: number;
  copyLabel: string;
  onLongPress: (id: string) => void;
};

const Highlighted = ({
  text,
  query,
  hitStyle,
}: {
  text: string;
  query: string;
  hitStyle: object;
}) => (
  <>
    {highlight(text, query).map((part, index) => (
      <Text key={index} style={part.hit ? hitStyle : undefined}>
        {part.text}
      </Text>
    ))}
  </>
);

/** One speaker turn: "Me · 04:12", then the text in the chosen view. Long press copies it. */
export const ParagraphItem = memo(function ParagraphItem({
  id,
  label,
  isMe,
  primary,
  secondary,
  query,
  active,
  scale,
  copyLabel,
  onLongPress,
}: Props) {
  const styles = useStyles();
  const reading = {
    fontSize: typography.reading.fontSize * scale,
    lineHeight: typography.reading.lineHeight * scale,
  };
  const small = {
    fontSize: typography.caption.fontSize * scale,
    lineHeight: typography.caption.lineHeight * scale,
  };

  return (
    <Pressable
      onLongPress={() => onLongPress(id)}
      accessibilityLabel={`${label}. ${primary}${secondary ? `. ${secondary}` : ''}`}
      accessibilityActions={[{ name: 'longpress', label: copyLabel }]}
      onAccessibilityAction={() => onLongPress(id)}
      style={[styles.item, active && styles.active]}
    >
      <Text style={[styles.label, isMe && styles.labelMe]}>{label}</Text>
      <Text style={[styles.primary, reading]}>
        <Highlighted text={primary} query={query} hitStyle={styles.hit} />
      </Text>
      {!!secondary && (
        <View style={styles.secondaryBlock}>
          <Text style={[styles.secondary, small]}>
            <Highlighted text={secondary} query={query} hitStyle={styles.hit} />
          </Text>
        </View>
      )}
    </Pressable>
  );
});

const useStyles = makeStyles((colors) => ({
  item: {
    marginHorizontal: spacing.sm,
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.md,
    borderRadius: radius.control,
    gap: spacing.xs,
  },
  active: { backgroundColor: colors.accentSoft },
  label: { ...typography.small, color: colors.textTertiary, fontVariant: ['tabular-nums'] },
  labelMe: { color: colors.accent },
  primary: { ...typography.reading, color: colors.text },
  secondaryBlock: {
    borderLeftWidth: 2,
    borderLeftColor: colors.border,
    paddingLeft: spacing.sm + 2,
    marginTop: 2,
  },
  secondary: { ...typography.caption, color: colors.textSecondary },
  hit: {
    backgroundColor: colors.warningSoft,
    color: colors.text,
    fontFamily: typography.bodyStrong.fontFamily,
  },
}));
