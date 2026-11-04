import { useTranslation } from 'react-i18next';
import { Text, View } from 'react-native';

import { RichText } from '@/features/settings/components/rich-text';
import { ScreenScroll } from '@/features/settings/components/screen-scroll';
import { makeStyles, spacing, typography } from '@/theme';

type Block = { type: 'heading' | 'paragraph' | 'bullet'; text: string };

type Props = { document: 'privacy' | 'terms' };

export const LegalDocumentScreen = ({ document }: Props) => {
  const { t } = useTranslation();
  const styles = useStyles();
  const blocks = t(`${document}.blocks`, { returnObjects: true }) as Block[];

  return (
    <ScreenScroll>
      <View style={styles.document}>
        <Text style={styles.updated}>{t(`${document}.updated`)}</Text>
        {blocks.map((block, index) => {
          switch (block.type) {
            case 'heading':
              return (
                <Text key={index} style={styles.heading} accessibilityRole="header">
                  {block.text}
                </Text>
              );
            case 'paragraph':
              return <RichText key={index} text={block.text} style={styles.paragraph} />;
            case 'bullet':
              return <RichText key={index} text={`•  ${block.text}`} style={styles.bullet} />;
          }
        })}
      </View>
    </ScreenScroll>
  );
};

const useStyles = makeStyles((colors) => ({
  document: { paddingHorizontal: spacing.xs, gap: spacing.sm },
  updated: { ...typography.small, color: colors.textTertiary, marginBottom: spacing.sm },
  heading: { ...typography.title, color: colors.text, marginTop: spacing.md },
  paragraph: { ...typography.body, color: colors.textSecondary },
  bullet: { ...typography.body, color: colors.textSecondary, marginLeft: spacing.sm },
}));
