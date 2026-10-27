import { Text, type TextStyle } from 'react-native';

import { parseInlineMarkup } from '@/lib/inline-markup';
import { fonts } from '@/theme';

/** Renders a translated string that may contain <bold>…</bold>. */
export const RichText = ({ text, style }: { text: string; style?: TextStyle }) => (
  <Text style={style}>
    {parseInlineMarkup(text).map((segment, index) => (
      <Text key={index} style={segment.bold ? { fontFamily: fonts.bold } : undefined}>
        {segment.text}
      </Text>
    ))}
  </Text>
);
