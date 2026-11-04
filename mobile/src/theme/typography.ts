import type { TextStyle } from 'react-native';

/** Pretendard is embedded per weight (see app.config.ts), so weight is chosen by family. */
export const fonts = {
  regular: 'Pretendard-Regular',
  medium: 'Pretendard-Medium',
  semibold: 'Pretendard-SemiBold',
  bold: 'Pretendard-Bold',
} as const;

export const typography = {
  wordmark: { fontFamily: fonts.bold, fontSize: 20, lineHeight: 24, letterSpacing: -0.6 },
  title: { fontFamily: fonts.bold, fontSize: 17, lineHeight: 22, letterSpacing: -0.2 },
  translation: { fontFamily: fonts.semibold, fontSize: 18, lineHeight: 26, letterSpacing: -0.2 },
  /** One sentence filling the screen, under the large text add-on. */
  display: { fontFamily: fonts.bold, fontSize: 34, lineHeight: 42, letterSpacing: -0.6 },
  /** Body text in the transcript viewer; scaled by the reading size control. */
  reading: { fontFamily: fonts.regular, fontSize: 17, lineHeight: 27 },
  body: { fontFamily: fonts.regular, fontSize: 15, lineHeight: 22 },
  bodyStrong: { fontFamily: fonts.medium, fontSize: 15, lineHeight: 22 },
  button: { fontFamily: fonts.bold, fontSize: 16, lineHeight: 20 },
  label: { fontFamily: fonts.semibold, fontSize: 13, lineHeight: 18 },
  caption: { fontFamily: fonts.regular, fontSize: 13, lineHeight: 19 },
  small: { fontFamily: fonts.medium, fontSize: 12, lineHeight: 16 },
  badge: { fontFamily: fonts.bold, fontSize: 11, lineHeight: 14 },
} satisfies Record<string, TextStyle>;
