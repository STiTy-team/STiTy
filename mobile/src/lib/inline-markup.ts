export type InlineSegment = { text: string; bold: boolean };

/** Splits "a <bold>b</bold> c" into styled segments for translated rich text. */
export const parseInlineMarkup = (text: string): InlineSegment[] =>
  text
    .split(/(<bold>.*?<\/bold>)/g)
    .filter(Boolean)
    .map((part) => {
      const match = /^<bold>(.*)<\/bold>$/.exec(part);
      return match ? { text: match[1] ?? '', bold: true } : { text: part, bold: false };
    });
