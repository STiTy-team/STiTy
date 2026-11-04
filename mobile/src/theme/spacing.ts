export const spacing = {
  xxs: 2,
  xs: 4,
  sm: 8,
  md: 12,
  lg: 16,
  xl: 20,
  xxl: 24,
  xxxl: 32,
} as const;

/** Radius grows with the size of the element: segments < controls < cards < sheets. */
export const radius = {
  segment: 9,
  control: 12,
  card: 16,
  sheet: 26,
  pill: 999,
} as const;
