export const TYPE_BUDGET_MS = 1_300;
const TYPE_MIN_MS = 10;
const TYPE_MAX_MS = 45;

export const commonPrefixLength = (a: string, b: string): number => {
  const limit = Math.min(a.length, b.length);
  let i = 0;
  while (i < limit && a[i] === b[i]) i += 1;
  return i;
};

/** Even pacing that finishes the backlog before the next partial usually arrives. */
export const typingInterval = (backlog: number): number =>
  Math.min(TYPE_MAX_MS, Math.max(TYPE_MIN_MS, TYPE_BUDGET_MS / Math.max(1, backlog)));
