const listeners = new Set<() => void>();

/** Called when the server no longer accepts this user's tokens, so the app can sign out. */
export const onUnauthorized = (listener: () => void): (() => void) => {
  listeners.add(listener);
  return () => listeners.delete(listener);
};

export const emitUnauthorized = () => listeners.forEach((listener) => listener());
