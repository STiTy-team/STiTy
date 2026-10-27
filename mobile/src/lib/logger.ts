type Level = 'debug' | 'info' | 'warn' | 'error';

const write = (level: Level, scope: string, message: string, detail?: unknown) => {
  if (!__DEV__ && level === 'debug') return;
  const line = `[${scope}] ${message}`;
  const sink = level === 'debug' ? console.log : console[level];
  if (detail === undefined) sink(line);
  else sink(line, detail);
};

export const createLogger = (scope: string) => ({
  debug: (message: string, detail?: unknown) => write('debug', scope, message, detail),
  info: (message: string, detail?: unknown) => write('info', scope, message, detail),
  warn: (message: string, detail?: unknown) => write('warn', scope, message, detail),
  error: (message: string, detail?: unknown) => write('error', scope, message, detail),
});

export type Logger = ReturnType<typeof createLogger>;
