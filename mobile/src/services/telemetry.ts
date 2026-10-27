import { createLogger } from '@/lib/logger';

/** Usage events. None carries transcript text: only timings and codes. */
export type TelemetryEvent =
  | { name: 'session_started' }
  | { name: 'first_translation_shown'; ms: number }
  | { name: 'session_ended'; durationMs: number }
  | { name: 'session_error'; code: string }
  | { name: 'record_opened' };

export type TelemetrySink = {
  track(event: TelemetryEvent): void;
  captureError(error: unknown, context: { fatal: boolean }): void;
};

const log = createLogger('telemetry');

const loggerSink: TelemetrySink = {
  track: (event) => log.info(event.name, event),
  captureError: (error, { fatal }) => log.error(fatal ? 'fatal error' : 'error', error),
};

let sink: TelemetrySink = loggerSink;

/** Plugs in the crash and analytics vendor. Until one is chosen, events go to the device log. */
export const setTelemetrySink = (next: TelemetrySink) => {
  sink = next;
};

export const track = (event: TelemetryEvent) => sink.track(event);

export const captureError = (error: unknown) => sink.captureError(error, { fatal: false });

type GlobalHandler = (error: unknown, isFatal?: boolean) => void;
type ErrorUtilsShape = {
  getGlobalHandler(): GlobalHandler;
  setGlobalHandler(h: GlobalHandler): void;
};

let installed = false;

/** Reports uncaught JavaScript errors, then hands them to React Native's own handler. */
export const installCrashReporting = () => {
  const errorUtils = (globalThis as { ErrorUtils?: ErrorUtilsShape }).ErrorUtils;
  if (installed || !errorUtils) return;
  installed = true;
  const previous = errorUtils.getGlobalHandler();
  errorUtils.setGlobalHandler((error, isFatal) => {
    sink.captureError(error, { fatal: !!isFatal });
    previous(error, isFatal);
  });
};
