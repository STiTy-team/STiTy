export const API_ERROR_KINDS = [
  'offline',
  'unauthorized',
  'notFound',
  'server',
  'invalid',
] as const;

export type ApiErrorKind = (typeof API_ERROR_KINDS)[number];

/** Every failure a screen can see. Screens branch on `kind`, never on HTTP codes. */
export class ApiError extends Error {
  constructor(
    readonly kind: ApiErrorKind,
    message: string = kind,
  ) {
    super(message);
  }
}

export const toApiError = (error: unknown): ApiError =>
  error instanceof ApiError ? error : new ApiError('server', String(error));

export const apiErrorKind = (error: unknown): ApiErrorKind => toApiError(error).kind;
