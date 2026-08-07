import { Request, Response, NextFunction } from 'express';
import crypto from 'crypto';
import { APIError } from './errorHandler';

const CSRF_COOKIE_NAME = 'healthlens_csrf';
const CSRF_HEADER_NAME = 'x-csrf-token';

export function generateCSRFToken(): string {
  return crypto.randomBytes(32).toString('hex');
}

export function setCSRFCookie(res: Response, token: string): void {
  res.cookie(CSRF_COOKIE_NAME, token, {
    httpOnly: false, // Accessible by frontend JavaScript to read and attach to header
    sameSite: 'strict',
    secure: process.env.NODE_ENV === 'production',
    path: '/',
  });
}

export function csrfProtection(
  req: Request,
  _res: Response,
  next: NextFunction
): void {
  // Skip GET, HEAD, OPTIONS
  if (['GET', 'HEAD', 'OPTIONS'].includes(req.method)) {
    return next();
  }

  const tokenInCookie = req.cookies?.[CSRF_COOKIE_NAME];
  const tokenInHeader = req.headers[CSRF_HEADER_NAME] as string;

  if (!tokenInCookie || !tokenInHeader) {
    return next(
      new APIError(
        'CSRF token missing. Please fetch a fresh CSRF token.',
        403
      )
    );
  }

  if (
    !crypto.timingSafeEqual(
      Buffer.from(tokenInCookie),
      Buffer.from(tokenInHeader)
    )
  ) {
    return next(
      new APIError(
        'Invalid CSRF token. Request rejected.',
        403
      )
    );
  }

  next();
}
