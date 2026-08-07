import { Request, Response, NextFunction } from 'express';
import { ENV } from '../config/env';

export class APIError extends Error {
  public statusCode: number;
  public details?: any;

  constructor(message: string, statusCode = 500, details?: any) {
    super(message);
    this.statusCode = statusCode;
    this.details = details;
    Object.setPrototypeOf(this, APIError.prototype);
  }
}

export function errorHandler(
  err: Error,
  _req: Request,
  res: Response,
  _next: NextFunction
): void {
  if (err instanceof APIError) {
    res.status(err.statusCode).json({
      error: err.message,
      ...(err.details && { details: err.details }),
    });
    return;
  }

  // Handle Multer file size errors
  if ((err as any).code === 'LIMIT_FILE_SIZE') {
    res.status(413).json({ error: 'File too large. Maximum size is 10 MB.' });
    return;
  }

  if (err.message?.startsWith('Unsupported file type')) {
    res.status(415).json({ error: err.message });
    return;
  }

  console.error('Unhandled Server Error:', err);

  res.status(500).json({
    error: 'Internal server error.',
    ...(!ENV.IS_PROD && { details: err.message, stack: err.stack }),
  });
}
