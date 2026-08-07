import { Request, Response, NextFunction } from 'express';
import { z } from 'zod';
import {
  registerUser,
  loginUser,
  logoutUser,
  verifyEmailToken,
  createSessionCookie,
  clearSessionCookie,
} from './auth.service';
import { generateCSRFToken, setCSRFCookie } from '../../middleware/csrf';
import { SESSION_COOKIE_NAME } from '../../middleware/auth';
import { APIError } from '../../middleware/errorHandler';

const registerSchema = z.object({
  email: z.string().email('Invalid email address'),
  password: z.string().min(8, 'Password must be at least 8 characters long'),
  role: z.enum(['PATIENT', 'CLINICIAN']).optional(),
  fullName: z.string().min(2).optional(),
});

const loginSchema = z.object({
  email: z.string().email('Invalid email address'),
  password: z.string().min(1, 'Password is required'),
});

export async function handleRegister(
  req: Request,
  res: Response,
  next: NextFunction
): Promise<void> {
  try {
    const parseResult = registerSchema.safeParse(req.body);
    if (!parseResult.success) {
      throw new APIError('Validation error', 400, parseResult.error.flatten());
    }

    const user = await registerUser(parseResult.data);

    res.status(201).json({
      message:
        'Registration successful. Please verify your email to activate account.',
      user: {
        id: user.id,
        email: user.email,
        role: user.role,
        status: user.status,
      },
      // Included in development mode for easy verification testing
      ...(process.env.NODE_ENV !== 'production' && {
        devVerifyToken: user.emailVerifyToken,
      }),
    });
  } catch (err) {
    next(err);
  }
}

export async function handleLogin(
  req: Request,
  res: Response,
  next: NextFunction
): Promise<void> {
  try {
    const parseResult = loginSchema.safeParse(req.body);
    if (!parseResult.success) {
      throw new APIError('Validation error', 400, parseResult.error.flatten());
    }

    const result = await loginUser(parseResult.data, {
      userAgent: req.headers['user-agent'],
      ipAddress: req.ip,
    });

    createSessionCookie(res, result.session.id);

    // Issue fresh CSRF token upon login
    const csrfToken = generateCSRFToken();
    setCSRFCookie(res, csrfToken);

    res.json({
      message: 'Sign in successful.',
      user: result.user,
      csrfToken,
    });
  } catch (err) {
    next(err);
  }
}

export async function handleLogout(
  req: Request,
  res: Response,
  next: NextFunction
): Promise<void> {
  try {
    const sessionId = req.cookies?.[SESSION_COOKIE_NAME];
    if (sessionId) {
      await logoutUser(sessionId, req.user?.id);
    }
    clearSessionCookie(res);
    res.json({ message: 'Signed out successfully.' });
  } catch (err) {
    next(err);
  }
}

export async function handleMe(req: Request, res: Response): Promise<void> {
  res.json({
    user: req.user,
  });
}

export async function handleCSRF(_req: Request, res: Response): Promise<void> {
  const csrfToken = generateCSRFToken();
  setCSRFCookie(res, csrfToken);
  res.json({ csrfToken });
}

export async function handleVerifyEmail(
  req: Request,
  res: Response,
  next: NextFunction
): Promise<void> {
  try {
    const token = req.body?.token || req.query?.token;
    if (!token || typeof token !== 'string') {
      throw new APIError('Verification token is required.', 400);
    }

    const user = await verifyEmailToken(token);

    res.json({
      message: 'Email address verified successfully.',
      user,
    });
  } catch (err) {
    next(err);
  }
}
