import { Request, Response, NextFunction } from 'express';
import { Role } from '@prisma/client';
import { prisma } from '../db/prisma';
import { APIError } from './errorHandler';

export interface AuthenticatedUser {
  id: string;
  email: string;
  role: Role;
  status: string;
}

declare global {
  namespace Express {
    interface Request {
      user?: AuthenticatedUser;
      sessionId?: string;
    }
  }
}

export const SESSION_COOKIE_NAME = 'healthlens_session';

export function requireAuth(allowedRoles?: Role[]) {
  return async (
    req: Request,
    _res: Response,
    next: NextFunction
  ): Promise<void> => {
    try {
      const sessionId = req.cookies?.[SESSION_COOKIE_NAME];

      if (!sessionId) {
        return next(
          new APIError('Authentication required. Please sign in.', 401)
        );
      }

      // Find session in PostgreSQL
      const session = await prisma.session.findUnique({
        where: { id: sessionId },
        include: {
          user: {
            select: {
              id: true,
              email: true,
              role: true,
              status: true,
            },
          },
        },
      });

      if (!session) {
        return next(
          new APIError('Session invalid or expired. Please sign in again.', 401)
        );
      }

      if (session.expiresAt < new Date()) {
        // Cleanup expired session
        await prisma.session.delete({ where: { id: sessionId } }).catch(() => {});
        return next(
          new APIError('Session expired. Please sign in again.', 401)
        );
      }

      if (session.user.status === 'SUSPENDED') {
        return next(
          new APIError('Account suspended. Contact administration.', 403)
        );
      }

      // Enforce RBAC if roles specified
      if (allowedRoles && allowedRoles.length > 0) {
        if (!allowedRoles.includes(session.user.role)) {
          return next(
            new APIError(
              'Access denied. Insufficient permissions for this action.',
              403
            )
          );
        }
      }

      req.user = session.user;
      req.sessionId = session.id;
      next();
    } catch (err) {
      next(err);
    }
  };
}
