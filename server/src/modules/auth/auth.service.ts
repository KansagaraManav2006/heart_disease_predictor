import argon2 from 'argon2';
import crypto from 'crypto';
import { Response } from 'express';
import { Role } from '@prisma/client';
import { prisma } from '../../db/prisma';
import { APIError } from '../../middleware/errorHandler';
import { SESSION_COOKIE_NAME } from '../../middleware/auth';
import { ENV } from '../../config/env';

const SESSION_TTL_DAYS = 7;

export async function hashPassword(password: string): Promise<string> {
  return argon2.hash(password, {
    type: argon2.argon2id,
    memoryCost: 65536, // 64 MB
    timeCost: 3,
    parallelism: 4,
  });
}

export async function verifyPassword(
  hash: string,
  plain: string
): Promise<boolean> {
  try {
    return await argon2.verify(hash, plain);
  } catch {
    return false;
  }
}

export function createSessionCookie(res: Response, sessionId: string): void {
  res.cookie(SESSION_COOKIE_NAME, sessionId, {
    httpOnly: true,
    sameSite: 'strict',
    secure: ENV.IS_PROD,
    maxAge: SESSION_TTL_DAYS * 24 * 60 * 60 * 1000, // 7 days in ms
    path: '/',
  });
}

export function clearSessionCookie(res: Response): void {
  res.clearCookie(SESSION_COOKIE_NAME, {
    httpOnly: true,
    sameSite: 'strict',
    secure: ENV.IS_PROD,
    path: '/',
  });
}

export async function registerUser(input: {
  email: string;
  password: string;
  role?: Role;
  fullName?: string;
}) {
  const normalizedEmail = input.email.trim().toLowerCase();

  // Check if user already exists
  const existing = await prisma.user.findUnique({
    where: { email: normalizedEmail },
  });

  if (existing) {
    throw new APIError('User with this email already exists.', 409);
  }

  const passwordHash = await hashPassword(input.password);
  const verifyToken = crypto.randomBytes(32).toString('hex');
  const role = input.role && ['PATIENT', 'CLINICIAN'].includes(input.role) ? input.role : 'PATIENT';

  const user = await prisma.user.create({
    data: {
      email: normalizedEmail,
      passwordHash,
      role: role as Role,
      status: 'PENDING_VERIFICATION',
      emailVerifyToken: verifyToken,
      ...(role === 'PATIENT' && {
        patientProfile: {
          create: {
            fullName: input.fullName || normalizedEmail.split('@')[0],
          },
        },
      }),
    },
    select: {
      id: true,
      email: true,
      role: true,
      status: true,
      emailVerifyToken: true, // Included for dev token-in-response
      createdAt: true,
    },
  });

  // Audit event
  await prisma.auditEvent.create({
    data: {
      actorId: user.id,
      action: 'USER_REGISTERED',
      entityType: 'User',
      entityId: user.id,
      metadata: { role: user.role },
    },
  });

  return user;
}

export async function loginUser(
  input: { email: string; password: string },
  meta?: { userAgent?: string; ipAddress?: string }
) {
  const normalizedEmail = input.email.trim().toLowerCase();

  const user = await prisma.user.findUnique({
    where: { email: normalizedEmail },
  });

  if (!user) {
    throw new APIError('Invalid email or password.', 401);
  }

  const isValid = await verifyPassword(user.passwordHash, input.password);
  if (!isValid) {
    throw new APIError('Invalid email or password.', 401);
  }

  if (user.status === 'SUSPENDED') {
    throw new APIError('Account suspended. Please contact administrator.', 403);
  }

  // Create session
  const expiresAt = new Date();
  expiresAt.setDate(expiresAt.getDate() + SESSION_TTL_DAYS);

  const session = await prisma.session.create({
    data: {
      userId: user.id,
      expiresAt,
      userAgent: meta?.userAgent,
      ipHash: meta?.ipAddress
        ? crypto.createHash('sha256').update(meta.ipAddress).digest('hex')
        : undefined,
    },
  });

  // Audit event
  await prisma.auditEvent.create({
    data: {
      actorId: user.id,
      action: 'USER_LOGGED_IN',
      entityType: 'Session',
      entityId: session.id,
    },
  });

  return {
    session,
    user: {
      id: user.id,
      email: user.email,
      role: user.role,
      status: user.status,
      verifiedAt: user.verifiedAt,
    },
  };
}

export async function logoutUser(sessionId: string, userId?: string) {
  await prisma.session.delete({ where: { id: sessionId } }).catch(() => {});

  if (userId) {
    await prisma.auditEvent.create({
      data: {
        actorId: userId,
        action: 'USER_LOGGED_OUT',
        entityType: 'Session',
        entityId: sessionId,
      },
    });
  }
}

export async function verifyEmailToken(token: string) {
  const user = await prisma.user.findFirst({
    where: { emailVerifyToken: token },
  });

  if (!user) {
    throw new APIError('Invalid or expired verification token.', 400);
  }

  const updated = await prisma.user.update({
    where: { id: user.id },
    data: {
      status: 'ACTIVE',
      verifiedAt: new Date(),
      emailVerifyToken: null,
    },
    select: {
      id: true,
      email: true,
      role: true,
      status: true,
      verifiedAt: true,
    },
  });

  await prisma.auditEvent.create({
    data: {
      actorId: user.id,
      action: 'EMAIL_VERIFIED',
      entityType: 'User',
      entityId: user.id,
    },
  });

  return updated;
}
