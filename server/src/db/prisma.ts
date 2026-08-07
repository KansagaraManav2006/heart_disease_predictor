import { PrismaClient } from '@prisma/client';
import { ENV } from '../config/env';

declare global {
  // Prevent multiple Prisma Client instances in dev hot-reloading
  // eslint-disable-next-line no-var
  var prismaGlobal: PrismaClient | undefined;
}

export const prisma =
  globalThis.prismaGlobal ??
  new PrismaClient({
    log: ENV.IS_PROD ? ['error', 'warn'] : ['query', 'error', 'warn'],
  });

if (!ENV.IS_PROD) {
  globalThis.prismaGlobal = prisma;
}
