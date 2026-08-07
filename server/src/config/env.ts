import dotenv from 'dotenv';
import path from 'path';

// Load .env from server directory or root
dotenv.config({ path: path.join(__dirname, '../../.env') });
dotenv.config();

export const ENV = {
  NODE_ENV: process.env.NODE_ENV || 'development',
  PORT: parseInt(process.env.PORT || '5000', 10),
  DATABASE_URL:
    process.env.DATABASE_URL ||
    'postgresql://healthlens:healthlens_pass@localhost:5432/healthlens_db?schema=public',
  SESSION_SECRET:
    process.env.SESSION_SECRET || 'healthlens-dev-secret-change-in-production',
  ALLOWED_ORIGINS: process.env.ALLOWED_ORIGIN
    ? process.env.ALLOWED_ORIGIN.split(',').map((s) => s.trim())
    : ['http://localhost:5173', 'http://localhost:5000'],
  INTERNAL_ML_SERVICE_URL:
    process.env.INTERNAL_ML_SERVICE_URL || 'http://localhost:8000',
  IS_PROD: process.env.NODE_ENV === 'production',
};
