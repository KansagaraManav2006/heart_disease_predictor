import { Router } from 'express';
import authRouter from '../modules/auth/auth.router';
import healthRouter from '../modules/health/health.router';

const router = Router();

// Versioned API v1 endpoints
router.use('/auth', authRouter);
router.use('/health', healthRouter);

export default router;
