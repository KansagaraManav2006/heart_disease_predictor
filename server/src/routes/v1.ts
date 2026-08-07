import { Router } from 'express';
import authRouter from '../modules/auth/auth.router';
import healthRouter from '../modules/health/health.router';
import patientRouter from '../modules/patient/patient.router';
import accessGrantRouter from '../modules/accessGrant/accessGrant.router';
import assessmentRouter from '../modules/assessment/assessment.router';

const router = Router();

// Versioned API v1 endpoints
router.use('/auth', authRouter);
router.use('/health', healthRouter);
router.use('/patient', patientRouter);
router.use('/access', accessGrantRouter);
router.use('/assessments', assessmentRouter);

export default router;
