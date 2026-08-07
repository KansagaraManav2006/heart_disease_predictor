import { Router } from 'express';
import authRouter from '../modules/auth/auth.router';
import healthRouter from '../modules/health/health.router';
import patientRouter from '../modules/patient/patient.router';
import accessGrantRouter from '../modules/accessGrant/accessGrant.router';
import assessmentRouter from '../modules/assessment/assessment.router';
import modelRegistryRouter from '../modules/modelRegistry/modelRegistry.router';
import knowledgeRouter from '../modules/knowledge/knowledge.router';
import auditRouter from '../modules/audit/audit.router';
import systemHealthRouter from '../modules/systemHealth/systemHealth.router';
import medicationRouter from '../modules/medication/medication.router';
import biomarkerTrendRouter from '../modules/biomarkerTrend/biomarkerTrend.router';
import riskScenarioRouter from '../modules/riskScenario/riskScenario.router';

const router = Router();

// Versioned API v1 endpoints
router.use('/auth', authRouter);
router.use('/health', healthRouter);
router.use('/patient', patientRouter);
router.use('/access', accessGrantRouter);
router.use('/assessments', assessmentRouter);
router.use('/models', modelRegistryRouter);
router.use('/knowledge', knowledgeRouter);
router.use('/audit', auditRouter);
router.use('/system-health', systemHealthRouter);
router.use('/medications', medicationRouter);
router.use('/biomarker-trends', biomarkerTrendRouter);
router.use('/risk-scenarios', riskScenarioRouter);

export default router;
