import { Router } from 'express';
import { handleGetAuditEvents } from './audit.controller';
import { requireAuth } from '../../middleware/auth';

const router = Router();

router.get('/', requireAuth(['ADMIN', 'CLINICIAN']), handleGetAuditEvents);

export default router;
