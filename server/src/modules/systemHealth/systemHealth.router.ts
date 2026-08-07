import { Router } from 'express';
import { handleGetSystemHealth } from './systemHealth.controller';
import { requireAuth } from '../../middleware/auth';

const router = Router();

router.get('/', requireAuth(['ADMIN', 'CLINICIAN']), handleGetSystemHealth);

export default router;
