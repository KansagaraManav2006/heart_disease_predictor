import { Router } from 'express';
import { handleGetMyProfile, handleUpdateMyProfile } from './patient.controller';
import { requireAuth } from '../../middleware/auth';
import { csrfProtection } from '../../middleware/csrf';

const router = Router();

router.get('/me', requireAuth(['PATIENT', 'ADMIN']), handleGetMyProfile);
router.put('/me', requireAuth(['PATIENT', 'ADMIN']), csrfProtection, handleUpdateMyProfile);

export default router;
