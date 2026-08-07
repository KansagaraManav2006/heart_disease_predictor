import { Router } from 'express';
import {
  handleCreateAssessment,
  handleGetMyAssessments,
  handleGetAssessmentById,
} from './assessment.controller';
import { requireAuth } from '../../middleware/auth';
import { csrfProtection } from '../../middleware/csrf';

const router = Router();

router.post('/', requireAuth(), csrfProtection, handleCreateAssessment);
router.get('/', requireAuth(), handleGetMyAssessments);
router.get('/:id', requireAuth(), handleGetAssessmentById);

export default router;
