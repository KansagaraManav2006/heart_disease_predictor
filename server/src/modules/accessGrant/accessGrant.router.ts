import { Router } from 'express';
import {
  handleGrantAccess,
  handleRevokeAccess,
  handleGetAssignedPatients,
} from './accessGrant.controller';
import { requireAuth } from '../../middleware/auth';
import { csrfProtection } from '../../middleware/csrf';

const router = Router();

// Patient routes for managing access grants
router.post('/grants', requireAuth(['PATIENT']), csrfProtection, handleGrantAccess);
router.delete('/grants/:id', requireAuth(['PATIENT']), csrfProtection, handleRevokeAccess);

// Clinician route for viewing assigned patients
router.get('/assigned-patients', requireAuth(['CLINICIAN', 'ADMIN']), handleGetAssignedPatients);

export default router;
