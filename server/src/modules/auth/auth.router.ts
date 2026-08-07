import { Router } from 'express';
import {
  handleRegister,
  handleLogin,
  handleLogout,
  handleMe,
  handleCSRF,
  handleVerifyEmail,
} from './auth.controller';
import { requireAuth } from '../../middleware/auth';
import { authRateLimiter } from '../../middleware/rateLimiter';
import { csrfProtection } from '../../middleware/csrf';

const router = Router();

// CSRF token endpoint (public)
router.get('/csrf', handleCSRF);

// Auth endpoints
router.post('/register', authRateLimiter, csrfProtection, handleRegister);
router.post('/login', authRateLimiter, csrfProtection, handleLogin);
router.post('/logout', csrfProtection, handleLogout);
router.post('/verify-email', handleVerifyEmail);

// Current user profile
router.get('/me', requireAuth(), handleMe);

export default router;
