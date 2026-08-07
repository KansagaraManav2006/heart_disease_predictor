import { Router } from 'express';
import { handleQueryKnowledge, handleGetDocuments } from './knowledge.controller';

const router = Router();

router.post('/query', handleQueryKnowledge);
router.get('/documents', handleGetDocuments);

export default router;
