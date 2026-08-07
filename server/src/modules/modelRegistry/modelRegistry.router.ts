import { Router } from 'express';
import { handleGetModels, handleGetCurrentModel } from './modelRegistry.controller';

const router = Router();

router.get('/', handleGetModels);
router.get('/:condition/current', handleGetCurrentModel);

export default router;
