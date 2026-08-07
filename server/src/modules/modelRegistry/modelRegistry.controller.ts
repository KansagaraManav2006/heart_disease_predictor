import { Request, Response, NextFunction } from 'express';
import { ConditionType } from '@prisma/client';
import { getModelRegistryList, getCurrentModelForCondition } from './modelRegistry.service';
import { APIError } from '../../middleware/errorHandler';

export async function handleGetModels(
  _req: Request,
  res: Response,
  next: NextFunction
): Promise<void> {
  try {
    const models = await getModelRegistryList();
    res.json({ models });
  } catch (err) {
    next(err);
  }
}

export async function handleGetCurrentModel(
  req: Request,
  res: Response,
  next: NextFunction
): Promise<void> {
  try {
    const condition = req.params.condition.toUpperCase() as ConditionType;
    if (!['DIABETES', 'HEART'].includes(condition)) {
      throw new APIError('Invalid condition type.', 400);
    }
    const model = await getCurrentModelForCondition(condition);
    res.json({ model });
  } catch (err) {
    next(err);
  }
}
