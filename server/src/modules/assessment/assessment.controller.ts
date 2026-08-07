import { Request, Response, NextFunction } from 'express';
import { z } from 'zod';
import { ConditionType, RiskBand } from '@prisma/client';
import {
  createAssessmentRecord,
  getUserAssessments,
  getAssessmentById,
} from './assessment.service';
import { APIError } from '../../middleware/errorHandler';

const createAssessmentSchema = z.object({
  condition: z.enum(['DIABETES', 'HEART']),
  inputPayload: z.record(z.any()),
  modelVersion: z.string().default('v1.0'),
  probability: z.number().min(0).max(1),
  riskBand: z.enum(['LOW', 'MODERATE', 'HIGH']),
  outOfDistribution: z.boolean().optional(),
  observations: z
    .array(
      z.object({
        name: z.string(),
        value: z.number(),
        unit: z.string(),
        source: z.string().optional(),
      })
    )
    .optional(),
});

export async function handleCreateAssessment(
  req: Request,
  res: Response,
  next: NextFunction
): Promise<void> {
  try {
    if (!req.user) throw new APIError('Unauthorized', 401);
    const parseResult = createAssessmentSchema.safeParse(req.body);
    if (!parseResult.success) {
      throw new APIError('Validation error', 400, parseResult.error.flatten());
    }

    const assessment = await createAssessmentRecord({
      creatorId: req.user.id,
      condition: parseResult.data.condition as ConditionType,
      inputPayload: parseResult.data.inputPayload,
      modelVersion: parseResult.data.modelVersion,
      probability: parseResult.data.probability,
      riskBand: parseResult.data.riskBand as RiskBand,
      outOfDistribution: parseResult.data.outOfDistribution,
      observations: parseResult.data.observations,
    });

    res.status(201).json({ message: 'Assessment recorded successfully.', assessment });
  } catch (err) {
    next(err);
  }
}

export async function handleGetMyAssessments(
  req: Request,
  res: Response,
  next: NextFunction
): Promise<void> {
  try {
    if (!req.user) throw new APIError('Unauthorized', 401);
    const condition = req.query.condition as ConditionType | undefined;
    const assessments = await getUserAssessments(req.user.id, condition);
    res.json({ assessments });
  } catch (err) {
    next(err);
  }
}

export async function handleGetAssessmentById(
  req: Request,
  res: Response,
  next: NextFunction
): Promise<void> {
  try {
    if (!req.user) throw new APIError('Unauthorized', 401);
    const { id } = req.params;
    const assessment = await getAssessmentById(id, req.user);
    res.json({ assessment });
  } catch (err) {
    next(err);
  }
}
