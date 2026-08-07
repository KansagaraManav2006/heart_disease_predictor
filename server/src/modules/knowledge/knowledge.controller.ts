import { Request, Response, NextFunction } from 'express';
import { z } from 'zod';
import { queryMedicalKnowledge, getKnowledgeDocuments } from './knowledge.service';
import { APIError } from '../../middleware/errorHandler';

const querySchema = z.object({
  query: z.string().min(2, 'Query must be at least 2 characters long'),
});

export async function handleQueryKnowledge(
  req: Request,
  res: Response,
  next: NextFunction
): Promise<void> {
  try {
    const parseResult = querySchema.safeParse(req.body);
    if (!parseResult.success) {
      throw new APIError('Validation error', 400, parseResult.error.flatten());
    }

    const result = await queryMedicalKnowledge(parseResult.data.query);
    res.json(result);
  } catch (err) {
    next(err);
  }
}

export async function handleGetDocuments(
  _req: Request,
  res: Response,
  next: NextFunction
): Promise<void> {
  try {
    const documents = await getKnowledgeDocuments();
    res.json({ documents });
  } catch (err) {
    next(err);
  }
}
