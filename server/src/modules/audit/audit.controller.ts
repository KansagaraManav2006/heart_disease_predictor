import { Request, Response, NextFunction } from 'express';
import { getAuditEvents } from './audit.service';

export async function handleGetAuditEvents(
  req: Request,
  res: Response,
  next: NextFunction
): Promise<void> {
  try {
    const limit = parseInt(req.query.limit as string || '50', 10);
    const skip = parseInt(req.query.skip as string || '0', 10);

    const data = await getAuditEvents(limit, skip);
    res.json(data);
  } catch (err) {
    next(err);
  }
}
