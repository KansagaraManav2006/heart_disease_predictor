import { Request, Response, NextFunction } from 'express';
import { getSystemHealthOverview } from './systemHealth.service';

export async function handleGetSystemHealth(
  _req: Request,
  res: Response,
  next: NextFunction
): Promise<void> {
  try {
    const health = await getSystemHealthOverview();
    res.json(health);
  } catch (err) {
    next(err);
  }
}
