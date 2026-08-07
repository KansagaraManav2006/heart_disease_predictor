import { Request, Response, NextFunction } from 'express';
import { z } from 'zod';
import { getPatientProfileByUserId, updatePatientProfile } from './patient.service';
import { APIError } from '../../middleware/errorHandler';

const profileUpdateSchema = z.object({
  fullName: z.string().min(2).optional(),
  dateOfBirth: z.string().optional(),
  biologicalSex: z.enum(['male', 'female', 'other']).optional(),
  emergencyPhone: z.string().optional(),
});

export async function handleGetMyProfile(
  req: Request,
  res: Response,
  next: NextFunction
): Promise<void> {
  try {
    if (!req.user) {
      throw new APIError('Unauthorized', 401);
    }
    const profile = await getPatientProfileByUserId(req.user.id);
    res.json({ profile });
  } catch (err) {
    next(err);
  }
}

export async function handleUpdateMyProfile(
  req: Request,
  res: Response,
  next: NextFunction
): Promise<void> {
  try {
    if (!req.user) {
      throw new APIError('Unauthorized', 401);
    }
    const parseResult = profileUpdateSchema.safeParse(req.body);
    if (!parseResult.success) {
      throw new APIError('Validation error', 400, parseResult.error.flatten());
    }
    const updated = await updatePatientProfile(req.user.id, parseResult.data);
    res.json({ message: 'Profile updated successfully.', profile: updated });
  } catch (err) {
    next(err);
  }
}
