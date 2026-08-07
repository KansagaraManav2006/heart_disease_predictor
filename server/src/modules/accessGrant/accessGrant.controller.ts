import { Request, Response, NextFunction } from 'express';
import { z } from 'zod';
import {
  grantAccessToClinician,
  revokeClinicianAccess,
  getAssignedPatientsForClinician,
} from './accessGrant.service';
import { APIError } from '../../middleware/errorHandler';

const grantSchema = z.object({
  clinicianEmail: z.string().email('Invalid clinician email'),
});

export async function handleGrantAccess(
  req: Request,
  res: Response,
  next: NextFunction
): Promise<void> {
  try {
    if (!req.user) throw new APIError('Unauthorized', 401);
    const parseResult = grantSchema.safeParse(req.body);
    if (!parseResult.success) {
      throw new APIError('Validation error', 400, parseResult.error.flatten());
    }
    const grant = await grantAccessToClinician(req.user.id, parseResult.data.clinicianEmail);
    res.status(201).json({ message: 'Clinician access granted successfully.', grant });
  } catch (err) {
    next(err);
  }
}

export async function handleRevokeAccess(
  req: Request,
  res: Response,
  next: NextFunction
): Promise<void> {
  try {
    if (!req.user) throw new APIError('Unauthorized', 401);
    const { id } = req.params;
    const revoked = await revokeClinicianAccess(req.user.id, id);
    res.json({ message: 'Clinician access revoked.', grant: revoked });
  } catch (err) {
    next(err);
  }
}

export async function handleGetAssignedPatients(
  req: Request,
  res: Response,
  next: NextFunction
): Promise<void> {
  try {
    if (!req.user) throw new APIError('Unauthorized', 401);
    const patients = await getAssignedPatientsForClinician(req.user.id);
    res.json({ patients });
  } catch (err) {
    next(err);
  }
}
