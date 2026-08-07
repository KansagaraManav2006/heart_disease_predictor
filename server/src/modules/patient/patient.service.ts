import { prisma } from '../../db/prisma';
import { APIError } from '../../middleware/errorHandler';

export async function getPatientProfileByUserId(userId: string) {
  const profile = await prisma.patientProfile.findUnique({
    where: { userId },
    include: {
      user: {
        select: { email: true, role: true, createdAt: true },
      },
      accesses: {
        where: { revokedAt: null },
        include: {
          clinicianUser: {
            select: { id: true, email: true },
          },
        },
      },
    },
  });

  if (!profile) {
    throw new APIError('Patient profile not found.', 404);
  }

  return profile;
}

export async function updatePatientProfile(
  userId: string,
  input: {
    fullName?: string;
    dateOfBirth?: string;
    biologicalSex?: string;
    emergencyPhone?: string;
  }
) {
  const profile = await prisma.patientProfile.findUnique({
    where: { userId },
  });

  if (!profile) {
    throw new APIError('Patient profile not found.', 404);
  }

  const updated = await prisma.patientProfile.update({
    where: { userId },
    data: {
      ...(input.fullName && { fullName: input.fullName.trim() }),
      ...(input.dateOfBirth && { dateOfBirth: new Date(input.dateOfBirth) }),
      ...(input.biologicalSex && { biologicalSex: input.biologicalSex }),
      ...(input.emergencyPhone && { emergencyPhone: input.emergencyPhone.trim() }),
    },
  });

  await prisma.auditEvent.create({
    data: {
      actorId: userId,
      action: 'PATIENT_PROFILE_UPDATED',
      entityType: 'PatientProfile',
      entityId: updated.id,
    },
  });

  return updated;
}
