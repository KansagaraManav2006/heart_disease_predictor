import { prisma } from '../../db/prisma';
import { APIError } from '../../middleware/errorHandler';

export async function grantAccessToClinician(patientUserId: string, clinicianEmail: string) {
  const patientProfile = await prisma.patientProfile.findUnique({
    where: { userId: patientUserId },
  });

  if (!patientProfile) {
    throw new APIError('Patient profile not found.', 404);
  }

  const clinicianUser = await prisma.user.findUnique({
    where: { email: clinicianEmail.trim().toLowerCase() },
  });

  if (!clinicianUser || clinicianUser.role !== 'CLINICIAN') {
    throw new APIError('Clinician with this email address was not found.', 404);
  }

  const existingGrant = await prisma.clinicianAccess.findUnique({
    where: {
      patientProfileId_clinicianId: {
        patientProfileId: patientProfile.id,
        clinicianId: clinicianUser.id,
      },
    },
  });

  if (existingGrant && !existingGrant.revokedAt) {
    throw new APIError('Access has already been granted to this clinician.', 409);
  }

  let grant;
  if (existingGrant) {
    grant = await prisma.clinicianAccess.update({
      where: { id: existingGrant.id },
      data: { revokedAt: null, grantedAt: new Date() },
    });
  } else {
    grant = await prisma.clinicianAccess.create({
      data: {
        patientProfileId: patientProfile.id,
        clinicianId: clinicianUser.id,
      },
    });
  }

  await prisma.auditEvent.create({
    data: {
      actorId: patientUserId,
      action: 'CLINICIAN_ACCESS_GRANTED',
      entityType: 'ClinicianAccess',
      entityId: grant.id,
      metadata: { clinicianEmail: clinicianUser.email },
    },
  });

  return grant;
}

export async function revokeClinicianAccess(patientUserId: string, grantId: string) {
  const patientProfile = await prisma.patientProfile.findUnique({
    where: { userId: patientUserId },
  });

  if (!patientProfile) {
    throw new APIError('Patient profile not found.', 404);
  }

  const grant = await prisma.clinicianAccess.findUnique({
    where: { id: grantId },
  });

  if (!grant || grant.patientProfileId !== patientProfile.id) {
    throw new APIError('Access grant not found.', 404);
  }

  const updated = await prisma.clinicianAccess.update({
    where: { id: grantId },
    data: { revokedAt: new Date() },
  });

  await prisma.auditEvent.create({
    data: {
      actorId: patientUserId,
      action: 'CLINICIAN_ACCESS_REVOKED',
      entityType: 'ClinicianAccess',
      entityId: grantId,
    },
  });

  return updated;
}

export async function getAssignedPatientsForClinician(clinicianUserId: string) {
  const grants = await prisma.clinicianAccess.findMany({
    where: {
      clinicianId: clinicianUserId,
      revokedAt: null,
    },
    include: {
      patientProfile: {
        include: {
          user: { select: { email: true, createdAt: true } },
          assessments: {
            take: 1,
            orderBy: { createdAt: 'desc' },
          },
        },
      },
    },
  });

  return grants.map((g) => ({
    grantId: g.id,
    grantedAt: g.grantedAt,
    patient: g.patientProfile,
  }));
}
