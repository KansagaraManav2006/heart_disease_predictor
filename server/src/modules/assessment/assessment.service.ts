import { ConditionType, RiskBand } from '@prisma/client';
import { prisma } from '../../db/prisma';
import { APIError } from '../../middleware/errorHandler';

export async function createAssessmentRecord(input: {
  creatorId: string;
  condition: ConditionType;
  inputPayload: any;
  modelVersion: string;
  probability: number;
  riskBand: RiskBand;
  outOfDistribution?: boolean;
  observations?: Array<{ name: string; value: number; unit: string; source?: string }>;
}) {
  // Check if patient profile exists for creator
  const patientProfile = await prisma.patientProfile.findUnique({
    where: { userId: input.creatorId },
  });

  const assessment = await prisma.assessment.create({
    data: {
      creatorId: input.creatorId,
      patientProfileId: patientProfile?.id || null,
      condition: input.condition,
      inputPayload: input.inputPayload,
      modelVersion: input.modelVersion,
      probability: input.probability,
      riskBand: input.riskBand,
      outOfDistribution: input.outOfDistribution || false,
      ...(input.observations && input.observations.length > 0 && {
        observations: {
          create: input.observations.map((obs) => ({
            name: obs.name,
            value: obs.value,
            unit: obs.unit,
            source: obs.source || 'MANUAL',
          })),
        },
      }),
    },
    include: {
      observations: true,
    },
  });

  await prisma.auditEvent.create({
    data: {
      actorId: input.creatorId,
      action: 'ASSESSMENT_CREATED',
      entityType: 'Assessment',
      entityId: assessment.id,
      metadata: { condition: input.condition, riskBand: input.riskBand },
    },
  });

  return assessment;
}

export async function getUserAssessments(userId: string, condition?: ConditionType) {
  const patientProfile = await prisma.patientProfile.findUnique({
    where: { userId },
  });

  return prisma.assessment.findMany({
    where: {
      OR: [
        { creatorId: userId },
        ...(patientProfile ? [{ patientProfileId: patientProfile.id }] : []),
      ],
      ...(condition && { condition }),
    },
    orderBy: { createdAt: 'desc' },
    include: {
      observations: true,
      report: true,
    },
  });
}

export async function getAssessmentById(assessmentId: string, requestingUser: { id: string; role: string }) {
  const assessment = await prisma.assessment.findUnique({
    where: { id: assessmentId },
    include: {
      observations: true,
      report: true,
      patientProfile: {
        include: { user: { select: { email: true } } },
      },
    },
  });

  if (!assessment) {
    throw new APIError('Assessment not found.', 404);
  }

  // Authorization check
  if (requestingUser.role === 'ADMIN' || assessment.creatorId === requestingUser.id) {
    return assessment;
  }

  if (requestingUser.role === 'CLINICIAN' && assessment.patientProfileId) {
    const grant = await prisma.clinicianAccess.findUnique({
      where: {
        patientProfileId_clinicianId: {
          patientProfileId: assessment.patientProfileId,
          clinicianId: requestingUser.id,
        },
      },
    });

    if (grant && !grant.revokedAt) {
      return assessment;
    }
  }

  throw new APIError('Access denied. You do not have permission to view this assessment.', 403);
}
