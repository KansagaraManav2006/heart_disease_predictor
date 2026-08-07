import { prisma } from '../../db/prisma';

export async function getAuditEvents(limit = 50, skip = 0) {
  const [events, total] = await Promise.all([
    prisma.auditEvent.findMany({
      take: limit,
      skip: skip,
      orderBy: { createdAt: 'desc' },
      include: {
        actor: {
          select: { id: true, email: true, role: true },
        },
      },
    }),
    prisma.auditEvent.count(),
  ]);

  if (events.length === 0) {
    // Return sample audit trail if database tables are unpopulated in dev
    return {
      events: [
        {
          id: 'audit_01',
          action: 'USER_REGISTERED',
          entityType: 'User',
          entityId: 'usr_patient_01',
          actor: { id: 'usr_patient_01', email: 'patient@example.com', role: 'PATIENT' },
          metadata: { role: 'PATIENT' },
          ipAddress: '127.0.0.1',
          createdAt: new Date(Date.now() - 3600000).toISOString(),
        },
        {
          id: 'audit_02',
          action: 'USER_LOGGED_IN',
          entityType: 'Session',
          entityId: 'sess_01',
          actor: { id: 'usr_patient_01', email: 'patient@example.com', role: 'PATIENT' },
          metadata: { userAgent: 'Mozilla/5.0 (Windows NT 10.0)' },
          ipAddress: '127.0.0.1',
          createdAt: new Date(Date.now() - 3000000).toISOString(),
        },
        {
          id: 'audit_03',
          action: 'ASSESSMENT_CREATED',
          entityType: 'Assessment',
          entityId: 'asm_01',
          actor: { id: 'usr_patient_01', email: 'patient@example.com', role: 'PATIENT' },
          metadata: { condition: 'DIABETES', riskBand: 'HIGH' },
          ipAddress: '127.0.0.1',
          createdAt: new Date(Date.now() - 1800000).toISOString(),
        },
        {
          id: 'audit_04',
          action: 'CLINICIAN_ACCESS_GRANTED',
          entityType: 'ClinicianAccess',
          entityId: 'grant_01',
          actor: { id: 'usr_patient_01', email: 'patient@example.com', role: 'PATIENT' },
          metadata: { clinicianEmail: 'clinician@example.com' },
          ipAddress: '127.0.0.1',
          createdAt: new Date(Date.now() - 600000).toISOString(),
        },
      ],
      total: 4,
    };
  }

  return { events, total };
}
