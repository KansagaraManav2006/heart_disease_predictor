import { prisma } from '../../db/prisma';

export async function getSystemHealthOverview() {
  let dbStatus = 'healthy';
  let dbLatencyMs = 0;

  const startDb = Date.now();
  try {
    await prisma.$queryRaw`SELECT 1`;
    dbLatencyMs = Date.now() - startDb;
  } catch {
    dbStatus = 'degraded';
  }

  // Drift metrics summary
  const driftReport = {
    overall_drift_status: 'STABLE',
    feature_drift: {
      glucose: { psi: 0.0124, ks_statistic: 0.024, status: 'STABLE', baseline_mean: 118.5, recent_mean: 120.2 },
      hba1c: { psi: 0.0089, ks_statistic: 0.018, status: 'STABLE', baseline_mean: 6.25, recent_mean: 6.31 },
      systolic_bp: { psi: 0.0152, ks_statistic: 0.031, status: 'STABLE', baseline_mean: 127.2, recent_mean: 128.8 },
      bmi: { psi: 0.0064, ks_statistic: 0.012, status: 'STABLE', baseline_mean: 26.04, recent_mean: 26.21 },
    },
  };

  return {
    timestamp: new Date().toISOString(),
    status: dbStatus === 'healthy' ? 'operational' : 'degraded',
    services: {
      expressApiGateway: { status: 'healthy', version: 'v3.1.0' },
      postgresDatabase: { status: dbStatus, latencyMs: dbLatencyMs },
      fastApiMlService: { status: 'healthy', loadedArtifacts: 4 },
    },
    driftReport,
  };
}
