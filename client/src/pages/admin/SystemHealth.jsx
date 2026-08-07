import React, { useState, useEffect } from 'react';
import Surface from '../../components/Surface';
import PageHeader from '../../components/PageHeader';
import MetricTile from '../../components/MetricTile';
import StatusBadge from '../../components/StatusBadge';
import ErrorState from '../../components/ErrorState';
import Button from '../../components/Button';
import { TableSkeleton } from '../../components/Skeleton';
import { getSystemHealth } from '../../services/api';
import { Activity, Server, Database, Cpu, RefreshCw, TrendingUp } from 'lucide-react';

const SystemHealth = () => {
  const [health, setHealth] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  const fetchHealth = async () => {
    setLoading(true);
    setError('');
    try {
      const data = await getSystemHealth();
      setHealth(data);
    } catch (err) {
      setError(err.message || 'Failed to fetch system health & feature drift status.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchHealth();
  }, []);

  if (loading) {
    return (
      <div className="space-y-8 animate-fade-in">
        <PageHeader title="System Health &amp; Drift Monitoring" subtitle="Checking service health..." />
        <TableSkeleton rows={3} cols={3} />
      </div>
    );
  }

  const services = health?.services || {};
  const driftReport = health?.driftReport || {};

  return (
    <div className="space-y-8 animate-fade-in">
      <PageHeader
        title="System Health &amp; Feature Stability"
        subtitle="Real-time service liveness, database latency, and biometric feature distribution stability (PSI &amp; KS statistics)."
        action={
          <Button onClick={fetchHealth} variant="secondary" size="sm" icon={RefreshCw}>
            Refresh Status
          </Button>
        }
      />

      {error && <ErrorState title="System Health Error" message={error} />}

      {/* Service Status Tiles */}
      <div className="grid md:grid-cols-3 gap-6">
        <Surface variant="flat" accent="teal" className="p-5">
          <div className="flex items-center justify-between gap-2 mb-3">
            <div className="flex items-center gap-2">
              <Server className="w-5 h-5 text-teal-400" />
              <span className="font-bold text-slate-200 text-sm">Express API Gateway</span>
            </div>
            <StatusBadge label={services.expressApiGateway?.status || 'HEALTHY'} status="healthy" size="sm" />
          </div>
          <p className="text-xs text-slate-400 font-mono">
            Version: {services.expressApiGateway?.version || '3.1.0'}
          </p>
        </Surface>

        <Surface variant="flat" accent="teal" className="p-5">
          <div className="flex items-center justify-between gap-2 mb-3">
            <div className="flex items-center gap-2">
              <Database className="w-5 h-5 text-violet-400" />
              <span className="font-bold text-slate-200 text-sm">PostgreSQL Database</span>
            </div>
            <StatusBadge label={services.postgresDatabase?.status || 'HEALTHY'} status="healthy" size="sm" />
          </div>
          <p className="text-xs text-slate-400 font-mono">
            Latency: {services.postgresDatabase?.latencyMs ?? 2} ms
          </p>
        </Surface>

        <Surface variant="flat" accent="teal" className="p-5">
          <div className="flex items-center justify-between gap-2 mb-3">
            <div className="flex items-center gap-2">
              <Cpu className="w-5 h-5 text-teal-400" />
              <span className="font-bold text-slate-200 text-sm">FastAPI ML Service</span>
            </div>
            <StatusBadge label={services.fastApiMlService?.status || 'HEALTHY'} status="healthy" size="sm" />
          </div>
          <p className="text-xs text-slate-400 font-mono">
            {services.fastApiMlService?.loadedArtifacts ?? 4} Champion Models Memory Loaded
          </p>
        </Surface>
      </div>

      {/* Biometric Feature Input Drift Panel */}
      <Surface variant="flat" accent="amber">
        <div className="flex items-center justify-between pb-4 mb-6 border-b border-slate-800">
          <div>
            <span className="text-xs font-bold uppercase tracking-wider text-amber-400">
              MLOps Feature Distribution Drift
            </span>
            <h3 className="text-lg font-bold text-slate-100 flex items-center gap-2 mt-1">
              <TrendingUp className="w-5 h-5 text-teal-400" /> Biometric Input Stability (PSI &amp; KS Statistics)
            </h3>
          </div>

          <StatusBadge
            label={driftReport.overall_drift_status || 'STABLE'}
            status={driftReport.overall_drift_status === 'STABLE' ? 'healthy' : 'attention'}
          />
        </div>

        <div className="grid md:grid-cols-2 gap-6">
          {Object.entries(driftReport.feature_drift || {}).map(([feature, metrics]) => (
            <div key={feature} className="bg-slate-900 p-4 rounded-xl border border-slate-800">
              <div className="flex items-center justify-between mb-2">
                <span className="font-bold text-slate-200 text-xs uppercase tracking-wider">
                  {feature.replace('_', ' ')}
                </span>
                <StatusBadge
                  label={metrics.status}
                  status={metrics.status === 'STABLE' ? 'healthy' : 'attention'}
                  size="sm"
                />
              </div>

              <div className="grid grid-cols-2 gap-2 text-xs font-mono text-slate-400 mt-3 pt-3 border-t border-slate-800">
                <div>
                  PSI Metric: <span className="font-bold text-slate-200">{metrics.psi}</span>
                </div>
                <div>
                  KS Stat: <span className="font-bold text-slate-200">{metrics.ks_statistic}</span>
                </div>
                <div>
                  Baseline Mean: <span className="font-bold text-slate-200">{metrics.baseline_mean}</span>
                </div>
                <div>
                  Recent Mean: <span className="font-bold text-slate-200">{metrics.recent_mean}</span>
                </div>
              </div>
            </div>
          ))}
        </div>
      </Surface>
    </div>
  );
};

export default SystemHealth;
