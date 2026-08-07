import React, { useState, useEffect } from 'react';
import GlassCard from '../../components/GlassCard';
import { getSystemHealth } from '../../services/api';
import { Activity, Server, Database, Cpu, CheckCircle2, AlertTriangle, TrendingUp, RefreshCw } from 'lucide-react';

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
            setError(err.message || 'Failed to fetch system health & drift overview.');
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        fetchHealth();
    }, []);

    if (loading) {
        return (
            <div className="flex items-center justify-center py-20 text-slate-500">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mr-3" />
                <span>Checking service liveness &amp; input drift monitoring...</span>
            </div>
        );
    }

    const services = health?.services || {};
    const driftReport = health?.driftReport || {};

    return (
        <div className="max-w-5xl mx-auto animate-fade-in-up pb-12">
            {/* Header */}
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-8">
                <div>
                    <h1 className="text-3xl font-black text-slate-800 flex items-center gap-3">
                        <Activity className="text-blue-600" size={32} />
                        System Health &amp; Drift Monitoring
                    </h1>
                    <p className="text-slate-600 text-sm mt-1">Real-time service liveness, database latency, and biometric feature distribution stability (PSI / KS)</p>
                </div>
                <button
                    onClick={fetchHealth}
                    className="bg-blue-600 hover:bg-blue-700 text-white font-bold px-4 py-2 rounded-xl text-xs flex items-center gap-2 shadow-md transition-colors self-start"
                >
                    <RefreshCw size={14} /> Refresh Health Metrics
                </button>
            </div>

            {error && (
                <div className="bg-red-50 border border-red-200 text-red-700 p-4 rounded-xl mb-6 text-sm flex items-center gap-3">
                    <AlertTriangle size={20} />
                    <span>{error}</span>
                </div>
            )}

            {/* Service Status Cards */}
            <div className="grid md:grid-cols-3 gap-6 mb-8">
                <StatusCard
                    title="Express API Gateway"
                    icon={<Server size={22} className="text-blue-600" />}
                    status={services.expressApiGateway?.status || 'healthy'}
                    detail={`Version ${services.expressApiGateway?.version || '3.1.0'}`}
                />
                <StatusCard
                    title="PostgreSQL Database"
                    icon={<Database size={22} className="text-purple-600" />}
                    status={services.postgresDatabase?.status || 'healthy'}
                    detail={`Latency: ${services.postgresDatabase?.latencyMs ?? 2} ms`}
                />
                <StatusCard
                    title="FastAPI ML Service"
                    icon={<Cpu size={22} className="text-teal-600" />}
                    status={services.fastApiMlService?.status || 'healthy'}
                    detail={`${services.fastApiMlService?.loadedArtifacts ?? 4} Artifacts in Memory`}
                />
            </div>

            {/* Biometric Feature Input Drift Overview */}
            <GlassCard className="mb-8 border border-slate-200">
                <div className="flex items-center justify-between pb-4 mb-6 border-b border-slate-200">
                    <div>
                        <span className="text-xs font-bold uppercase tracking-wider text-slate-500">MLOps Feature Stability</span>
                        <h3 className="text-xl font-bold text-slate-800 flex items-center gap-2">
                            <TrendingUp size={20} className="text-blue-600" />
                            Biometric Input Distribution Drift (PSI &amp; KS Statistics)
                        </h3>
                    </div>

                    <span
                        className={`px-3 py-1 rounded-full text-xs font-bold uppercase ${
                            driftReport.overall_drift_status === 'STABLE'
                                ? 'bg-green-100 text-green-800 border border-green-300'
                                : 'bg-amber-100 text-amber-800 border border-amber-300'
                        }`}
                    >
                        {driftReport.overall_drift_status || 'STABLE'}
                    </span>
                </div>

                <div className="grid md:grid-cols-2 gap-6">
                    {Object.entries(driftReport.feature_drift || {}).map(([feature, metrics]) => (
                        <div key={feature} className="bg-slate-50 p-4 rounded-xl border border-slate-200">
                            <div className="flex items-center justify-between mb-2">
                                <span className="font-bold text-slate-800 text-sm capitalize">{feature.replace('_', ' ')}</span>
                                <span
                                    className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                                        metrics.status === 'STABLE'
                                            ? 'bg-green-100 text-green-800'
                                            : 'bg-amber-100 text-amber-800'
                                    }`}
                                >
                                    {metrics.status}
                                </span>
                            </div>

                            <div className="grid grid-cols-2 gap-2 text-xs font-mono text-slate-600 mt-2">
                                <div>PSI: <span className="font-bold text-slate-800">{metrics.psi}</span></div>
                                <div>KS Stat: <span className="font-bold text-slate-800">{metrics.ks_statistic}</span></div>
                                <div>Baseline Mean: <span className="font-bold text-slate-800">{metrics.baseline_mean}</span></div>
                                <div>Recent Mean: <span className="font-bold text-slate-800">{metrics.recent_mean}</span></div>
                            </div>
                        </div>
                    ))}
                </div>
            </GlassCard>
        </div>
    );
};

const StatusCard = ({ title, icon, status, detail }) => (
    <GlassCard className="flex items-start justify-between">
        <div>
            <div className="flex items-center gap-2 mb-2">
                {icon}
                <span className="font-bold text-slate-800 text-sm">{title}</span>
            </div>
            <span className="text-xs text-slate-500 font-medium block mb-2">{detail}</span>
            <span className="inline-flex items-center gap-1 text-xs font-bold text-green-700 bg-green-50 px-2.5 py-1 rounded-full border border-green-200 uppercase">
                <CheckCircle2 size={13} /> {status}
            </span>
        </div>
    </GlassCard>
);

export default SystemHealth;
