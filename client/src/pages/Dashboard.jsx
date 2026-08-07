import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { getHistory } from '../services/api';
import Surface from '../components/Surface';
import PageHeader from '../components/PageHeader';
import MetricTile from '../components/MetricTile';
import RiskBadge from '../components/RiskBadge';
import EmptyState from '../components/EmptyState';
import ErrorState from '../components/ErrorState';
import { DashboardSkeleton } from '../components/Skeleton';
import Button from '../components/Button';
import { Activity, Heart, Clock, LineChart, Plus } from 'lucide-react';

const Dashboard = () => {
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const navigate = useNavigate();

  useEffect(() => {
    const fetchHistory = async () => {
      try {
        const data = await getHistory();
        const sorted = data.sort((a, b) => new Date(b.date) - new Date(a.date));
        setHistory(sorted);
      } catch (err) {
        console.error('Failed to load history:', err);
        setError('Failed to load patient assessment history.');
      } finally {
        setLoading(false);
      }
    };

    fetchHistory();
  }, []);

  let userId = localStorage.getItem('userId');
  if (!userId) {
    userId = 'user_' + Date.now();
    localStorage.setItem('userId', userId);
  }

  const userHistory = history.filter((item) => item.userId === userId);
  const diabetesHistory = userHistory.filter((item) => item.type === 'diabetes');
  const heartHistory = userHistory.filter((item) => item.type === 'heart');

  const latestDiabetes = diabetesHistory[0];
  const latestHeart = heartHistory[0];

  const renderMetricTrend = (list, metricKey, label, unit = '') => {
    if (list.length < 2) return null;
    const current = list[0]?.inputs?.[metricKey];
    const previous = list[1]?.inputs?.[metricKey];

    if (current === undefined || previous === undefined) return null;

    const diff = Number(current) - Number(previous);
    const isImprovement = diff <= 0;
    const trendLabel = `${diff > 0 ? '+' : ''}${diff.toFixed(1)}${unit} (${isImprovement ? 'improved' : 'elevated'})`;

    return (
      <div className="flex items-center justify-between p-3 bg-slate-950 rounded-xl border border-slate-800 text-xs">
        <span className="font-medium text-slate-300">{label}</span>
        <div className="flex items-center gap-2 font-mono tabular-nums">
          <span className="text-slate-400">{previous}{unit}</span>
          <span className="text-slate-600 font-bold">→</span>
          <span className="text-slate-100 font-bold">{current}{unit}</span>
          <span
            className={`px-2 py-0.5 rounded-full font-sans font-semibold text-[10px] ${
              isImprovement
                ? 'bg-teal-500/20 text-teal-300 border border-teal-500/30'
                : 'bg-coral-500/20 text-coral-300 border border-coral-500/30'
            }`}
          >
            {trendLabel}
          </span>
        </div>
      </div>
    );
  };

  if (loading) {
    return (
      <div className="space-y-8 animate-fade-in">
        <PageHeader title="Patient Dashboard" subtitle="Loading clinical biometric history..." />
        <DashboardSkeleton />
      </div>
    );
  }

  return (
    <div className="space-y-8 animate-fade-in">
      <PageHeader
        title="Patient Dashboard"
        subtitle="Track historical health metrics, calibrated risk trends, and biometric changes."
        action={
          <div className="flex gap-2">
            <Button onClick={() => navigate('/diabetes')} variant="primary" size="sm" icon={Plus}>
              New Diabetes Scan
            </Button>
            <Button onClick={() => navigate('/heart')} variant="secondary" size="sm" icon={Plus}>
              New Cardiac Scan
            </Button>
          </div>
        }
      />

      {error && <ErrorState title="Data Load Failure" message={error} />}

      {/* Compact Top Metric Tiles (Squircle Icons + Tabular Numerals) */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricTile
          title="Total Assessments"
          value={userHistory.length}
          subtitle="Recorded in session"
          icon={LineChart}
          accent="teal"
        />
        <MetricTile
          title="Latest Glycemic Risk"
          value={latestDiabetes ? `${Math.round(latestDiabetes.probability * 100)}%` : 'N/A'}
          subtitle={latestDiabetes ? (latestDiabetes.prediction === 1 ? 'Elevated Stratum' : 'Lower Stratum') : 'No Glycemic Scan'}
          icon={Activity}
          accent={latestDiabetes?.prediction === 1 ? 'coral' : 'teal'}
        />
        <MetricTile
          title="Latest Cardiac Risk"
          value={latestHeart ? `${Math.round(latestHeart.probability * 100)}%` : 'N/A'}
          subtitle={latestHeart ? (latestHeart.prediction === 1 ? 'Elevated Stratum' : 'Lower Stratum') : 'No Cardiac Scan'}
          icon={Heart}
          accent={latestHeart?.prediction === 1 ? 'coral' : 'amber'}
        />
        <MetricTile
          title="Biometric Tracking"
          value={userHistory.length > 0 ? 'Active' : 'Pending'}
          subtitle="Session memory persistent"
          icon={Clock}
          accent="none"
        />
      </div>

      {/* No-history overall empty state */}
      {userHistory.length === 0 && (
        <EmptyState
          icon={LineChart}
          title="No Patient History Recorded"
          description="Initiate your first Diabetes or Cardiac risk assessment to begin tracking calibrated metrics."
          actionLabel="Run First Scan"
          onAction={() => navigate('/diabetes')}
          actionIcon={Plus}
        />
      )}

      {/* History Grid */}
      {userHistory.length > 0 && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Diabetes Tracking Panel */}
          <Surface variant="flat" accent="teal" className="flex flex-col justify-between">
            <div>
              <div className="flex items-center gap-3 pb-4 mb-6 border-b border-slate-800">
                <div className="w-10 h-10 rounded-xl bg-teal-500/20 text-teal-400 border border-teal-500/30 flex items-center justify-center flex-shrink-0 shadow-inner">
                  <Activity className="w-5 h-5" />
                </div>
                <div>
                  <h2 className="text-base font-bold text-slate-100">Diabetes Risk Tracking</h2>
                  <p className="text-xs text-slate-400">Glycemic biomarkers &amp; glucose trends</p>
                </div>
              </div>

              {diabetesHistory.length >= 2 ? (
                <div className="mb-6 space-y-2">
                  <h3 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-2">
                    Biometric Metric Changes
                  </h3>
                  {renderMetricTrend(diabetesHistory, 'glucose', 'Fasting Glucose', ' mg/dL')}
                  {renderMetricTrend(diabetesHistory, 'hba1c', 'HbA1c Level', '%')}
                  {renderMetricTrend(diabetesHistory, 'bmi', 'Body Mass Index', ' kg/m²')}
                </div>
              ) : (
                diabetesHistory.length === 1 && (
                  <p className="text-xs text-slate-400 mb-6 italic">
                    Single assessment recorded. Complete a second scan to unlock metric comparison trends.
                  </p>
                )
              )}

              <div>
                <h3 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-3">Assessment Logs</h3>
                {diabetesHistory.length > 0 ? (
                  <div className="space-y-2.5 max-h-64 overflow-y-auto pr-1">
                    {diabetesHistory.map((item, idx) => (
                      <div
                        key={idx}
                        className="p-3 bg-slate-950 rounded-xl border border-slate-800 flex items-center justify-between"
                      >
                        <div className="flex items-center gap-2.5">
                          <Clock className="w-4 h-4 text-slate-500" />
                          <span className="text-xs font-medium text-slate-300">
                            {new Date(item.date).toLocaleDateString()}
                          </span>
                        </div>
                        <RiskBadge
                          riskBand={item.prediction === 1 ? 'HIGH' : 'LOW'}
                          score={Math.round(item.probability * 100)}
                          size="sm"
                        />
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="text-xs text-slate-500 italic">No diabetes scans recorded yet.</p>
                )}
              </div>
            </div>
          </Surface>

          {/* Cardiac Tracking Panel */}
          <Surface variant="flat" accent="amber" className="flex flex-col justify-between">
            <div>
              <div className="flex items-center gap-3 pb-4 mb-6 border-b border-slate-800">
                <div className="w-10 h-10 rounded-xl bg-amber-500/20 text-amber-400 border border-amber-500/30 flex items-center justify-center flex-shrink-0 shadow-inner">
                  <Heart className="w-5 h-5" />
                </div>
                <div>
                  <h2 className="text-base font-bold text-slate-100">Cardiac Risk Tracking</h2>
                  <p className="text-xs text-slate-400">Cardiovascular vitals &amp; lipid trends</p>
                </div>
              </div>

              {heartHistory.length >= 2 ? (
                <div className="mb-6 space-y-2">
                  <h3 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-2">
                    Biometric Metric Changes
                  </h3>
                  {renderMetricTrend(heartHistory, 'systolic_bp', 'Systolic BP', ' mmHg')}
                  {renderMetricTrend(heartHistory, 'cholesterol', 'Serum Cholesterol', ' mg/dL')}
                </div>
              ) : (
                heartHistory.length === 1 && (
                  <p className="text-xs text-slate-400 mb-6 italic">
                    Single assessment recorded. Complete a second scan to unlock metric comparison trends.
                  </p>
                )
              )}

              <div>
                <h3 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-3">Assessment Logs</h3>
                {heartHistory.length > 0 ? (
                  <div className="space-y-2.5 max-h-64 overflow-y-auto pr-1">
                    {heartHistory.map((item, idx) => (
                      <div
                        key={idx}
                        className="p-3 bg-slate-950 rounded-xl border border-slate-800 flex items-center justify-between"
                      >
                        <div className="flex items-center gap-2.5">
                          <Clock className="w-4 h-4 text-slate-500" />
                          <span className="text-xs font-medium text-slate-300">
                            {new Date(item.date).toLocaleDateString()}
                          </span>
                        </div>
                        <RiskBadge
                          riskBand={item.prediction === 1 ? 'HIGH' : 'LOW'}
                          score={Math.round(item.probability * 100)}
                          size="sm"
                        />
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="text-xs text-slate-500 italic">No cardiac scans recorded yet.</p>
                )}
              </div>
            </div>
          </Surface>
        </div>
      )}
    </div>
  );
};

export default Dashboard;
