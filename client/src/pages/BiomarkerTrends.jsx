import React, { useState } from 'react';
import Surface from '../components/Surface';
import PageHeader from '../components/PageHeader';
import MetricTile from '../components/MetricTile';
import StatusBadge from '../components/StatusBadge';
import { Activity, Heart, Calendar, CheckCircle2 } from 'lucide-react';

const HISTORICAL_DATA = [
  { date: '2025-03-01', glucose: 135, hba1c: 7.2, systolic_bp: 138, cholesterol: 215, bmi: 28.4 },
  { date: '2025-06-15', glucose: 124, hba1c: 6.8, systolic_bp: 132, cholesterol: 198, bmi: 27.9 },
  { date: '2025-09-20', glucose: 118, hba1c: 6.5, systolic_bp: 126, cholesterol: 188, bmi: 27.3 },
  { date: '2026-01-10', glucose: 110, hba1c: 6.3, systolic_bp: 122, cholesterol: 178, bmi: 26.8 },
];

const BiomarkerTrends = () => {
  const [timeframe, setTimeframe] = useState('ALL');
  const [selectedBiomarker, setSelectedBiomarker] = useState('hba1c');

  const getBiomarkerConfig = (key) => {
    switch (key) {
      case 'hba1c':
        return { label: 'HbA1c Level', unit: '%', target: 'ADA Target < 7.0%', normalRange: '4.0% - 5.6%', current: '6.3%', initial: '7.2%' };
      case 'glucose':
        return { label: 'Fasting Glucose', unit: 'mg/dL', target: 'ADA Target < 100 mg/dL', normalRange: '70 - 99 mg/dL', current: '110 mg/dL', initial: '135 mg/dL' };
      case 'systolic_bp':
        return { label: 'Systolic BP', unit: 'mmHg', target: 'ACC/AHA Target < 120 mmHg', normalRange: '90 - 119 mmHg', current: '122 mmHg', initial: '138 mmHg' };
      case 'cholesterol':
        return { label: 'Serum Cholesterol', unit: 'mg/dL', target: 'NCEP Target < 200 mg/dL', normalRange: '125 - 199 mg/dL', current: '178 mg/dL', initial: '215 mg/dL' };
      case 'bmi':
        return { label: 'Body Mass Index', unit: 'kg/m²', target: 'Target 18.5 - 24.9', normalRange: '18.5 - 24.9', current: '26.8', initial: '28.4' };
      default:
        return { label: 'Biomarker', unit: '', target: '', normalRange: '', current: '', initial: '' };
    }
  };

  const currentConfig = getBiomarkerConfig(selectedBiomarker);

  const displayedData = timeframe === '3M'
    ? HISTORICAL_DATA.slice(-2)
    : timeframe === '6M'
    ? HISTORICAL_DATA.slice(-3)
    : HISTORICAL_DATA;

  return (
    <div className="space-y-8 animate-fade-in">
      <PageHeader
        title="Clinical Biomarker Trend &amp; Longitudinal Tracker"
        subtitle="Track historical trajectory against clinical guideline targets (ADA, ACC/AHA, NCEP) over time."
        badge={{ label: '4 Timepoints Logged', status: 'healthy' }}
      />

      {/* Top Summary Metrics */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricTile
          title="HbA1c Trajectory"
          value="6.3%"
          subtitle="Down from 7.2% (-0.9%)"
          icon={Activity}
          accent="teal"
        />
        <MetricTile
          title="Fasting Glucose"
          value="110 mg/dL"
          subtitle="Down from 135 mg/dL"
          icon={Activity}
          accent="teal"
        />
        <MetricTile
          title="Systolic BP"
          value="122 mmHg"
          subtitle="Down from 138 mmHg"
          icon={Heart}
          accent="amber"
        />
        <MetricTile
          title="Serum Cholesterol"
          value="178 mg/dL"
          subtitle="Target < 200 Achieved"
          icon={Heart}
          accent="teal"
        />
      </div>

      {/* Main Interactive Graph & Threshold View */}
      <Surface variant="flat" accent="teal" className="space-y-6">
        <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4 pb-4 border-b border-border">
          <div>
            <span className="text-xs font-semibold text-primary uppercase tracking-wider block mb-1">
              Longitudinal Clinical Trajectory
            </span>
            <h3 className="text-lg font-bold text-foreground">{currentConfig.label} Progress</h3>
          </div>

          <div className="flex flex-wrap items-center gap-3">
            {/* Timeframe selector */}
            <div className="flex gap-1 bg-muted p-1 rounded border border-border text-xs">
              {['ALL', '1Y', '6M', '3M'].map((tf) => (
                <button
                  key={tf}
                  onClick={() => setTimeframe(tf)}
                  className={`px-2.5 py-1 rounded font-semibold transition-all ${
                    timeframe === tf
                      ? 'bg-primary text-primary-foreground font-bold shadow-sm'
                      : 'text-muted-foreground hover:text-foreground'
                  }`}
                >
                  {tf}
                </button>
              ))}
            </div>

            {/* Biomarker Selector */}
            <div className="flex gap-1 bg-muted p-1 rounded border border-border text-xs">
              {[
                { id: 'hba1c', label: 'HbA1c' },
                { id: 'glucose', label: 'Glucose' },
                { id: 'systolic_bp', label: 'Systolic BP' },
                { id: 'cholesterol', label: 'Cholesterol' },
                { id: 'bmi', label: 'BMI' },
              ].map((b) => (
                <button
                  key={b.id}
                  onClick={() => setSelectedBiomarker(b.id)}
                  className={`px-3 py-1 rounded font-semibold transition-all ${
                    selectedBiomarker === b.id
                      ? 'bg-card text-primary shadow-sm border border-primary/40 font-bold'
                      : 'text-muted-foreground hover:text-foreground'
                  }`}
                >
                  {b.label}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Clinical Target Overlay Box */}
        <div className="bg-muted/80 p-4 rounded border border-border flex flex-col md:flex-row items-start md:items-center justify-between gap-4 text-xs">
          <div className="flex items-center gap-2">
            <CheckCircle2 className="w-4 h-4 text-emerald-400" />
            <span className="font-bold text-foreground">Clinical Benchmark Target:</span>
            <span className="text-primary font-mono font-semibold">{currentConfig.target}</span>
          </div>
          <div className="text-muted-foreground font-mono">
            Normal Reference Range: <strong className="text-foreground">{currentConfig.normalRange}</strong>
          </div>
        </div>

        {/* Timeline Visualization Table */}
        <div className="space-y-3 pt-2">
          <h4 className="text-xs font-bold text-muted-foreground uppercase tracking-wider">
            Historical Data Points ({displayedData.length} Records)
          </h4>

          <div className="space-y-2.5">
            {displayedData.map((row, idx) => {
              const val = row[selectedBiomarker];
              const isInitial = idx === 0;
              const isLatest = idx === displayedData.length - 1;

              return (
                <div
                  key={row.date}
                  className={`p-4 rounded border flex items-center justify-between transition-all ${
                    isLatest
                      ? 'bg-card border-primary/40 shadow-sm'
                      : 'bg-muted/40 border-border'
                  }`}
                >
                  <div className="flex items-center gap-3">
                    <Calendar className="w-4 h-4 text-muted-foreground" />
                    <div>
                      <span className="text-xs font-mono font-bold text-foreground">{row.date}</span>
                      {isLatest && <StatusBadge label="LATEST RECORD" status="healthy" size="sm" className="ml-2" />}
                      {isInitial && <span className="ml-2 text-[10px] text-muted-foreground font-mono">(Baseline)</span>}
                    </div>
                  </div>

                  <div className="flex items-center gap-4 font-mono tabular-nums">
                    <span className="text-base font-bold text-foreground">
                      {val} {currentConfig.unit}
                    </span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </Surface>
    </div>
  );
};

export default BiomarkerTrends;
