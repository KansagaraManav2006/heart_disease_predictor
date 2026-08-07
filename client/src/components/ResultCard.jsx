import React, { useState } from 'react';
import Surface from './Surface';
import Button from './Button';
import RiskBadge from './RiskBadge';
import StatusBadge from './StatusBadge';
import { AlertTriangle, Info, TrendingUp, ShieldCheck, FileSpreadsheet, Eye, Activity, Sparkles } from 'lucide-react';

let _sessionReportCounter = 0;

const ResultCard = ({ prediction, probability, riskBand = 'LOW', explanation = null, extras = [], suggestions = [] }) => {
  const [viewMode, setViewMode] = useState('patient'); // 'patient' | 'clinician'
  const [downloading, setDownloading] = useState(false);

  const isHighRisk = prediction === 1 || riskBand === 'HIGH';
  const isModerateRisk = riskBand === 'MODERATE';
  const probPercent = (probability * 100).toFixed(1);

  const surfaceAccent = isHighRisk ? 'coral' : isModerateRisk ? 'amber' : 'teal';

  // Dynamic code-split PDF generator
  const generatePDF = async () => {
    try {
      setDownloading(true);
      const { jsPDF } = await import('jspdf');
      const autoTableModule = await import('jspdf-autotable');
      const autoTable = autoTableModule.default || autoTableModule;

      const doc = new jsPDF();
      const date = new Date().toLocaleDateString();
      const time = new Date().toLocaleTimeString();
      const patientName = extras.find((e) => e.label === 'Patient' || e.label === 'Patient Name')?.value || 'Anonymous';
      _sessionReportCounter += 1;
      const reportId = `SR-${Date.now()}-${String(_sessionReportCounter).padStart(3, '0')}`;

      // Header Ribbon
      doc.setFillColor(13, 148, 136); // Electric Teal
      doc.rect(0, 0, 210, 40, 'F');

      doc.setTextColor(255, 255, 255);
      doc.setFontSize(20);
      doc.setFont('helvetica', 'bold');
      doc.text('HealthLens AI — Calibrated Risk Summary', 14, 18);

      doc.setFontSize(9);
      doc.setFont('helvetica', 'normal');
      doc.text('Clinical Decision Support & Statistical Screening Report (Research Use Only)', 14, 28);
      doc.text(`Generated: ${date} ${time} | Report ID: ${reportId}`, 14, 35);

      doc.setDrawColor(200, 200, 200);
      doc.setFillColor(248, 250, 252);
      doc.rect(14, 48, 182, 28, 'FD');

      doc.setTextColor(50, 50, 50);
      doc.setFontSize(10);
      doc.setFont('helvetica', 'bold');
      doc.text('SUBJECT & MODEL METADATA', 18, 55);

      doc.setFont('helvetica', 'normal');
      doc.text(`Subject: ${patientName}`, 18, 64);
      doc.text(`Date: ${date}`, 100, 64);
      doc.text(`Time: ${time}`, 100, 70);

      // Risk Result
      doc.setFont('helvetica', 'bold');
      doc.setFontSize(13);
      doc.setTextColor(30, 41, 59);
      doc.text('Calibrated Risk Stratification', 14, 88);
      doc.setLineWidth(0.5);
      doc.line(14, 90, 196, 90);

      doc.setFontSize(11);
      doc.text(`Risk Stratum: ${riskBand} (Event Probability: ${probPercent}%)`, 14, 100);

      const summaryText = isHighRisk
        ? 'The predictive model identified metric patterns associated with an elevated risk profile. This is not a diagnostic result. Please consult a qualified healthcare professional.'
        : 'The parameters provided align with a lower-risk profile relative to population benchmarks. Continue healthy lifestyle habits and regular screening.';
      const splitText = doc.splitTextToSize(summaryText, 175);
      doc.text(splitText, 14, 110);

      // Biometric Inputs Table
      const tableBody = extras.map((item) => [item.label, item.value]);
      autoTable(doc, {
        startY: 125,
        head: [['Clinical Metric / Biomarker', 'Recorded Input']],
        body: tableBody,
        theme: 'striped',
        headStyles: { fillStyle: [13, 148, 136] },
      });

      // Footer Disclaimer
      const pageCount = doc.internal.getNumberOfPages();
      for (let i = 1; i <= pageCount; i++) {
        doc.setPage(i);
        doc.setFontSize(8);
        doc.setTextColor(150, 150, 150);
        doc.line(14, doc.internal.pageSize.height - 18, 196, doc.internal.pageSize.height - 18);
        doc.text('FOR RESEARCH USE ONLY. Not a medical diagnosis. Consult a qualified physician.', 14, doc.internal.pageSize.height - 10);
        doc.text(`Page ${i} of ${pageCount}`, 185, doc.internal.pageSize.height - 10);
      }

      doc.save(`${patientName.replace(/\s+/g, '_')}_Risk_Summary.pdf`);
    } catch (err) {
      console.error('Failed to generate PDF:', err);
    } finally {
      setDownloading(false);
    }
  };

  return (
    <Surface variant="glass" accent={surfaceAccent} className="mt-8 shadow-2xl animate-fade-in-up">
      {/* View Mode Toggle Header */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 pb-6 border-b border-slate-800">
        <div>
          <span className="text-[11px] font-bold text-teal-400 uppercase tracking-wider block mb-1">
            HealthLens AI Calibrated Risk Engine
          </span>
          <h2 className="text-xl md:text-2xl font-bold text-slate-100">Screening Assessment Result</h2>
        </div>

        <div className="bg-slate-950 p-1 rounded-xl flex text-xs font-semibold border border-slate-800">
          <button
            onClick={() => setViewMode('patient')}
            className={`flex items-center gap-1.5 px-3.5 py-1.5 rounded-lg transition-all ${
              viewMode === 'patient'
                ? 'bg-slate-850 text-teal-400 shadow-sm border border-teal-500/30 font-bold'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Eye className="w-3.5 h-3.5" /> Patient View
          </button>
          <button
            onClick={() => setViewMode('clinician')}
            className={`flex items-center gap-1.5 px-3.5 py-1.5 rounded-lg transition-all ${
              viewMode === 'clinician'
                ? 'bg-slate-850 text-teal-400 shadow-sm border border-teal-500/30 font-bold'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Activity className="w-3.5 h-3.5" /> SHAP Attribution
          </button>
        </div>
      </div>

      {/* Out-Of-Distribution Warning Banner */}
      {explanation?.out_of_distribution && (
        <div className="flex items-start gap-3 bg-amber-500/10 border border-amber-500/30 text-amber-300 p-4 rounded-xl my-6 text-xs">
          <AlertTriangle className="text-amber-400 flex-shrink-0 w-4 h-4 mt-0.5" />
          <div>
            <span className="font-bold block mb-1">Out-Of-Distribution Parameter Notice:</span>
            <ul className="list-disc list-inside space-y-0.5 text-amber-200/90">
              {explanation.ood_warnings?.map((w, idx) => (
                <li key={idx}>{w}</li>
              ))}
            </ul>
          </div>
        </div>
      )}

      {/* Primary Risk Stratification Grid */}
      <div className="grid md:grid-cols-2 gap-6 my-6">
        {/* Calibrated Probability Box */}
        <div className="bg-slate-900 p-6 rounded-2xl border border-slate-800 flex flex-col justify-between shadow-sm">
          <span className="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2">
            Calibrated Event Stratification
          </span>
          <div className="flex items-center gap-4 my-2">
            <RiskBadge riskBand={riskBand} score={probPercent} size="lg" />
          </div>
          <p className="text-xs text-slate-400 mt-3 leading-relaxed">
            Probability score calibrated via 5-fold cross-validation on validated population cohort data.
          </p>
        </div>

        {/* View Mode Context Box */}
        {viewMode === 'patient' ? (
          <div className="bg-slate-900 p-6 rounded-2xl border border-slate-800 flex flex-col justify-between shadow-sm">
            <span className="text-xs font-semibold uppercase tracking-wider text-teal-400 mb-2">
              Patient Guidance Summary
            </span>
            <p className="text-xs md:text-sm text-slate-200 leading-relaxed mb-3">
              {isHighRisk
                ? 'The predictive model identified biometric factors associated with an elevated risk profile. Please schedule an evaluation with your healthcare provider.'
                : 'Your entered parameters align with a lower risk category relative to benchmark population reference values.'}
            </p>
            <div className="text-xs text-slate-400 bg-slate-950 p-3 rounded-xl border border-slate-800">
              <strong>Notice:</strong> Statistical risk screening tool for clinical decision support. Not a standalone diagnosis.
            </div>
          </div>
        ) : (
          <div className="bg-slate-900 p-6 rounded-2xl border border-slate-800 font-mono text-xs text-slate-200 flex flex-col justify-between">
            <div className="space-y-2">
              <span className="text-teal-400 font-bold uppercase tracking-wider block font-sans">
                Clinician Attribution Metadata
              </span>
              <div>Model Architecture: HistGradientBoosting (Calibrated)</div>
              <div>Pipeline Version: {explanation?.condition || 'model'}-v3.0</div>
              <div>OOD Status: {explanation?.out_of_distribution ? 'TRUE (Exceeds Bounds)' : 'FALSE (In Bounds)'}</div>
            </div>
            <div className="text-slate-400 text-[11px] mt-4 font-sans">
              SHAP feature attributions calculate exact Shapley marginal contributions to calibrated log-odds.
            </div>
          </div>
        )}
      </div>

      {/* Amber Surface for AI Recommendations */}
      {suggestions && suggestions.length > 0 && (
        <div className="my-6 bg-amber-500/10 border border-amber-500/30 rounded-2xl p-5 text-xs text-amber-200">
          <div className="flex items-center gap-2 mb-3">
            <Sparkles className="w-4 h-4 text-amber-400" />
            <h4 className="font-bold text-amber-300 uppercase tracking-wider text-xs font-sans">
              AI Decision Support &amp; Clinical Recommendations
            </h4>
          </div>
          <ul className="space-y-2 text-slate-200 list-disc list-inside">
            {suggestions.map((sug, idx) => (
              <li key={idx} className="leading-relaxed">
                {sug}
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* SHAP Feature Attribution Breakdown */}
      {explanation && (
        <div className="my-6 pt-6 border-t border-slate-800">
          <h3 className="text-sm font-bold text-slate-100 mb-4 flex items-center gap-2">
            <TrendingUp className="w-4 h-4 text-teal-400" /> Feature Attribution Breakdown (SHAP Analysis)
          </h3>

          {viewMode === 'patient' ? (
            <div className="grid md:grid-cols-2 gap-6">
              {/* Primary Risk Drivers */}
              <div className="bg-coral-950/20 p-5 rounded-2xl border border-coral-500/30">
                <h4 className="text-xs font-bold text-coral-300 mb-3 flex items-center gap-2 uppercase tracking-wider">
                  <AlertTriangle className="w-4 h-4 text-coral-400" /> Primary Risk Drivers
                </h4>
                <ul className="space-y-2 text-xs">
                  {explanation.patient_explanation?.primary_risk_drivers?.map((driver, idx) => (
                    <li key={idx} className="bg-slate-900 p-2.5 rounded-xl border border-slate-800 text-slate-200 font-medium">
                      {driver}
                    </li>
                  ))}
                </ul>
              </div>

              {/* Favorable Factors */}
              <div className="bg-teal-950/20 p-5 rounded-2xl border border-teal-500/30">
                <h4 className="text-xs font-bold text-teal-300 mb-3 flex items-center gap-2 uppercase tracking-wider">
                  <ShieldCheck className="w-4 h-4 text-teal-400" /> Favorable Indicators
                </h4>
                <ul className="space-y-2 text-xs">
                  {explanation.patient_explanation?.favorable_factors?.map((factor, idx) => (
                    <li key={idx} className="bg-slate-900 p-2.5 rounded-xl border border-slate-800 text-slate-200 font-medium">
                      {factor}
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          ) : (
            <div className="overflow-x-auto rounded-2xl border border-slate-800">
              <table className="w-full text-left text-xs font-mono tabular-nums">
                <thead className="bg-slate-900 text-slate-400 border-b border-slate-800">
                  <tr>
                    <th className="p-3">Feature Name</th>
                    <th className="p-3">Recorded Input</th>
                    <th className="p-3">SHAP Weight</th>
                    <th className="p-3">Attribution Direction</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800 bg-slate-950/40">
                  {explanation.top_risk_contributors?.concat(explanation.top_protective_factors || []).map((item, idx) => (
                    <tr key={idx} className="hover:bg-slate-900/60">
                      <td className="p-3 font-semibold text-slate-100 font-sans">{item.feature}</td>
                      <td className="p-3 text-slate-300">{item.raw_value}</td>
                      <td className={`p-3 font-bold ${item.is_risk_factor ? 'text-coral-400' : 'text-teal-400'}`}>
                        {item.shap_attribution > 0 ? `+${item.shap_attribution}` : item.shap_attribution}
                      </td>
                      <td className="p-3">
                        <StatusBadge
                          status={item.is_risk_factor ? 'high_risk' : 'healthy'}
                          label={item.is_risk_factor ? 'ELEVATES RISK' : 'PROTECTIVE'}
                          size="sm"
                        />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}

      {/* Model Limitations Box */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 my-6 text-xs text-slate-400">
        <span className="font-bold text-slate-200 flex items-center gap-1.5 mb-1">
          <Info className="w-4 h-4 text-teal-400" /> Model Scope &amp; Boundaries:
        </span>
        <ul className="list-disc list-inside space-y-0.5">
          {explanation?.limitations?.map((lim, idx) => (
            <li key={idx}>{lim}</li>
          )) || (
            <li>Statistical risk estimate for decision support. Consult a physician for diagnosis.</li>
          )}
        </ul>
      </div>

      {/* Download Action */}
      <div className="flex justify-end pt-4 border-t border-slate-800">
        <Button
          onClick={generatePDF}
          loading={downloading}
          loadingLabel="Generating Report PDF..."
          variant="primary"
          icon={FileSpreadsheet}
          className="font-bold"
        >
          Download assessment report (PDF)
        </Button>
      </div>
    </Surface>
  );
};

export default ResultCard;
