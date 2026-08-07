import React, { useState } from 'react';
import GlassCard from './GlassCard';
import Button from './Button';
import { jsPDF } from 'jspdf';
import autoTable from 'jspdf-autotable';
import { AlertTriangle, Info, TrendingUp, ShieldCheck, FileSpreadsheet, Eye, Activity } from 'lucide-react';

let _sessionReportCounter = 0;

const ResultCard = ({ prediction, probability, riskBand = 'LOW', explanation = null, extras = [], suggestions = [] }) => {
    const [viewMode, setViewMode] = useState('patient'); // 'patient' | 'clinician'
    const isHighRisk = prediction === 1 || riskBand === 'HIGH';
    const probPercent = (probability * 100).toFixed(1);

    const generatePDF = () => {
        const doc = new jsPDF();
        const date = new Date().toLocaleDateString();
        const time = new Date().toLocaleTimeString();
        const patientName = extras.find(e => e.label === 'Patient' || e.label === 'Patient Name')?.value || 'Anonymous';
        _sessionReportCounter += 1;
        const reportId = `SR-${Date.now()}-${String(_sessionReportCounter).padStart(3, '0')}`;

        // 1. Header Ribbon
        doc.setFillColor(30, 136, 229);
        doc.rect(0, 0, 210, 40, 'F');

        doc.setTextColor(255, 255, 255);
        doc.setFontSize(22);
        doc.setFont("helvetica", "bold");
        doc.text('HealthLens AI — Risk Intelligence Summary', 14, 18);

        doc.setFontSize(10);
        doc.setFont("helvetica", "normal");
        doc.text('Research Risk Decision Support (Research Use Only) — Not a clinical diagnosis', 14, 28);
        doc.text(`Generated: ${date} ${time}`, 14, 35);

        doc.setDrawColor(200, 200, 200);
        doc.setFillColor(248, 250, 252);
        doc.rect(14, 48, 182, 30, 'FD');

        doc.setTextColor(50, 50, 50);
        doc.setFontSize(10);
        doc.setFont("helvetica", "bold");
        doc.text('SUBJECT & MODEL METADATA', 18, 55);

        doc.setFont("helvetica", "normal");
        doc.text(`Name: ${patientName}`, 18, 64);
        doc.text(`Date: ${date}`, 100, 64);
        doc.text(`Time: ${time}`, 100, 72);
        doc.text(`Report ID: ${reportId}`, 18, 72);

        // 3. Risk Result
        doc.setFont("helvetica", "bold");
        doc.setFontSize(14);
        doc.setTextColor(30, 41, 59);
        doc.text('Calibrated Risk Stratification', 14, 90);
        doc.setLineWidth(0.5);
        doc.line(14, 92, 196, 92);

        doc.setFontSize(12);
        doc.text(`Risk Band: ${riskBand} (Calibrated Event Likelihood: ${probPercent}%)`, 14, 102);

        const summaryText = isHighRisk
            ? "The screening model identified metrics associated with an elevated risk profile. This is not a diagnosis. Please consult a qualified healthcare professional for a clinical evaluation."
            : "The parameters provided are consistent with a lower-risk profile. Continue healthy lifestyle practices and attend regular check-ups.";
        const splitText = doc.splitTextToSize(summaryText, 175);
        doc.text(splitText, 14, 112);

        // 4. Biometric Inputs Table
        const tableBody = extras.map(item => [item.label, item.value]);
        autoTable(doc, {
            startY: 125,
            head: [['Parameter', 'Recorded Value']],
            body: tableBody,
            theme: 'striped',
            headStyles: { fillStyle: [30, 136, 229] },
        });

        // Footer Disclaimer
        const pageCount = doc.internal.getNumberOfPages();
        for (let i = 1; i <= pageCount; i++) {
            doc.setPage(i);
            doc.setFontSize(8);
            doc.setTextColor(150, 150, 150);
            doc.line(14, doc.internal.pageSize.height - 18, 196, doc.internal.pageSize.height - 18);
            doc.text('FOR RESEARCH USE ONLY. Not a medical diagnosis. Consult a qualified healthcare professional.', 14, doc.internal.pageSize.height - 10);
            doc.text(`Page ${i} of ${pageCount}`, 185, doc.internal.pageSize.height - 10);
        }

        doc.save(`${patientName.replace(/\s+/g, '_')}_Risk_Summary.pdf`);
    };

    return (
        <GlassCard className="mt-8 border-2 border-blue-500/20 shadow-xl animate-fade-in-up">
            {/* View Mode Toggle Header */}
            <div className="flex flex-col sm:flex-row items-center justify-between gap-4 border-b border-slate-200 pb-4 mb-6">
                <div>
                    <span className="text-xs font-bold text-blue-600 uppercase tracking-wider">HealthLens AI Risk Intelligence</span>
                    <h2 className="text-2xl font-black text-slate-800">Screening Assessment</h2>
                </div>

                <div className="bg-slate-100 p-1 rounded-xl flex text-xs font-bold shadow-inner">
                    <button
                        onClick={() => setViewMode('patient')}
                        className={`flex items-center gap-1.5 px-4 py-2 rounded-lg transition-all ${
                            viewMode === 'patient' ? 'bg-white text-blue-600 shadow-sm font-bold' : 'text-slate-500 hover:text-slate-700'
                        }`}
                    >
                        <Eye size={15} /> Patient View
                    </button>
                    <button
                        onClick={() => setViewMode('clinician')}
                        className={`flex items-center gap-1.5 px-4 py-2 rounded-lg transition-all ${
                            viewMode === 'clinician' ? 'bg-white text-blue-600 shadow-sm font-bold' : 'text-slate-500 hover:text-slate-700'
                        }`}
                    >
                        <Activity size={15} /> Clinician Attribution (SHAP)
                    </button>
                </div>
            </div>

            {/* OOD Warning Banner if applicable */}
            {explanation?.out_of_distribution && (
                <div className="flex items-start gap-3 bg-amber-50 border border-amber-300 text-amber-900 p-4 rounded-xl mb-6 text-xs">
                    <AlertTriangle className="text-amber-600 flex-shrink-0 mt-0.5" size={18} />
                    <div>
                        <span className="font-bold block">Out-Of-Distribution Parameter Notice:</span>
                        <ul className="list-disc list-inside mt-1 space-y-0.5">
                            {explanation.ood_warnings?.map((w, idx) => (
                                <li key={idx}>{w}</li>
                            ))}
                        </ul>
                    </div>
                </div>
            )}

            {/* Risk Badge Banner */}
            <div className="grid md:grid-cols-2 gap-6 mb-8">
                <div className="bg-slate-50 p-6 rounded-2xl border border-slate-200 flex flex-col justify-between">
                    <span className="text-xs font-bold uppercase tracking-wider text-slate-500 mb-1">Calibrated Risk Stratification</span>
                    <div className="flex items-center gap-4 my-2">
                        <div
                            className={`px-4 py-2 rounded-xl text-lg font-black tracking-wide border uppercase ${
                                riskBand === 'HIGH' || isHighRisk
                                    ? 'bg-red-100 text-red-800 border-red-300'
                                    : riskBand === 'MODERATE'
                                    ? 'bg-amber-100 text-amber-800 border-amber-300'
                                    : 'bg-green-100 text-green-800 border-green-300'
                            }`}
                        >
                            {riskBand || (isHighRisk ? 'HIGH' : 'LOW')} RISK
                        </div>
                        <div>
                            <span className="text-3xl font-black text-slate-800">{probPercent}%</span>
                            <span className="text-xs text-slate-500 block font-medium">Estimated Event Likelihood</span>
                        </div>
                    </div>
                    <p className="text-xs text-slate-500 mt-2">
                        Probability is calibrated via 5-fold cross-validation on population research benchmarks.
                    </p>
                </div>

                {/* Patient View: Plain Language Summary */}
                {viewMode === 'patient' ? (
                    <div className="bg-blue-50/60 p-6 rounded-2xl border border-blue-100 flex flex-col justify-between">
                        <span className="text-xs font-bold uppercase tracking-wider text-blue-900 mb-2">Patient Guidance Summary</span>
                        <p className="text-sm text-slate-700 leading-relaxed mb-3">
                            {isHighRisk
                                ? 'The predictive model identified metric patterns associated with an elevated risk profile. Please review with a healthcare provider.'
                                : 'Your entered parameters align with a lower risk category relative to cohort reference averages.'}
                        </p>
                        <div className="text-xs text-blue-800 bg-white/80 p-3 rounded-xl border border-blue-200">
                            <strong>Note:</strong> Statistical risk screening tool for research support. Not a diagnostic confirmation.
                        </div>
                    </div>
                ) : (
                    /* Clinician View: Technical Metadata */
                    <div className="bg-slate-900 text-slate-100 p-6 rounded-2xl border border-slate-800 font-mono text-xs flex flex-col justify-between">
                        <div className="space-y-2">
                            <span className="text-blue-400 font-bold uppercase tracking-wider block">Clinician Attribution Metadata</span>
                            <div>Model Architecture: HistGradientBoosting (Calibrated)</div>
                            <div>Pipeline Version: {explanation?.condition || 'model'}-v3.0</div>
                            <div>OOD Status: {explanation?.out_of_distribution ? 'TRUE (Exceeds Bounds)' : 'FALSE (In Bounds)'}</div>
                        </div>
                        <div className="text-slate-400 mt-4 text-[11px]">
                            SHAP values compute Shapley feature marginal contributions to the log-odds output.
                        </div>
                    </div>
                )}
            </div>

            {/* Explainable AI Breakdown Section */}
            {explanation && (
                <div className="mb-8 border-t border-slate-200 pt-6">
                    <h3 className="text-lg font-bold text-slate-800 mb-4 flex items-center gap-2">
                        <TrendingUp className="text-blue-600" size={20} />
                        Feature Attribution Breakdown (SHAP Analysis)
                    </h3>

                    {viewMode === 'patient' ? (
                        <div className="grid md:grid-cols-2 gap-6">
                            {/* Primary Risk Drivers */}
                            <div className="bg-red-50/50 p-5 rounded-2xl border border-red-200">
                                <h4 className="text-sm font-bold text-red-900 mb-3 flex items-center gap-2">
                                    <AlertTriangle size={16} className="text-red-600" />
                                    Primary Risk Drivers
                                </h4>
                                <ul className="space-y-2 text-xs text-red-800">
                                    {explanation.patient_explanation?.primary_risk_drivers?.map((driver, idx) => (
                                        <li key={idx} className="bg-white p-2.5 rounded-xl border border-red-100 font-medium">
                                            {driver}
                                        </li>
                                    ))}
                                </ul>
                            </div>

                            {/* Favorable Factors */}
                            <div className="bg-green-50/50 p-5 rounded-2xl border border-green-200">
                                <h4 className="text-sm font-bold text-green-900 mb-3 flex items-center gap-2">
                                    <ShieldCheck size={16} className="text-green-600" />
                                    Favorable Reference Indicators
                                </h4>
                                <ul className="space-y-2 text-xs text-green-800">
                                    {explanation.patient_explanation?.favorable_factors?.map((factor, idx) => (
                                        <li key={idx} className="bg-white p-2.5 rounded-xl border border-green-100 font-medium">
                                            {factor}
                                        </li>
                                    ))}
                                </ul>
                            </div>
                        </div>
                    ) : (
                        /* Clinician SHAP Attribution Matrix Table */
                        <div className="overflow-x-auto rounded-xl border border-slate-200">
                            <table className="w-full text-left text-xs font-mono">
                                <thead className="bg-slate-100 text-slate-700 border-b border-slate-200">
                                    <tr>
                                        <th className="p-3">Feature Name</th>
                                        <th className="p-3">Recorded Value</th>
                                        <th className="p-3">SHAP Weight</th>
                                        <th className="p-3">Direction</th>
                                    </tr>
                                </thead>
                                <tbody className="divide-y divide-slate-200 bg-white">
                                    {explanation.top_risk_contributors?.concat(explanation.top_protective_factors || []).map((item, idx) => (
                                        <tr key={idx} className="hover:bg-slate-50">
                                            <td className="p-3 font-semibold text-slate-800">{item.feature}</td>
                                            <td className="p-3 text-slate-600">{item.raw_value}</td>
                                            <td className={`p-3 font-bold ${item.is_risk_factor ? 'text-red-600' : 'text-green-600'}`}>
                                                {item.shap_attribution > 0 ? `+${item.shap_attribution}` : item.shap_attribution}
                                            </td>
                                            <td className="p-3">
                                                <span
                                                    className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                                                        item.is_risk_factor ? 'bg-red-100 text-red-800' : 'bg-green-100 text-green-800'
                                                    }`}
                                                >
                                                    {item.is_risk_factor ? 'ELEVATES RISK' : 'PROTECTIVE'}
                                                </span>
                                            </td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    )}
                </div>
            )}

            {/* Tailored Suggestions */}
            {suggestions && suggestions.length > 0 && (
                <div className="mb-6 bg-blue-50/40 border border-blue-200 rounded-xl p-4">
                    <h4 className="text-xs font-bold uppercase tracking-wider text-blue-900 mb-2">Tailored Lifestyle Guidance</h4>
                    <ul className="space-y-1 text-xs text-slate-700 list-disc list-inside">
                        {suggestions.map((sug, idx) => (
                            <li key={idx}>{sug}</li>
                        ))}
                    </ul>
                </div>
            )}

            {/* Model Limitations Box */}
            <div className="bg-slate-50 border border-slate-200 rounded-xl p-4 mb-6 text-xs text-slate-600">
                <span className="font-bold text-slate-800 flex items-center gap-1.5 mb-1">
                    <Info size={15} className="text-blue-600" /> Model Limitations &amp; Scope:
                </span>
                <ul className="list-disc list-inside space-y-0.5">
                    {explanation?.limitations?.map((lim, idx) => (
                        <li key={idx}>{lim}</li>
                    )) || (
                        <li>Screening statistical estimate for research use. Consult a physician for diagnosis.</li>
                    )}
                </ul>
            </div>

            {/* Download Report Button */}
            <div className="flex justify-end border-t border-slate-200 pt-4">
                <Button
                    onClick={generatePDF}
                    className="flex items-center gap-2 bg-blue-600 hover:bg-blue-700 text-white font-bold py-2.5 px-6 rounded-xl shadow-md"
                >
                    <FileSpreadsheet size={18} />
                    Download Research Summary (PDF)
                </Button>
            </div>
        </GlassCard>
    );
};

export default ResultCard;
