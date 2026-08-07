import React, { useState } from 'react';
import Surface from '../components/Surface';
import PageHeader from '../components/PageHeader';
import RiskBadge from '../components/RiskBadge';
import StatusBadge from '../components/StatusBadge';
import Button from '../components/Button';
import { FileSpreadsheet, Printer, User, Calendar, ShieldAlert, CheckCircle2, HelpCircle, FileText } from 'lucide-react';

const ConsultationExport = () => {
  const [downloading, setDownloading] = useState(false);

  const patientData = {
    name: 'Smit Kansagara',
    dob: '1988-06-14',
    mrn: 'HL-2026-8841',
    date: new Date().toLocaleDateString(),
    diabetesRisk: { band: 'LOW', prob: 28, date: '2026-01-10' },
    cardiacRisk: { band: 'MODERATE', prob: 42, date: '2026-01-10' },
    topRiskDrivers: [
      'Systolic Blood Pressure (138 mmHg) — Elevates vascular resistance',
      'Serum Cholesterol (215 mg/dL) — Moderate lipid elevation',
    ],
    favorableFactors: [
      'HbA1c (6.3%) — Improved glycemic control',
      'Non-smoker status & Normal resting ECG',
    ],
    activeMedications: [
      'Metformin HCl 1000 mg twice daily',
      'Empagliflozin 10 mg once daily',
      'Atorvastatin 20 mg once daily',
      'Lisinopril 10 mg once daily',
    ],
    questionsForDoctor: [
      'Should we adjust the Lisinopril dosage based on my recent 122 mmHg BP reading?',
      'Is an eGFR blood test recommended at my next 6-month checkup?',
      'Would adding an SGLT2 inhibitor provide additional renal protection?',
    ],
  };

  const handleGeneratePhysicianPDF = async () => {
    try {
      setDownloading(true);
      const { jsPDF } = await import('jspdf');
      const autoTableModule = await import('jspdf-autotable');
      const autoTable = autoTableModule.default || autoTableModule;

      const doc = new jsPDF();
      const date = new Date().toLocaleDateString();

      // Header Banner
      doc.setFillColor(185, 28, 28); // Sturdy Crimson
      doc.rect(0, 0, 210, 36, 'F');

      doc.setTextColor(255, 255, 255);
      doc.setFontSize(18);
      doc.setFont('helvetica', 'bold');
      doc.text('HealthLens AI — Physician Consultation Handoff', 14, 16);

      doc.setFontSize(9);
      doc.setFont('helvetica', 'normal');
      doc.text(`Patient: ${patientData.name} | MRN: ${patientData.mrn} | Date: ${date}`, 14, 28);

      // Patient Metadata Box
      doc.setDrawColor(200, 200, 200);
      doc.setFillColor(248, 250, 252);
      doc.rect(14, 42, 182, 22, 'FD');

      doc.setTextColor(40, 40, 40);
      doc.setFontSize(10);
      doc.setFont('helvetica', 'bold');
      doc.text('CLINICAL SUMMARY & RISK STRATIFICATION', 18, 50);
      doc.setFont('helvetica', 'normal');
      doc.text(`Glycemic Risk: ${patientData.diabetesRisk.band} (${patientData.diabetesRisk.prob}%)`, 18, 58);
      doc.text(`Cardiac Risk: ${patientData.cardiacRisk.band} (${patientData.cardiacRisk.prob}%)`, 105, 58);

      // Table 1: Active Prescriptions
      autoTable(doc, {
        startY: 70,
        head: [['Active Regimen / Medication', 'Dosage & Schedule']],
        body: patientData.activeMedications.map((m) => {
          const parts = m.split(' ');
          return [parts.slice(0, 2).join(' '), parts.slice(2).join(' ')];
        }),
        theme: 'striped',
        headStyles: { fillStyle: [185, 28, 28] },
      });

      // Table 2: Top Risk Drivers & Favorable Factors
      const currentY = doc.lastAutoTable.finalY + 10;
      doc.setFontSize(11);
      doc.setFont('helvetica', 'bold');
      doc.text('SHAP Feature Drivers & Patient Consultation Questions', 14, currentY);

      autoTable(doc, {
        startY: currentY + 4,
        head: [['Primary Risk Drivers', 'Favorable Protective Indicators']],
        body: [
          [patientData.topRiskDrivers.join('\n'), patientData.favorableFactors.join('\n')],
        ],
        theme: 'plain',
        headStyles: { fillStyle: [230, 230, 230], textColor: [40, 40, 40] },
      });

      // Questions for Doctor Section
      const qY = doc.lastAutoTable.finalY + 10;
      doc.setFontSize(10);
      doc.setFont('helvetica', 'bold');
      doc.text('Structured Patient Questions for Clinician:', 14, qY);
      doc.setFont('helvetica', 'normal');
      doc.setFontSize(9);

      patientData.questionsForDoctor.forEach((q, idx) => {
        doc.text(`${idx + 1}. ${q}`, 16, qY + 6 + idx * 6);
      });

      // Footer Disclaimer
      doc.setFontSize(8);
      doc.setTextColor(150, 150, 150);
      doc.line(14, doc.internal.pageSize.height - 18, 196, doc.internal.pageSize.height - 18);
      doc.text('FOR PHYSICIAN CONSULTATION & DECISION SUPPORT. Not a diagnostic order.', 14, doc.internal.pageSize.height - 10);

      doc.save(`${patientData.name.replace(/\s+/g, '_')}_Physician_Consultation.pdf`);
    } catch (err) {
      console.error('Failed to generate consultation PDF:', err);
    } finally {
      setDownloading(false);
    }
  };

  return (
    <div className="space-y-8 animate-fade-in">
      <PageHeader
        title="Physician Consultation &amp; Handoff Summary"
        subtitle="Generates a structured 1-page clinical summary optimized for primary care doctor appointments."
        badge={{ label: 'Ready for Doctor Visit', status: 'healthy' }}
        action={
          <Button
            onClick={handleGeneratePhysicianPDF}
            loading={downloading}
            loadingLabel="Generating Physician PDF..."
            variant="primary"
            size="sm"
            icon={FileSpreadsheet}
          >
            Export Physician PDF
          </Button>
        }
      />

      {/* 1-Page Document Preview Container */}
      <Surface variant="hero" accent="teal" className="space-y-6">
        {/* Document Header Bar */}
        <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4 pb-6 border-b border-border">
          <div className="flex items-center gap-3">
            <div className="p-3 rounded-md bg-primary/10 text-primary border border-primary/30">
              <FileText className="w-6 h-6" />
            </div>
            <div>
              <h2 className="text-xl font-bold text-foreground font-serif">Clinical Appointment Summary</h2>
              <p className="text-xs text-muted-foreground font-mono">
                Patient: {patientData.name} | MRN: {patientData.mrn} | Date: {patientData.date}
              </p>
            </div>
          </div>
          <StatusBadge label="PDF Handoff Ready" status="healthy" />
        </div>

        {/* Stratification Summary Grid */}
        <div className="grid md:grid-cols-2 gap-6">
          <div className="bg-card p-5 rounded border border-border space-y-3">
            <span className="text-xs font-bold text-muted-foreground uppercase tracking-wider block">
              Glycemic Risk Stratum
            </span>
            <div className="flex items-center gap-3">
              <RiskBadge riskBand={patientData.diabetesRisk.band} score={patientData.diabetesRisk.prob} size="md" />
              <span className="text-xs text-muted-foreground font-mono">Tested: {patientData.diabetesRisk.date}</span>
            </div>
          </div>

          <div className="bg-card p-5 rounded border border-border space-y-3">
            <span className="text-xs font-bold text-muted-foreground uppercase tracking-wider block">
              Cardiovascular Risk Stratum
            </span>
            <div className="flex items-center gap-3">
              <RiskBadge riskBand={patientData.cardiacRisk.band} score={patientData.cardiacRisk.prob} size="md" />
              <span className="text-xs text-muted-foreground font-mono">Tested: {patientData.cardiacRisk.date}</span>
            </div>
          </div>
        </div>

        {/* Risk Drivers & Protective Factors */}
        <div className="grid md:grid-cols-2 gap-6">
          <div className="bg-destructive/10 p-5 rounded border border-destructive/30 space-y-3">
            <h4 className="text-xs font-bold text-destructive-foreground flex items-center gap-2 uppercase tracking-wider">
              <ShieldAlert className="w-4 h-4 text-destructive" /> Primary Clinical Risk Drivers
            </h4>
            <ul className="space-y-2 text-xs text-foreground font-medium">
              {patientData.topRiskDrivers.map((driver, idx) => (
                <li key={idx} className="bg-card p-3 rounded border border-border">
                  {driver}
                </li>
              ))}
            </ul>
          </div>

          <div className="bg-card p-5 rounded border border-border space-y-3">
            <h4 className="text-xs font-bold text-foreground flex items-center gap-2 uppercase tracking-wider">
              <CheckCircle2 className="w-4 h-4 text-primary" /> Favorable Protective Biomarkers
            </h4>
            <ul className="space-y-2 text-xs text-foreground font-medium">
              {patientData.favorableFactors.map((factor, idx) => (
                <li key={idx} className="bg-muted p-3 rounded border border-border">
                  {factor}
                </li>
              ))}
            </ul>
          </div>
        </div>

        {/* Active Prescriptions */}
        <div className="bg-card p-5 rounded border border-border space-y-3">
          <h4 className="text-xs font-bold text-foreground uppercase tracking-wider">
            Active Pharmacotherapy Regimen ({patientData.activeMedications.length} Prescriptions)
          </h4>
          <div className="grid sm:grid-cols-2 gap-3 text-xs">
            {patientData.activeMedications.map((med, idx) => (
              <div key={idx} className="p-3 bg-muted rounded border border-border font-medium text-foreground">
                {med}
              </div>
            ))}
          </div>
        </div>

        {/* Structured Questions for Doctor */}
        <div className="bg-accent/10 border border-amber-500/30 p-5 rounded space-y-3 text-xs">
          <h4 className="text-xs font-bold text-amber-500 uppercase tracking-wider flex items-center gap-2">
            <HelpCircle className="w-4 h-4" /> Prepared Questions for Physician Visit
          </h4>
          <ol className="space-y-2 list-decimal list-inside text-foreground font-medium">
            {patientData.questionsForDoctor.map((q, idx) => (
              <li key={idx} className="bg-card p-3 rounded border border-border">
                {q}
              </li>
            ))}
          </ol>
        </div>

        {/* Export Button */}
        <div className="flex justify-end pt-4 border-t border-border">
          <Button
            onClick={handleGeneratePhysicianPDF}
            loading={downloading}
            loadingLabel="Exporting Document..."
            variant="primary"
            icon={Printer}
            className="font-bold"
          >
            Download Consultation PDF
          </Button>
        </div>
      </Surface>
    </div>
  );
};

export default ConsultationExport;
