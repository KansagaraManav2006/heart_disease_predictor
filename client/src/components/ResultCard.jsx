import React from 'react';
import GlassCard from './GlassCard';
import Button from './Button';
import { jsPDF } from 'jspdf';
import autoTable from 'jspdf-autotable';

const ResultCard = ({ prediction, probability, riskLevel, extras = [], suggestions = [] }) => {
    const isHighRisk = prediction === 1;
    const probPercent = (probability * 100).toFixed(1);

    const generatePDF = () => {
        const doc = new jsPDF();
        const date = new Date().toLocaleDateString();
        const time = new Date().toLocaleTimeString();
        const patientName = extras.find(e => e.label === 'Patient' || e.label === 'Patient Name')?.value || 'Anonymous';

        // 1. Header Ribbon (Medical Blue)
        doc.setFillColor(30, 136, 229); // #1E88E5
        doc.rect(0, 0, 210, 40, 'F');
        
        doc.setTextColor(255, 255, 255);
        doc.setFontSize(24);
        doc.setFont("helvetica", "bold");
        doc.text('Disease Prediction System', 14, 20);
        
        doc.setFontSize(12);
        doc.setFont("helvetica", "normal");
        doc.text('Official Health Risk Assessment Report', 14, 28);

        // 2. Patient Demographics Box
        doc.setDrawColor(200, 200, 200);
        doc.setFillColor(248, 250, 252);
        doc.rect(14, 48, 182, 30, 'FD'); // Fill and border
        
        doc.setTextColor(50, 50, 50);
        doc.setFontSize(10);
        doc.setFont("helvetica", "bold");
        doc.text('PATIENT DETAILS', 18, 55);
        
        doc.setFont("helvetica", "normal");
        doc.text(`Name: ${patientName}`, 18, 64);
        doc.text(`Date of Scan: ${date}`, 100, 64);
        doc.text(`Time of Scan: ${time}`, 100, 72);
        doc.text(`Report ID: #${Math.floor(Math.random() * 1000000)}`, 18, 72);

        // 3. Clinical Prediction Results
        doc.setFont("helvetica", "bold");
        doc.setFontSize(14);
        doc.setTextColor(30, 41, 59);
        doc.text('Diagnostic Prediction', 14, 90);
        doc.setLineWidth(0.5);
        doc.line(14, 92, 196, 92);
        
        const riskColor = isHighRisk ? [239, 68, 68] : [34, 197, 94]; // Red or Green
        
        doc.setFontSize(12);
        doc.setTextColor(50, 50, 50);
        doc.text('Evaluated Risk Level:', 14, 102);
        
        doc.setFont("helvetica", "bold");
        doc.setTextColor(riskColor[0], riskColor[1], riskColor[2]);
        doc.text(`${riskLevel.toUpperCase()}`, 60, 102);
        
        doc.setFont("helvetica", "normal");
        doc.setTextColor(50, 50, 50);
        doc.text('Model Confidence:', 14, 110);
        doc.setFont("helvetica", "bold");
        doc.text(`${probPercent}%`, 52, 110);

        // Summary Text Inside a bounding box
        doc.setDrawColor(riskColor[0], riskColor[1], riskColor[2]);
        doc.setFillColor(255, 255, 255);
        doc.setLineWidth(0.2);
        doc.rect(14, 118, 182, 25, 'FD');

        doc.setFont("helvetica", "normal");
        doc.setFontSize(10);
        doc.setTextColor(70, 70, 70);
        const summaryText = isHighRisk
            ? "ATTENTION: The predictive model has identified patterns consistent with a significantly elevated risk profile. It is strongly recommended to use this report as a supplementary tool and consult a certified healthcare professional immediately for clinical diagnosis."
            : "The parameters provided indicate a low risk profile corresponding to average healthy baselines. Continue to maintain a healthy lifestyle, focus on preventative care, and maintain standard regular medical check-ups.";
        const splitText = doc.splitTextToSize(summaryText, 175);
        doc.text(splitText, 18, 126);

        // 4. Clinical Parameters Table
        if (extras && extras.length > 0) {
            // Filter out 'Patient' since it's in the header box
            const tableExtras = extras.filter(e => e.label !== 'Patient' && e.label !== 'Patient Name');
            const tableData = tableExtras.map(item => [item.label, item.value]);

            autoTable(doc, {
                startY: 155,
                head: [['Clinical Parameter', 'Recorded Value']],
                body: tableData,
                theme: 'grid',
                headStyles: { 
                    fillColor: [30, 136, 229],
                    textColor: 255,
                    fontStyle: 'bold'
                },
                alternateRowStyles: { fillColor: [248, 250, 252] },
                margin: { left: 14, right: 14 }
            });
        }

        // 5. Smart Health Recommendations
        let finalY = doc?.lastAutoTable?.finalY ? doc.lastAutoTable.finalY + 15 : 180;
        
        if (suggestions && suggestions.length > 0) {
            doc.setFontSize(12);
            doc.setTextColor(30, 41, 59);
            doc.setFont("helvetica", "bold");
            doc.text('Health Recommendations', 14, finalY);
            doc.setLineWidth(0.5);
            doc.line(14, finalY + 2, 196, finalY + 2);
            
            finalY += 10;
            doc.setFont("helvetica", "normal");
            doc.setFontSize(10);
            doc.setTextColor(50, 50, 50);
            
            suggestions.forEach((sug) => {
                const textLines = doc.splitTextToSize(`• ${sug}`, 175);
                doc.text(textLines, 14, finalY);
                finalY += (textLines.length * 6) + 2; 
                if (finalY > 270) {
                    doc.addPage();
                    finalY = 20;
                }
            });
            finalY += 15;
        }

        // 6. Official Signature Line
        doc.setDrawColor(150, 150, 150);
        doc.line(140, finalY, 196, finalY);
        doc.setFontSize(10);
        doc.setTextColor(100, 100, 100);
        doc.text("Attending Physician / Reviewer", 145, finalY + 6);

        // 7. Footer (Applies to all pages)
        const pageCount = doc.internal.getNumberOfPages();
        for (let i = 1; i <= pageCount; i++) {
            doc.setPage(i);
            doc.setFontSize(8);
            doc.setTextColor(150, 150, 150);
            doc.line(14, doc.internal.pageSize.height - 15, 196, doc.internal.pageSize.height - 15);
            const footerText = 'CONFIDENTIAL RECORD. Computed algorithmically. Do not replace professional medical judgment.';
            doc.text(footerText, 14, doc.internal.pageSize.height - 10);
            doc.text(`Page ${i} of ${pageCount}`, 185, doc.internal.pageSize.height - 10);
        }

        doc.save(`${patientName.replace(/\s+/g, '_')}_Risk_Report.pdf`);
    };

    return (
        <GlassCard className="mt-8 animate-fade-in-up border-t-4" strong={true} style={{ borderTopColor: isHighRisk ? '#ef4444' : '#22c55e' }}>
            <h3 className="text-xl font-bold mb-4 text-slate-800 border-b pb-2">Prediction Results</h3>

            <div className="flex flex-col md:flex-row items-center justify-between gap-6 py-4">
                <div className="flex-1 text-center md:text-left">
                    <p className="text-slate-500 text-sm mb-1 font-medium">Status Classification</p>
                    <div className="flex items-center justify-center md:justify-start gap-3 mb-2">
                        <span className={`inline-flex items-center justify-center w-3 h-3 rounded-full ${isHighRisk ? 'bg-danger animate-pulse shadow-sm shadow-danger/50' : 'bg-primary shadow-sm shadow-primary/50'}`}></span>
                        <p className={`text-3xl font-bold ${isHighRisk ? 'text-danger' : 'text-primary'}`}>
                            {riskLevel}
                        </p>
                    </div>

                    <p className="text-slate-600 mt-4 leading-relaxed font-medium">
                        {isHighRisk
                            ? "The model has detected patterns consistent with a high risk profile. It is strongly recommended to consult a healthcare professional for a formal diagnosis."
                            : "The parameters provided indicate a low risk profile. Continue to maintain a healthy lifestyle and regular check-ups."}
                    </p>
                </div>

                <div className="bg-slate-50 p-6 rounded-xl border border-borderLight w-full md:w-auto text-center min-w-[200px] flex flex-col justify-center items-center shadow-sm">
                    <p className="text-slate-500 text-sm mb-2 font-medium">Confidence Score</p>
                    <p className="text-5xl font-black text-slate-800">{probPercent}%</p>

                    {/* Progress Bar Indicator */}
                    <div className="w-full bg-slate-200 rounded-full h-2.5 mt-4 overflow-hidden">
                        <div className={`h-2.5 rounded-full ${isHighRisk ? 'bg-danger' : 'bg-primary'}`} style={{ width: `${probPercent}%` }}></div>
                    </div>
                </div>
            </div>

            {/* Smart Suggestions UI */}
            {suggestions && suggestions.length > 0 && (
                <div className="mt-8 bg-blue-50/50 rounded-xl p-6 border border-blue-100">
                    <h4 className="text-lg font-bold text-slate-800 mb-4 flex items-center gap-2">
                        <svg className="w-5 h-5 text-blue-500" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" /></svg>
                        Personalized Recommendations
                    </h4>
                    <ul className="space-y-3">
                        {suggestions.map((sug, idx) => (
                            <li key={idx} className="flex items-start gap-3">
                                <div className="mt-1 flex-shrink-0 w-5 h-5 rounded-full bg-blue-100 flex items-center justify-center text-blue-600">
                                    <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" /></svg>
                                </div>
                                <p className="text-slate-700 font-medium leading-relaxed">{sug}</p>
                            </li>
                        ))}
                    </ul>
                </div>
            )}

            <div className="mt-6 flex justify-center border-t border-borderLight pt-6">
                <Button onClick={generatePDF} variant="secondary" className="flex items-center gap-2 max-w-xs">
                    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                        <polyline points="7 10 12 15 17 10"></polyline>
                        <line x1="12" y1="15" x2="12" y2="3"></line>
                    </svg>
                    Download PDF Report
                </Button>
            </div>
        </GlassCard>
    );
};

export default ResultCard;
