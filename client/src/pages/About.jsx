import React from 'react';
import Surface from '../components/Surface';
import PageHeader from '../components/PageHeader';
import { Component, Server, Database, Shield, AlertTriangle, Cpu } from 'lucide-react';

const About = () => {
  return (
    <div className="space-y-8 animate-fade-in">
      <PageHeader
        title="Platform Methodology &amp; Architecture"
        subtitle="HealthLens AI is an open clinical decision support platform utilizing calibrated machine learning and SHAP attributions."
        badge={{ label: 'v3.0 Release', status: 'healthy' }}
      />

      {/* Research Use Disclaimer */}
      <Surface variant="flat" accent="amber" className="bg-amber-500/10 border-amber-500/30 text-amber-200">
        <div className="flex items-start gap-4">
          <div className="w-12 h-12 rounded-xl bg-amber-500/20 text-amber-400 border border-amber-500/30 flex items-center justify-center flex-shrink-0">
            <AlertTriangle className="w-6 h-6" />
          </div>
          <div>
            <h2 className="text-base font-bold text-amber-300 mb-1">Research &amp; Decision Support Notice</h2>
            <p className="text-xs md:text-sm text-slate-300 leading-relaxed">
              This platform is an exploratory research and clinical decision support system. It is not a standalone diagnostic medical device. Predictions are calibrated statistical estimates and must be interpreted by a qualified healthcare professional.
            </p>
          </div>
        </div>
      </Surface>

      {/* Methodology Section */}
      <Surface variant="flat">
        <h2 className="text-base font-bold text-slate-100 uppercase tracking-wider mb-4 pb-2 border-b border-slate-800 flex items-center gap-2">
          <Shield className="w-5 h-5 text-teal-400" /> Platform Overview &amp; Governance
        </h2>
        <p className="text-xs md:text-sm text-slate-300 leading-relaxed mb-4">
          HealthLens AI bridges machine learning algorithms (HistGradientBoosting with probability calibration) and clinical workflows. The platform provides explainable risk scores, out-of-distribution parameter detection, and evidence-grounded medical guideline retrieval.
        </p>
        <p className="text-xs md:text-sm text-slate-300 leading-relaxed">
          The system implements strict role-based access control, cryptographic verification tokens, and append-only security audit logging compliant with OWASP ASVS level 2 security standards.
        </p>
      </Surface>

      {/* Technology Stack Grid */}
      <div>
        <h2 className="text-base font-bold text-slate-100 uppercase tracking-wider mb-6 text-center">
          Engineered Technology Stack
        </h2>
        <div className="grid md:grid-cols-3 gap-6">
          <Surface variant="flat" accent="teal" className="p-6">
            <div className="w-10 h-10 rounded-xl bg-slate-900 text-teal-400 border border-slate-800 flex items-center justify-center mb-4">
              <Component className="w-5 h-5" />
            </div>
            <h3 className="text-sm font-bold text-slate-200 mb-3">Frontend Architecture</h3>
            <ul className="text-xs text-slate-400 space-y-2 list-disc list-inside font-mono">
              <li>React 19 &amp; Vite 7</li>
              <li>Tailwind CSS v4</li>
              <li>React Router v7</li>
              <li>Lucide UI Icons</li>
              <li>Dynamic PDF Generation</li>
            </ul>
          </Surface>

          <Surface variant="flat" accent="amber" className="p-6">
            <div className="w-10 h-10 rounded-xl bg-slate-900 text-amber-400 border border-slate-800 flex items-center justify-center mb-4">
              <Server className="w-5 h-5" />
            </div>
            <h3 className="text-sm font-bold text-slate-200 mb-3">Backend Gateway &amp; Auth</h3>
            <ul className="text-xs text-slate-400 space-y-2 list-disc list-inside font-mono">
              <li>Node.js &amp; Express REST</li>
              <li>PostgreSQL Persistence</li>
              <li>BCrypt Password Hashing</li>
              <li>JWT Session Tokens</li>
              <li>Append-Only Audit Log</li>
            </ul>
          </Surface>

          <Surface variant="flat" accent="coral" className="p-6">
            <div className="w-10 h-10 rounded-xl bg-slate-900 text-coral-400 border border-slate-800 flex items-center justify-center mb-4">
              <Cpu className="w-5 h-5" />
            </div>
            <h3 className="text-sm font-bold text-slate-200 mb-3">Machine Learning Engine</h3>
            <ul className="text-xs text-slate-400 space-y-2 list-disc list-inside font-mono">
              <li>FastAPI Python 3.11</li>
              <li>HistGradientBoosting</li>
              <li>5-Fold Cross-Validation</li>
              <li>SHAP Feature Attribution</li>
              <li>PyMuPDF + Tesseract OCR</li>
            </ul>
          </Surface>
        </div>
      </div>
    </div>
  );
};

export default About;
