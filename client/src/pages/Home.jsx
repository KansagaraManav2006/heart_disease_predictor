import React from 'react';
import { useNavigate } from 'react-router-dom';
import Surface from '../components/Surface';
import Button from '../components/Button';
import StatusBadge from '../components/StatusBadge';
import { Activity, Heart, Shield, Clock, FileText, ArrowRight, Sparkles, CheckCircle2 } from 'lucide-react';

const Home = () => {
  const navigate = useNavigate();

  return (
    <div className="space-y-12 animate-fade-in">
      {/* Restrained Hero Glass Panel */}
      <Surface variant="hero" accent="teal" className="text-center py-12 md:py-16 px-6 relative">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-teal-500/15 border border-teal-500/30 text-teal-300 text-xs font-semibold mb-6">
          <Sparkles className="w-3.5 h-3.5" />
          <span>Research Platform · Calibrated Risk Stratification</span>
        </div>

        <h1 className="text-3xl md:text-5xl font-black tracking-tight text-slate-100 mb-4 max-w-4xl mx-auto leading-tight">
          Clinical Decision Support for <span className="text-teal-400">Cardiovascular</span> &amp; <span className="text-amber-400">Diabetic</span> Health
        </h1>

        <p className="text-sm md:text-base text-slate-300 max-w-2xl mx-auto leading-relaxed mb-8">
          HealthLens AI processes clinical biometrics and lab data through cross-validated machine learning models to provide explainable risk stratification and SHAP feature attribution.
        </p>

        <div className="flex flex-wrap items-center justify-center gap-4">
          <Button
            onClick={() => navigate('/diabetes')}
            variant="primary"
            size="lg"
            icon={Activity}
          >
            Start Diabetes Scan
          </Button>
          <Button
            onClick={() => navigate('/heart')}
            variant="secondary"
            size="lg"
            icon={Heart}
          >
            Start Cardiac Scan
          </Button>
        </div>
      </Surface>

      {/* Two Flat Assessment Cards with Condition Labels */}
      <div className="grid md:grid-cols-2 gap-8">
        <Surface variant="flat" accent="teal" className="flex flex-col justify-between h-full">
          <div>
            <div className="flex items-center justify-between gap-2 mb-4">
              <div className="p-3 rounded-2xl bg-teal-500/15 text-teal-400 border border-teal-500/30">
                <Activity className="w-6 h-6" />
              </div>
              <StatusBadge label="Condition Identity: Glycemic" status="processing" size="sm" />
            </div>

            <h2 className="text-xl font-bold text-slate-100 mb-2">Diabetes Risk Assessment</h2>
            <p className="text-xs md:text-sm text-slate-400 leading-relaxed mb-6">
              Evaluates fasting glucose, HbA1c estimates, BMI, blood pressure, insulin, and age to stratify calibrated diabetic risk.
            </p>

            <div className="space-y-2 mb-8 text-xs text-slate-300">
              <div className="flex items-center gap-2">
                <CheckCircle2 className="w-4 h-4 text-teal-400" />
                <span>Manual, Lab OCR Upload, and Guided Chatbot modes</span>
              </div>
              <div className="flex items-center gap-2">
                <CheckCircle2 className="w-4 h-4 text-teal-400" />
                <span>Out-Of-Distribution bound checking &amp; warnings</span>
              </div>
              <div className="flex items-center gap-2">
                <CheckCircle2 className="w-4 h-4 text-teal-400" />
                <span>Downloadable Research PDF summary report</span>
              </div>
            </div>
          </div>

          <Button
            onClick={() => navigate('/diabetes')}
            variant="primary"
            fullWidth
            icon={ArrowRight}
            iconPosition="right"
          >
            Launch Diabetes Scan
          </Button>
        </Surface>

        <Surface variant="flat" accent="amber" className="flex flex-col justify-between h-full">
          <div>
            <div className="flex items-center justify-between gap-2 mb-4">
              <div className="p-3 rounded-2xl bg-amber-500/15 text-amber-400 border border-amber-500/30">
                <Heart className="w-6 h-6" />
              </div>
              <StatusBadge label="Condition Identity: Cardiac" status="secondary" size="sm" />
            </div>

            <h2 className="text-xl font-bold text-slate-100 mb-2">Cardiac Risk Assessment</h2>
            <p className="text-xs md:text-sm text-slate-400 leading-relaxed mb-6">
              Comprehensive cardiovascular risk check analyzing blood pressure, serum cholesterol, max heart rate, and ST-depression.
            </p>

            <div className="space-y-2 mb-8 text-xs text-slate-300">
              <div className="flex items-center gap-2">
                <CheckCircle2 className="w-4 h-4 text-amber-400" />
                <span>Calibrated 5-fold cross-validation probability</span>
              </div>
              <div className="flex items-center gap-2">
                <CheckCircle2 className="w-4 h-4 text-amber-400" />
                <span>SHAP feature attribution matrix for clinicians</span>
              </div>
              <div className="flex items-center gap-2">
                <CheckCircle2 className="w-4 h-4 text-amber-400" />
                <span>Patient-friendly plain language risk summary</span>
              </div>
            </div>
          </div>

          <Button
            onClick={() => navigate('/heart')}
            variant="primary"
            fullWidth
            icon={ArrowRight}
            iconPosition="right"
          >
            Launch Cardiac Scan
          </Button>
        </Surface>
      </div>

      {/* Structured Methodology & Decision Support Pillars */}
      <div>
        <h2 className="text-xl font-bold text-slate-100 text-center mb-8">Platform Governance &amp; Research Foundations</h2>
        <div className="grid md:grid-cols-3 gap-6">
          <Surface variant="flat" className="p-6">
            <div className="p-2.5 rounded-xl bg-slate-900 text-teal-400 border border-slate-800 w-fit mb-4">
              <Shield className="w-5 h-5" />
            </div>
            <h3 className="text-sm font-bold text-slate-200 mb-2">Calibrated Models</h3>
            <p className="text-xs text-slate-400 leading-relaxed">
              Trained on validated reference datasets with probability calibration for consistent risk interpretation.
            </p>
          </Surface>

          <Surface variant="flat" className="p-6">
            <div className="p-2.5 rounded-xl bg-slate-900 text-amber-400 border border-slate-800 w-fit mb-4">
              <Clock className="w-5 h-5" />
            </div>
            <h3 className="text-sm font-bold text-slate-200 mb-2">Explainable AI (SHAP)</h3>
            <p className="text-xs text-slate-400 leading-relaxed">
              Provides granular feature importance weights to explain model predictions for both clinicians and patients.
            </p>
          </Surface>

          <Surface variant="flat" className="p-6">
            <div className="p-2.5 rounded-xl bg-slate-900 text-cyan-400 border border-slate-800 w-fit mb-4">
              <FileText className="w-5 h-5" />
            </div>
            <h3 className="text-sm font-bold text-slate-200 mb-2">Data Privacy &amp; Security</h3>
            <p className="text-xs text-slate-400 leading-relaxed">
              Role-based access control, cryptographic verification tokens, and compliance audit logging for research security.
            </p>
          </Surface>
        </div>
      </div>
    </div>
  );
};

export default Home;
