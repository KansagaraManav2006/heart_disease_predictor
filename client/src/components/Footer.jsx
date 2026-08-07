import React from 'react';
import { Activity, Shield } from 'lucide-react';

const Footer = () => {
  return (
    <footer className="mt-auto border-t border-slate-800/80 bg-slate-900/60 py-8 px-4 md:px-8 text-slate-400 text-xs">
      <div className="max-w-6xl mx-auto flex flex-col md:flex-row items-center justify-between gap-6">
        <div className="flex flex-col gap-1 items-center md:items-start text-center md:text-left">
          <div className="flex items-center gap-2 text-slate-100 font-bold text-sm tracking-tight">
            <div className="w-6 h-6 rounded-lg bg-teal-500/20 text-teal-400 border border-teal-500/30 flex items-center justify-center">
              <Activity className="w-3.5 h-3.5" />
            </div>
            <span>HealthLens AI Platform</span>
          </div>
          <p className="text-slate-400 text-xs max-w-md">
            AI-driven clinical risk stratification decision support system for cardiovascular and diabetic conditions.
          </p>
        </div>

        <div className="flex flex-col items-center md:items-end gap-1 text-slate-400 text-center md:text-right">
          <p className="flex items-center gap-1.5 text-[11px] font-medium text-slate-200">
            <Shield className="w-3.5 h-3.5 text-teal-400" />
            Research Platform · Calibrated HistGradientBoosting Models
          </p>
          <p className="text-[11px]">&copy; {new Date().getFullYear()} HealthLens AI System. All rights reserved.</p>
        </div>
      </div>

      <div className="max-w-6xl mx-auto text-[11px] text-slate-400 mt-6 pt-6 border-t border-slate-800/50 text-center leading-relaxed">
        <span className="font-bold text-amber-400 uppercase tracking-wide">Research &amp; Decision Support Notice:</span> The predictions provided by this system are for research, educational, and screening support purposes only. They do not constitute a clinical diagnosis or medical treatment plan. Always consult a licensed healthcare professional.
      </div>
    </footer>
  );
};

export default Footer;
