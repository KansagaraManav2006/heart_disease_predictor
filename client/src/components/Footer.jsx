import React from 'react';
import { Activity, Shield } from 'lucide-react';

const Footer = () => {
  return (
    <footer className="mt-auto border-t border-border bg-card/60 py-8 px-4 md:px-8 text-muted-foreground text-xs">
      <div className="max-w-6xl mx-auto flex flex-col md:flex-row items-center justify-between gap-6">
        <div className="flex flex-col gap-1 items-center md:items-start text-center md:text-left">
          <div className="flex items-center gap-2 text-foreground font-bold text-sm tracking-tight font-serif">
            <Activity className="w-4 h-4 text-primary" />
            <span>HealthLens AI Platform</span>
          </div>
          <p className="text-muted-foreground text-xs max-w-md">
            AI-driven clinical risk stratification decision support system for cardiovascular and diabetic conditions.
          </p>
        </div>

        <div className="flex flex-col items-center md:items-end gap-1 text-muted-foreground text-center md:text-right">
          <p className="flex items-center gap-1.5 text-[11px] font-medium text-foreground">
            <Shield className="w-3.5 h-3.5 text-primary" />
            Research Platform · Calibrated HistGradientBoosting Models
          </p>
          <p className="text-[11px]">&copy; {new Date().getFullYear()} HealthLens AI System. All rights reserved.</p>
        </div>
      </div>

      <div className="max-w-6xl mx-auto text-[11px] text-muted-foreground mt-6 pt-6 border-t border-border/50 text-center leading-relaxed">
        <span className="font-bold text-amber-500 uppercase tracking-wide">Research &amp; Decision Support Notice:</span> The predictions provided by this system are for research, educational, and screening support purposes only. They do not constitute a clinical diagnosis or medical treatment plan. Always consult a licensed healthcare professional.
      </div>
    </footer>
  );
};

export default Footer;
