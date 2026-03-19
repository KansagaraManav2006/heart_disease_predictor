import React from 'react';
import { Activity } from 'lucide-react';

const Footer = () => {
    return (
        <footer className="mt-auto border-t border-borderLight bg-card/80 py-8 px-8 text-center sm:text-left mt-12 w-full">
            <div className="max-w-4xl mx-auto flex flex-col md:flex-row items-center justify-between gap-6">
                <div className="flex flex-col gap-2 items-center md:items-start">
                    <div className="flex items-center gap-2 text-slate-700">
                        <Activity size={18} className="text-primary" />
                        <span className="font-semibold text-sm">HealthPredict System</span>
                    </div>
                    <p className="text-xs text-slate-500 max-w-xs">
                        An AI-driven platform for assessing risks of cardiovascular and diabetic conditions based on clinical data.
                    </p>
                </div>

                <div className="text-xs text-slate-500 flex flex-col gap-1 items-center md:items-end">
                    <p>Powered by React, Express, & Python ML Models</p>
                    <p>&copy; {new Date().getFullYear()} Disease Prediction Project. All Rights Reserved.</p>
                </div>
            </div>
            <div className="max-w-4xl mx-auto text-[10px] text-slate-400 mt-6 pt-6 border-t border-borderLight/50 text-center">
                <span className="font-semibold text-danger">Disclaimer:</span> The predictions provided by this system are for informational and educational purposes only. They are not a substitute for professional medical advice, diagnosis, or treatment. Always consult a healthcare professional.
            </div>
        </footer>
    );
};

export default Footer;
