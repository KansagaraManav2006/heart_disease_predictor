import React from 'react';
import { Menu } from 'lucide-react';

const Header = () => {
    return (
        <header className="bg-card border-b border-borderLight h-20 flex items-center justify-between px-8 sticky top-0 z-30 shadow-sm w-full">
            <div className="flex items-center gap-4">
                <button className="md:hidden p-2 text-slate-500 hover:text-slate-800 rounded-lg hover:bg-slate-50 transition-colors">
                    <Menu size={24} />
                </button>
                <div>
                    <h2 className="text-xl font-bold text-slate-800">Disease Prediction System</h2>
                    <p className="text-xs text-slate-500 font-medium hidden sm:block">AI-Based Health Risk Assessment</p>
                </div>
            </div>
            {/* Kept header clean right side */}
        </header>
    );
};

export default Header;
