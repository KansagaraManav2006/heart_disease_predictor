import React from 'react';
import { Menu, Bell, User } from 'lucide-react';

const Header = () => {
    return (
        <header className="bg-card border-b border-borderLight h-20 flex items-center justify-between px-8 sticky top-0 z-30 shadow-sm w-full">
            <div className="flex items-center gap-4">
                <button className="md:hidden p-2 text-slate-500 hover:text-slate-800 rounded-lg hover:bg-slate-50 transition-colors">
                    <Menu size={24} />
                </button>
                <div>
                    <h2 className="text-xl font-bold text-slate-800">Disease Prediction System</h2>
                    <p className="text-xs text-slate-500 font-medium hidden sm:block">AI-Based Health Risk Assessment Dashboard</p>
                </div>
            </div>

            <div className="flex items-center gap-4">
                <button className="p-2 text-slate-400 hover:text-primary transition-colors relative">
                    <Bell size={20} />
                    <span className="absolute top-1.5 right-1.5 w-2 h-2 bg-danger rounded-full"></span>
                </button>
                <div className="h-8 w-8 rounded-full bg-primary/10 text-primary flex items-center justify-center border border-primary/20">
                    <User size={16} />
                </div>
            </div>
        </header>
    );
};

export default Header;
