import React from 'react';
import { NavLink } from 'react-router-dom';
import { Home, Activity, Heart, Info, LineChart, Users, BookOpen, Cpu, Lock, ShieldCheck } from 'lucide-react';
import { useAuth } from '../context/useAuth';

const Sidebar = () => {
    const { user } = useAuth();
    const isClinicianOrAdmin = user?.role === 'CLINICIAN' || user?.role === 'ADMIN';

    return (
        <aside className="w-64 bg-card border-r border-borderLight h-screen fixed left-0 top-0 flex flex-col z-40 hidden md:flex">
            {/* Logo Area */}
            <div className="flex items-center gap-3 px-6 py-8 border-b border-borderLight h-20">
                <div className="bg-blue-600/10 p-2 rounded-xl text-blue-600">
                    <Activity size={24} />
                </div>
                <h1 className="text-xl font-black text-slate-800 tracking-tight">
                    HealthLens <span className="text-blue-600">AI</span>
                </h1>
            </div>

            {/* Navigation Links */}
            <div className="flex flex-col gap-1.5 px-4 py-6 flex-grow overflow-y-auto">
                <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider px-3 mb-1">Scans &amp; Patients</span>
                <NavItem to="/" icon={<Home size={18} />} label="Overview" />
                <NavItem to="/dashboard" icon={<LineChart size={18} />} label="Patient Dashboard" />
                <NavItem to="/diabetes" icon={<Activity size={18} />} label="Diabetes Risk Scan" />
                <NavItem to="/heart" icon={<Heart size={18} />} label="Cardiac Risk Scan" />
                <NavItem to="/knowledge" icon={<BookOpen size={18} />} label="Medical Guidelines" />

                {isClinicianOrAdmin && (
                    <>
                        <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider px-3 mt-4 mb-1">Clinical &amp; Admin</span>
                        <NavItem to="/worklist" icon={<Users size={18} />} label="Clinician Worklist" />
                        <NavItem to="/models" icon={<Cpu size={18} />} label="Model Registry" />
                        <NavItem to="/audit" icon={<Lock size={18} />} label="Security Audit" />
                        <NavItem to="/system-health" icon={<ShieldCheck size={18} />} label="System Health" />
                    </>
                )}

                <div className="mt-auto pt-4 border-t border-slate-200">
                    <NavItem to="/about" icon={<Info size={18} />} label="About Platform" />
                </div>
            </div>
        </aside>
    );
};

const NavItem = ({ to, icon, label }) => {
    return (
        <NavLink
            to={to}
            className={({ isActive }) =>
                `flex items-center gap-3 px-3.5 py-2.5 rounded-xl transition-all duration-200 font-semibold ${isActive
                    ? 'bg-blue-50 text-blue-600 border border-blue-200 shadow-sm'
                    : 'text-slate-600 hover:bg-slate-50 hover:text-slate-900 border border-transparent'
                }`
            }
        >
            {icon}
            <span className="text-xs">{label}</span>
        </NavLink>
    );
};

export default Sidebar;
