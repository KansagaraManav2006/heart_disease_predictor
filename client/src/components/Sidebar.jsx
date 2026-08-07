import React from 'react';
import { NavLink } from 'react-router-dom';
import { Activity, LineChart, Heart, Users, Cpu, Lock, ShieldCheck, BookOpen, Info } from 'lucide-react';
import { useAuth } from '../context/useAuth';

const Sidebar = () => {
  const { user } = useAuth();
  const isClinicianOrAdmin = user?.role === 'CLINICIAN' || user?.role === 'ADMIN';

  return (
    <aside className="w-64 bg-slate-900 border-r border-slate-800 h-screen fixed left-0 top-0 flex flex-col z-40 hidden md:flex">
      <div className="flex items-center gap-3 px-6 py-6 border-b border-slate-800 h-16">
        <Activity className="w-5 h-5 text-teal-400" />
        <span className="text-base font-black text-white">HealthLens AI</span>
      </div>

      <div className="flex flex-col gap-1.5 px-4 py-6 flex-grow overflow-y-auto">
        <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider px-3 mb-1">Navigation</span>
        <NavItem to="/dashboard" icon={LineChart} label="Patient Dashboard" />
        <NavItem to="/diabetes" icon={Activity} label="Diabetes Scan" />
        <NavItem to="/heart" icon={Heart} label="Cardiac Scan" />
        <NavItem to="/knowledge" icon={BookOpen} label="Medical Knowledge" />

        {isClinicianOrAdmin && (
          <>
            <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider px-3 mt-4 mb-1">Clinical &amp; Admin</span>
            <NavItem to="/worklist" icon={Users} label="Clinician Worklist" />
            <NavItem to="/models" icon={Cpu} label="Model Registry" />
            <NavItem to="/audit" icon={Lock} label="Security Audit" />
            <NavItem to="/system-health" icon={ShieldCheck} label="System Health" />
          </>
        )}

        <div className="mt-auto pt-4 border-t border-slate-800">
          <NavItem to="/about" icon={Info} label="About Platform" />
        </div>
      </div>
    </aside>
  );
};

const NavItem = ({ to, icon: Icon, label }) => {
  return (
    <NavLink
      to={to}
      className={({ isActive }) =>
        `flex items-center gap-3 px-3 py-2.5 rounded-xl transition-all text-xs font-semibold ${
          isActive
            ? 'bg-slate-800 text-teal-400 border border-teal-500/30'
            : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/40 border border-transparent'
        }`
      }
    >
      {Icon && <Icon className="w-4 h-4" />}
      <span>{label}</span>
    </NavLink>
  );
};

export default Sidebar;
