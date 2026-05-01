import React from 'react';
import { NavLink } from 'react-router-dom';
import { Home, Activity, Heart, Info, Menu, LineChart } from 'lucide-react';

const Sidebar = () => {
    return (
        <aside className="w-64 bg-card border-r border-borderLight h-screen fixed left-0 top-0 flex flex-col z-40 hidden md:flex">
            {/* Logo Area */}
            <div className="flex items-center gap-3 px-6 py-8 border-b border-borderLight h-20">
                <div className="bg-primary/10 p-2 rounded-lg text-primary">
                    <Activity size={24} />
                </div>
                <h1 className="text-xl font-bold text-slate-800 tracking-tight">
                    Health<span className="text-primary">Predict</span>
                </h1>
            </div>

            {/* Navigation Links */}
            <div className="flex flex-col gap-2 px-4 py-8 flex-grow">
                <NavItem to="/" icon={<Home size={20} />} label="Overview" />
                <NavItem to="/dashboard" icon={<LineChart size={20} />} label="Dashboard" />
                <NavItem to="/diabetes" icon={<Activity size={20} />} label="Diabetes Scan" />
                <NavItem to="/heart" icon={<Heart size={20} />} label="Cardiac Scan" />
                
                <div className="mt-auto">
                   <NavItem to="/about" icon={<Info size={20} />} label="About System" />
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
                `flex items-center gap-3 px-4 py-3 rounded-lg transition-all duration-200 font-medium ${isActive
                    ? 'bg-primary/10 text-primary border border-primary/20 shadow-sm'
                    : 'text-slate-500 hover:bg-slate-50 hover:text-slate-800 border border-transparent'
                }`
            }
        >
            {icon}
            <span className="text-sm">{label}</span>
        </NavLink>
    );
};

export default Sidebar;
