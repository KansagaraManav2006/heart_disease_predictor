import React from 'react';
import { NavLink } from 'react-router-dom';
import { Home, Activity, Heart, Info, Menu } from 'lucide-react';

const Navbar = () => {
    return (
        <nav className="glass sticky top-0 z-50 px-4 py-3 md:px-8 flex items-center justify-between mb-8 rounded-none border-x-0 border-t-0 bg-darkBg/80">
            <div className="flex items-center gap-3">
                <div className="bg-primary/20 p-2 rounded-lg text-primary">
                    <Activity size={24} />
                </div>
                <h1 className="text-xl md:text-2xl font-bold text-white tracking-tight">
                    Health<span className="text-primary">Predict</span>
                </h1>
            </div>

            {/* Desktop Navigation */}
            <div className="hidden md:flex items-center gap-2">
                <NavItem to="/" icon={<Home size={18} />} label="Home" />
                <NavItem to="/diabetes" icon={<Activity size={18} />} label="Diabetes Scan" />
                <NavItem to="/heart" icon={<Heart size={18} />} label="Cardiac Scan" />
                <NavItem to="/about" icon={<Info size={18} />} label="About" />
            </div>

            {/* Mobile Menu Button - In a real app this would toggle a mobile menu state */}
            <div className="md:hidden">
                <button className="p-2 text-slate-300 hover:text-white glass-pill hover:bg-slate-800">
                    <Menu size={24} />
                </button>
            </div>
        </nav>
    );
};

const NavItem = ({ to, icon, label }) => {
    return (
        <NavLink
            to={to}
            className={({ isActive }) =>
                `flex items-center gap-2 px-4 py-2 rounded-full transition-all duration-300 ${isActive
                    ? 'bg-primary/20 text-primary border border-primary/30 shadow-lg shadow-primary/20'
                    : 'text-slate-300 hover:bg-white/5 hover:text-white border border-transparent'
                }`
            }
        >
            {icon}
            <span className="text-sm font-medium">{label}</span>
        </NavLink>
    );
};

export default Navbar;
