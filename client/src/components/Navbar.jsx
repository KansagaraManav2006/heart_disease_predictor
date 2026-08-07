import React, { useState } from 'react';
import { NavLink, Link } from 'react-router-dom';
import { Home, Activity, Heart, Info, Menu, X, LogIn, UserPlus, LogOut, UserCheck } from 'lucide-react';
import { useAuth } from '../context/useAuth';

const Navbar = () => {
    const [menuOpen, setMenuOpen] = useState(false);
    const { user, isAuthenticated, signOut } = useAuth();

    const closeMenu = () => setMenuOpen(false);

    return (
        <>
            <nav className="glass sticky top-0 z-50 px-4 py-3 md:px-8 flex items-center justify-between mb-8 rounded-none border-x-0 border-t-0 bg-darkBg/80">
                <Link to="/" className="flex items-center gap-3">
                    <div className="bg-primary/20 p-2 rounded-lg text-primary">
                        <Activity size={24} />
                    </div>
                    <h1 className="text-xl md:text-2xl font-bold text-white tracking-tight">
                        Health<span className="text-primary">Lens AI</span>
                    </h1>
                </Link>

                {/* Desktop Navigation */}
                <div className="hidden md:flex items-center gap-2">
                    <NavItem to="/" icon={<Home size={18} />} label="Home" />
                    <NavItem to="/diabetes" icon={<Activity size={18} />} label="Diabetes Scan" />
                    <NavItem to="/heart" icon={<Heart size={18} />} label="Cardiac Scan" />
                    <NavItem to="/about" icon={<Info size={18} />} label="About" />

                    <div className="h-6 w-[1px] bg-slate-700 mx-2" />

                    {isAuthenticated ? (
                        <div className="flex items-center gap-3">
                            <span className="text-xs font-semibold px-2.5 py-1 rounded-full bg-blue-500/20 text-blue-300 border border-blue-500/30 flex items-center gap-1.5">
                                <UserCheck size={14} />
                                {user?.role} ({user?.email.split('@')[0]})
                            </span>
                            <button
                                onClick={signOut}
                                className="flex items-center gap-1.5 text-xs text-slate-300 hover:text-white px-3 py-2 rounded-full hover:bg-white/10 transition-colors"
                            >
                                <LogOut size={16} /> Sign Out
                            </button>
                        </div>
                    ) : (
                        <div className="flex items-center gap-2">
                            <NavLink
                                to="/login"
                                className="flex items-center gap-1.5 px-4 py-2 rounded-full text-xs font-semibold text-slate-200 hover:text-white hover:bg-white/10 transition-colors"
                            >
                                <LogIn size={15} /> Sign In
                            </NavLink>
                            <NavLink
                                to="/register"
                                className="flex items-center gap-1.5 px-4 py-2 rounded-full text-xs font-semibold bg-primary/20 text-primary border border-primary/30 hover:bg-primary/30 transition-colors"
                            >
                                <UserPlus size={15} /> Register
                            </NavLink>
                        </div>
                    )}
                </div>

                {/* Mobile Menu Toggle */}
                <div className="md:hidden">
                    <button
                        id="mobile-menu-toggle"
                        onClick={() => setMenuOpen(prev => !prev)}
                        className="p-2 text-slate-300 hover:text-white glass-pill hover:bg-slate-800 transition-colors"
                        aria-label={menuOpen ? 'Close navigation menu' : 'Open navigation menu'}
                        aria-expanded={menuOpen}
                        aria-controls="mobile-nav-drawer"
                    >
                        {menuOpen ? <X size={24} /> : <Menu size={24} />}
                    </button>
                </div>
            </nav>

            {/* Mobile Navigation Drawer */}
            {menuOpen && (
                <>
                    {/* Backdrop */}
                    <div
                        className="fixed inset-0 bg-black/40 z-40 md:hidden"
                        onClick={closeMenu}
                        aria-hidden="true"
                    />
                    {/* Drawer */}
                    <nav
                        id="mobile-nav-drawer"
                        role="navigation"
                        aria-label="Mobile navigation"
                        className="fixed top-0 right-0 h-full w-64 bg-darkBg border-l border-white/10 z-50 flex flex-col pt-20 px-4 gap-2 md:hidden animate-slide-in-right"
                    >
                        <button
                            onClick={closeMenu}
                            className="absolute top-4 right-4 p-2 text-slate-400 hover:text-white"
                            aria-label="Close navigation menu"
                        >
                            <X size={20} />
                        </button>

                        <MobileNavItem to="/" icon={<Home size={18} />} label="Home" onClick={closeMenu} />
                        <MobileNavItem to="/diabetes" icon={<Activity size={18} />} label="Diabetes Scan" onClick={closeMenu} />
                        <MobileNavItem to="/heart" icon={<Heart size={18} />} label="Cardiac Scan" onClick={closeMenu} />
                        <MobileNavItem to="/dashboard" icon={<Activity size={18} />} label="Dashboard" onClick={closeMenu} />
                        <MobileNavItem to="/about" icon={<Info size={18} />} label="About" onClick={closeMenu} />

                        <div className="my-2 border-t border-slate-700" />

                        {isAuthenticated ? (
                            <div className="space-y-2">
                                <div className="text-xs text-slate-400 px-3">
                                    Signed in as <span className="text-white font-medium">{user?.email}</span> ({user?.role})
                                </div>
                                <button
                                    onClick={() => {
                                        signOut();
                                        closeMenu();
                                    }}
                                    className="w-full flex items-center gap-3 px-4 py-3 rounded-xl font-medium text-slate-300 hover:bg-white/5 hover:text-white text-sm"
                                >
                                    <LogOut size={18} /> Sign Out
                                </button>
                            </div>
                        ) : (
                            <div className="space-y-2">
                                <MobileNavItem to="/login" icon={<LogIn size={18} />} label="Sign In" onClick={closeMenu} />
                                <MobileNavItem to="/register" icon={<UserPlus size={18} />} label="Register" onClick={closeMenu} />
                            </div>
                        )}
                    </nav>
                </>
            )}
        </>
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

const MobileNavItem = ({ to, icon, label, onClick }) => {
    return (
        <NavLink
            to={to}
            onClick={onClick}
            className={({ isActive }) =>
                `flex items-center gap-3 px-4 py-3 rounded-xl transition-all font-medium ${isActive
                    ? 'bg-primary/20 text-primary border border-primary/30'
                    : 'text-slate-300 hover:bg-white/5 hover:text-white border border-transparent'
                }`
            }
        >
            {icon}
            <span className="text-sm">{label}</span>
        </NavLink>
    );
};

export default Navbar;
