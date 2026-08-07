import React, { useState, useEffect, useRef } from 'react';
import { NavLink, Link, useNavigate } from 'react-router-dom';
import { Activity, LogIn, UserPlus, Menu, X, Info, Shield, BookOpen } from 'lucide-react';
import { useAuth } from '../../context/useAuth';
import Footer from '../Footer';

const PublicShell = ({ children }) => {
  const [mobileOpen, setMobileOpen] = useState(false);
  const { isAuthenticated, user } = useAuth();
  const navigate = useNavigate();
  const drawerRef = useRef(null);
  const toggleBtnRef = useRef(null);

  const closeDrawer = () => {
    setMobileOpen(false);
    toggleBtnRef.current?.focus();
  };

  // Keyboard Escape listener and Focus Trapping for mobile drawer
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === 'Escape' && mobileOpen) {
        closeDrawer();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [mobileOpen]);

  return (
    <div className="min-h-screen bg-canvas text-slate-100 flex flex-col selection:bg-teal-500/30 selection:text-teal-200">
      {/* Accessibility Skip Link */}
      <a href="#main-public-content" className="skip-to-content">
        Skip to main content
      </a>

      {/* Public Header */}
      <header className="sticky top-0 z-40 bg-slate-950/80 backdrop-blur-md border-b border-slate-800/80 px-4 md:px-8 py-3.5 flex items-center justify-between">
        <Link to="/" className="flex items-center gap-3 group focus-visible:ring-2 focus-visible:ring-teal-400 rounded-xl px-1 py-0.5">
          <div className="bg-teal-600/20 p-2 rounded-xl text-teal-400 border border-teal-500/30 group-hover:bg-teal-600/30 transition-colors">
            <Activity className="w-5 h-5" />
          </div>
          <span className="text-lg font-black tracking-tight text-white">
            Health<span className="text-teal-400">Lens AI</span>
          </span>
        </Link>

        {/* Desktop Navigation */}
        <nav className="hidden md:flex items-center gap-6" aria-label="Public main navigation">
          <NavLink
            to="/"
            className={({ isActive }) =>
              `text-sm font-medium transition-colors ${isActive ? 'text-teal-400 font-semibold' : 'text-slate-300 hover:text-white'}`
            }
          >
            Home
          </NavLink>
          <NavLink
            to="/about"
            className={({ isActive }) =>
              `text-sm font-medium transition-colors ${isActive ? 'text-teal-400 font-semibold' : 'text-slate-300 hover:text-white'}`
            }
          >
            Methodology &amp; About
          </NavLink>

          <div className="h-4 w-[1px] bg-slate-800" />

          {isAuthenticated ? (
            <button
              onClick={() => navigate('/dashboard')}
              className="px-4 py-2 text-xs font-bold text-slate-950 bg-teal-400 hover:bg-teal-300 rounded-xl transition-all shadow-md shadow-teal-950/30 flex items-center gap-2"
            >
              Workspace Dashboard ({user?.role})
            </button>
          ) : (
            <div className="flex items-center gap-3">
              <Link
                to="/login"
                className="flex items-center gap-1.5 px-4 py-2 rounded-xl text-xs font-semibold text-slate-300 hover:text-white hover:bg-slate-800/80 transition-colors border border-transparent"
              >
                <LogIn className="w-4 h-4" /> Sign In
              </Link>
              <Link
                to="/register"
                className="flex items-center gap-1.5 px-4 py-2 rounded-xl text-xs font-bold text-slate-950 bg-teal-400 hover:bg-teal-300 transition-all shadow-md shadow-teal-950/30"
              >
                <UserPlus className="w-4 h-4" /> Register
              </Link>
            </div>
          )}
        </nav>

        {/* Mobile Toggle Button */}
        <button
          ref={toggleBtnRef}
          onClick={() => setMobileOpen((prev) => !prev)}
          className="md:hidden p-2 text-slate-300 hover:text-white rounded-xl bg-slate-800/80 border border-slate-700/60"
          aria-label={mobileOpen ? 'Close main menu' : 'Open main menu'}
          aria-expanded={mobileOpen}
          aria-controls="public-mobile-drawer"
        >
          {mobileOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
        </button>
      </header>

      {/* Mobile Drawer Backdrop & Menu */}
      {mobileOpen && (
        <>
          <div
            className="fixed inset-0 bg-slate-950/80 backdrop-blur-sm z-40 md:hidden"
            onClick={closeDrawer}
            aria-hidden="true"
          />
          <div
            id="public-mobile-drawer"
            ref={drawerRef}
            role="dialog"
            aria-modal="true"
            aria-label="Public mobile navigation menu"
            className="fixed top-0 right-0 h-full w-72 bg-slate-900 border-l border-slate-800 z-50 p-6 flex flex-col justify-between md:hidden"
          >
            <div>
              <div className="flex items-center justify-between pb-6 border-b border-slate-800">
                <span className="font-bold text-slate-200 text-sm">HealthLens Navigation</span>
                <button
                  onClick={closeDrawer}
                  className="p-2 text-slate-400 hover:text-white rounded-lg"
                  aria-label="Close menu"
                >
                  <X className="w-5 h-5" />
                </button>
              </div>

              <div className="flex flex-col gap-3 py-6">
                <NavLink
                  to="/"
                  onClick={closeDrawer}
                  className="flex items-center gap-3 p-3 rounded-xl text-sm font-semibold text-slate-200 hover:bg-slate-800"
                >
                  <Activity className="w-4 h-4 text-teal-400" /> Home Overview
                </NavLink>
                <NavLink
                  to="/about"
                  onClick={closeDrawer}
                  className="flex items-center gap-3 p-3 rounded-xl text-sm font-semibold text-slate-200 hover:bg-slate-800"
                >
                  <Info className="w-4 h-4 text-teal-400" /> Methodology &amp; Platform
                </NavLink>
              </div>
            </div>

            <div className="pt-6 border-t border-slate-800 space-y-3">
              {isAuthenticated ? (
                <button
                  onClick={() => {
                    closeDrawer();
                    navigate('/dashboard');
                  }}
                  className="w-full py-3 bg-teal-400 text-slate-950 font-bold rounded-xl text-sm shadow-md"
                >
                  Enter Workspace
                </button>
              ) : (
                <>
                  <Link
                    to="/login"
                    onClick={closeDrawer}
                    className="w-full flex items-center justify-center gap-2 py-3 bg-slate-800 text-slate-100 font-semibold rounded-xl text-sm border border-slate-700"
                  >
                    <LogIn className="w-4 h-4" /> Sign In
                  </Link>
                  <Link
                    to="/register"
                    onClick={closeDrawer}
                    className="w-full flex items-center justify-center gap-2 py-3 bg-teal-400 text-slate-950 font-bold rounded-xl text-sm"
                  >
                    <UserPlus className="w-4 h-4" /> Register
                  </Link>
                </>
              )}
            </div>
          </div>
        </>
      )}

      {/* Main Content Viewport */}
      <main id="main-public-content" className="flex-1 w-full max-w-6xl mx-auto px-4 md:px-8 py-8">
        {children}
      </main>

      <Footer />
    </div>
  );
};

export default PublicShell;
