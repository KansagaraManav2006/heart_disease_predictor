import React, { useState, useEffect, useRef } from 'react';
import { NavLink, Link, useNavigate } from 'react-router-dom';
import { Activity, LogIn, UserPlus, Menu, X, Info } from 'lucide-react';
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
    <div className="min-h-screen bg-background text-foreground flex flex-col selection:bg-primary/30 selection:text-primary">
      <a href="#main-public-content" className="skip-to-content">
        Skip to main content
      </a>

      {/* Public Header */}
      <header className="sticky top-0 z-40 bg-card/90 backdrop-blur-md border-b border-border px-4 md:px-8 py-3.5 flex items-center justify-between">
        <Link to="/" className="flex items-center gap-3 group focus-visible:ring-2 focus-visible:ring-ring rounded-md px-1 py-0.5">
          <div className="bg-primary/10 p-2 rounded-md text-primary border border-primary/30 group-hover:bg-primary/20 transition-colors">
            <Activity className="w-5 h-5" />
          </div>
          <span className="text-lg font-bold tracking-tight text-foreground font-serif">
            Health<span className="text-primary">Lens AI</span>
          </span>
        </Link>

        {/* Desktop Navigation */}
        <nav className="hidden md:flex items-center gap-6" aria-label="Public main navigation">
          <NavLink
            to="/"
            className={({ isActive }) =>
              `text-sm font-medium transition-colors ${isActive ? 'text-primary font-semibold' : 'text-muted-foreground hover:text-foreground'}`
            }
          >
            Home
          </NavLink>
          <NavLink
            to="/about"
            className={({ isActive }) =>
              `text-sm font-medium transition-colors ${isActive ? 'text-primary font-semibold' : 'text-muted-foreground hover:text-foreground'}`
            }
          >
            Methodology &amp; About
          </NavLink>

          <div className="h-4 w-[1px] bg-border" />

          {isAuthenticated ? (
            <button
              onClick={() => navigate('/dashboard')}
              className="px-4 py-2 text-xs font-bold text-primary-foreground bg-primary hover:bg-red-700 rounded-md transition-all shadow-md flex items-center gap-2"
            >
              Workspace Dashboard ({user?.role})
            </button>
          ) : (
            <div className="flex items-center gap-3">
              <Link
                to="/login"
                className="flex items-center gap-1.5 px-4 py-2 rounded-md text-xs font-semibold text-foreground hover:bg-muted transition-colors border border-transparent"
              >
                <LogIn className="w-4 h-4" /> Sign In
              </Link>
              <Link
                to="/register"
                className="flex items-center gap-1.5 px-4 py-2 rounded-md text-xs font-bold text-primary-foreground bg-primary hover:bg-red-700 transition-all shadow-md"
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
          className="md:hidden p-2 text-foreground rounded-md bg-muted border border-border"
          aria-label={mobileOpen ? 'Close main menu' : 'Open main menu'}
          aria-expanded={mobileOpen}
          aria-controls="public-mobile-drawer"
        >
          {mobileOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
        </button>
      </header>

      {/* Mobile Drawer */}
      {mobileOpen && (
        <>
          <div
            className="fixed inset-0 bg-black/60 backdrop-blur-sm z-40 md:hidden"
            onClick={closeDrawer}
            aria-hidden="true"
          />
          <div
            id="public-mobile-drawer"
            ref={drawerRef}
            role="dialog"
            aria-modal="true"
            aria-label="Public mobile navigation menu"
            className="fixed top-0 right-0 h-full w-72 bg-card border-l border-border z-50 p-6 flex flex-col justify-between md:hidden"
          >
            <div>
              <div className="flex items-center justify-between pb-6 border-b border-border">
                <span className="font-bold text-foreground text-sm font-serif">HealthLens Navigation</span>
                <button
                  onClick={closeDrawer}
                  className="p-2 text-muted-foreground hover:text-foreground rounded-md"
                  aria-label="Close menu"
                >
                  <X className="w-5 h-5" />
                </button>
              </div>

              <div className="flex flex-col gap-3 py-6">
                <NavLink
                  to="/"
                  onClick={closeDrawer}
                  className="flex items-center gap-3 p-3 rounded-md text-sm font-semibold text-foreground hover:bg-muted"
                >
                  <Activity className="w-4 h-4 text-primary" /> Home Overview
                </NavLink>
                <NavLink
                  to="/about"
                  onClick={closeDrawer}
                  className="flex items-center gap-3 p-3 rounded-md text-sm font-semibold text-foreground hover:bg-muted"
                >
                  <Info className="w-4 h-4 text-primary" /> Methodology &amp; Platform
                </NavLink>
              </div>
            </div>

            <div className="pt-6 border-t border-border space-y-3">
              {isAuthenticated ? (
                <button
                  onClick={() => {
                    closeDrawer();
                    navigate('/dashboard');
                  }}
                  className="w-full py-3 bg-primary text-primary-foreground font-bold rounded-md text-sm shadow-md"
                >
                  Enter Workspace
                </button>
              ) : (
                <>
                  <Link
                    to="/login"
                    onClick={closeDrawer}
                    className="w-full flex items-center justify-center gap-2 py-3 bg-muted text-foreground font-semibold rounded-md text-sm border border-border"
                  >
                    <LogIn className="w-4 h-4" /> Sign In
                  </Link>
                  <Link
                    to="/register"
                    onClick={closeDrawer}
                    className="w-full flex items-center justify-center gap-2 py-3 bg-primary text-primary-foreground font-bold rounded-md text-sm"
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
