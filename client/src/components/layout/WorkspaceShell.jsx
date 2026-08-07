import React, { useState, useEffect, useRef } from 'react';
import { NavLink, useLocation } from 'react-router-dom';
import {
  Activity,
  Heart,
  LineChart,
  Users,
  Cpu,
  Lock,
  ShieldCheck,
  BookOpen,
  Info,
  LogOut,
  Menu,
  X,
  UserCheck,
  ChevronRight,
} from 'lucide-react';
import { useAuth } from '../../context/useAuth';
import Footer from '../Footer';

const WorkspaceShell = ({ children, wide = false }) => {
  const [mobileDrawerOpen, setMobileDrawerOpen] = useState(false);
  const { user, signOut } = useAuth();
  const location = useLocation();

  const toggleBtnRef = useRef(null);
  const isClinicianOrAdmin = user?.role === 'CLINICIAN' || user?.role === 'ADMIN';

  const closeDrawer = () => {
    setMobileDrawerOpen(false);
    toggleBtnRef.current?.focus();
  };

  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === 'Escape' && mobileDrawerOpen) {
        closeDrawer();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [mobileDrawerOpen]);

  const getPageTitle = () => {
    switch (location.pathname) {
      case '/dashboard':
        return 'Patient Health Risk Dashboard';
      case '/diabetes':
        return 'Diabetes Risk Assessment';
      case '/heart':
        return 'Cardiac Health Assessment';
      case '/worklist':
        return 'Clinician Worklist & Triage';
      case '/models':
        return 'AI Model Registry & Performance';
      case '/audit':
        return 'Security Audit & Compliance';
      case '/system-health':
        return 'System Health & Infrastructure';
      case '/knowledge':
        return 'Medical Knowledge Assistant';
      case '/about':
        return 'Platform Methodology & Governance';
      default:
        return 'Workspace Environment';
    }
  };

  const navGroups = [
    {
      groupTitle: 'Overview & Assessments',
      items: [
        { to: '/dashboard', label: 'Patient Dashboard', icon: LineChart },
        { to: '/diabetes', label: 'Diabetes Risk Scan', icon: Activity, badge: 'Glycemic' },
        { to: '/heart', label: 'Cardiac Risk Scan', icon: Heart, badge: 'Cardio' },
      ],
    },
    {
      groupTitle: 'Intelligence & Research',
      items: [
        { to: '/knowledge', label: 'Medical Guidelines', icon: BookOpen },
      ],
    },
    ...(isClinicianOrAdmin
      ? [
          {
            groupTitle: 'Clinical & Administration',
            items: [
              { to: '/worklist', label: 'Clinician Worklist', icon: Users },
              { to: '/models', label: 'Model Registry', icon: Cpu },
              { to: '/audit', label: 'Security Audit', icon: Lock },
              { to: '/system-health', label: 'System Health', icon: ShieldCheck },
            ],
          },
        ]
      : []),
    {
      groupTitle: 'Platform',
      items: [{ to: '/about', label: 'Platform & Safety', icon: Info }],
    },
  ];

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col selection:bg-teal-500/30 selection:text-teal-200">
      <a href="#workspace-main-content" className="skip-to-content">
        Skip to main content
      </a>

      <div className="flex flex-1">
        {/* Desktop Sidebar (256px expanded / 72px collapsed rail) */}
        <aside
          className="hidden md:flex flex-col w-20 lg:w-64 bg-slate-900 border-r border-slate-800/90 h-screen sticky top-0 z-30 transition-all duration-300 select-none"
          aria-label="Workspace sidebar navigation"
        >
          {/* Rounded-square product icon tile */}
          <div className="h-16 px-4 lg:px-6 flex items-center justify-between border-b border-slate-800/90">
            <NavLink to="/" className="flex items-center gap-3 group focus-visible:ring-2 focus-visible:ring-teal-400 rounded-xl px-1">
              <div className="w-10 h-10 rounded-xl bg-teal-500/20 text-teal-400 border border-teal-500/30 flex items-center justify-center flex-shrink-0 shadow-inner">
                <Activity className="w-5 h-5" />
              </div>
              <span className="hidden lg:inline text-lg font-bold tracking-tight text-white">
                Health<span className="text-teal-400">Lens AI</span>
              </span>
            </NavLink>
          </div>

          {/* Navigation Items */}
          <div className="flex-1 overflow-y-auto py-6 px-3 lg:px-4 space-y-6">
            {navGroups.map((group, idx) => (
              <div key={idx} className="space-y-1">
                <h3 className="hidden lg:block text-[10px] font-bold text-slate-400 uppercase tracking-wider px-3 mb-2">
                  {group.groupTitle}
                </h3>
                {group.items.map((item) => {
                  const Icon = item.icon;
                  return (
                    <NavLink
                      key={item.to}
                      to={item.to}
                      title={item.label}
                      className={({ isActive }) =>
                        `flex items-center gap-3 px-3 py-2.5 rounded-xl transition-all duration-200 text-xs font-semibold relative ${
                          isActive
                            ? 'bg-slate-850 text-teal-400 border border-teal-500/40 shadow-sm shadow-teal-950/40 font-bold before:absolute before:left-0 before:top-2 before:bottom-2 before:w-1 before:bg-teal-400 before:rounded-r'
                            : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50 border border-transparent'
                        }`
                      }
                    >
                      <Icon className="w-4 h-4 flex-shrink-0" />
                      <span className="hidden lg:inline truncate">{item.label}</span>
                    </NavLink>
                  );
                })}
              </div>
            ))}
          </div>

          {/* Account Profile Footer */}
          <div className="p-3 lg:p-4 border-t border-slate-800/90 bg-slate-950/40">
            <div className="flex items-center justify-between">
              <div className="hidden lg:flex flex-col truncate pr-2">
                <span className="text-xs font-bold text-slate-200 truncate">{user?.email?.split('@')[0]}</span>
                <span className="text-[10px] text-teal-400 font-mono font-medium uppercase tracking-wide">
                  {user?.role || 'GUEST'}
                </span>
              </div>
              <button
                onClick={signOut}
                title="Sign Out"
                aria-label="Sign Out"
                className="p-2 rounded-xl text-slate-400 hover:text-coral-400 hover:bg-slate-800 transition-colors"
              >
                <LogOut className="w-4 h-4" />
              </button>
            </div>
          </div>
        </aside>

        {/* Workspace Main Body Wrapper */}
        <div className="flex-1 flex flex-col min-h-screen min-w-0">
          {/* Sticky Workspace Glass Header */}
          <header className="sticky top-0 z-20 h-16 bg-slate-900/80 backdrop-blur-xl border-b border-slate-800/80 px-4 md:px-8 flex items-center justify-between">
            <div className="flex items-center gap-3">
              {/* Mobile Drawer Trigger */}
              <button
                ref={toggleBtnRef}
                onClick={() => setMobileDrawerOpen(true)}
                className="md:hidden p-2 text-slate-300 hover:text-white rounded-xl bg-slate-850 border border-slate-700/60 min-w-[44px] min-h-[44px] flex items-center justify-center"
                aria-label="Open workspace menu"
                aria-expanded={mobileDrawerOpen}
                aria-controls="workspace-mobile-drawer"
              >
                <Menu className="w-5 h-5" />
              </button>

              <div className="flex items-center gap-2 text-xs text-slate-400 font-medium hidden sm:flex">
                <span>Workspace</span>
                <ChevronRight className="w-3.5 h-3.5 text-slate-600" />
                <h2 className="text-sm font-bold text-slate-100">{getPageTitle()}</h2>
              </div>
              <h2 className="text-sm font-bold text-slate-100 sm:hidden truncate">{getPageTitle()}</h2>
            </div>

            {/* Account Info Pill */}
            <div className="flex items-center gap-3">
              {user && (
                <div className="flex items-center gap-2 px-3 py-1.5 rounded-xl bg-slate-900 border border-slate-800 text-xs">
                  <UserCheck className="w-3.5 h-3.5 text-teal-400" />
                  <span className="hidden sm:inline text-slate-300 font-medium">{user.email}</span>
                  <span className="px-2 py-0.5 text-[10px] font-bold rounded-full bg-teal-500/20 text-teal-300 border border-teal-500/30">
                    {user.role}
                  </span>
                </div>
              )}
              <button
                onClick={signOut}
                className="hidden sm:flex items-center gap-1.5 px-3.5 py-1.5 rounded-xl text-xs font-semibold text-slate-300 hover:text-coral-400 hover:bg-slate-900 border border-slate-800 transition-colors min-h-[38px]"
              >
                <LogOut className="w-3.5 h-3.5" /> Sign Out
              </button>
            </div>
          </header>

          {/* Mobile Drawer - Glass & Focus Trap */}
          {mobileDrawerOpen && (
            <>
              <div
                className="fixed inset-0 bg-slate-950/80 backdrop-blur-sm z-40 md:hidden"
                onClick={closeDrawer}
                aria-hidden="true"
              />
              <nav
                id="workspace-mobile-drawer"
                role="dialog"
                aria-modal="true"
                aria-label="Workspace mobile navigation drawer"
                className="fixed top-0 left-0 h-full w-72 bg-slate-900/90 backdrop-blur-2xl border-r border-slate-800 z-50 p-6 flex flex-col justify-between md:hidden"
              >
                <div>
                  <div className="flex items-center justify-between pb-6 border-b border-slate-800">
                    <div className="flex items-center gap-2">
                      <div className="w-8 h-8 rounded-xl bg-teal-500/20 text-teal-400 border border-teal-500/30 flex items-center justify-center">
                        <Activity className="w-4 h-4" />
                      </div>
                      <span className="font-bold text-white text-base">HealthLens AI</span>
                    </div>
                    <button
                      onClick={closeDrawer}
                      className="p-2 text-slate-400 hover:text-white rounded-xl"
                      aria-label="Close navigation menu"
                    >
                      <X className="w-5 h-5" />
                    </button>
                  </div>

                  <div className="py-6 space-y-6 overflow-y-auto max-h-[calc(100vh-200px)]">
                    {navGroups.map((group, gIdx) => (
                      <div key={gIdx} className="space-y-1">
                        <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider px-2 block mb-2">
                          {group.groupTitle}
                        </span>
                        {group.items.map((item) => {
                          const Icon = item.icon;
                          return (
                            <NavLink
                              key={item.to}
                              to={item.to}
                              onClick={closeDrawer}
                              className={({ isActive }) =>
                                `flex items-center gap-3 px-3 py-2.5 rounded-xl text-xs font-semibold ${
                                  isActive
                                    ? 'bg-slate-850 text-teal-400 border border-teal-500/30'
                                    : 'text-slate-300 hover:bg-slate-800/50'
                                }`
                              }
                            >
                              <Icon className="w-4 h-4 text-teal-400" />
                              <span>{item.label}</span>
                            </NavLink>
                          );
                        })}
                      </div>
                    ))}
                  </div>
                </div>

                <div className="pt-6 border-t border-slate-800 space-y-3">
                  <div className="text-xs text-slate-400 px-1">
                    Signed in as <span className="text-slate-100 font-semibold">{user?.email}</span>
                  </div>
                  <button
                    onClick={() => {
                      closeDrawer();
                      signOut();
                    }}
                    className="w-full flex items-center justify-center gap-2 py-2.5 bg-slate-850 hover:bg-slate-800 text-coral-300 font-semibold rounded-xl text-xs border border-slate-700"
                  >
                    <LogOut className="w-4 h-4" /> Sign Out
                  </button>
                </div>
              </nav>
            </>
          )}

          {/* Main Page Workspace Content */}
          <main
            id="workspace-main-content"
            className={`flex-1 w-full mx-auto px-4 md:px-8 py-8 ${wide ? 'max-w-7xl' : 'max-w-5xl'}`}
          >
            {children}
          </main>

          <Footer />
        </div>
      </div>
    </div>
  );
};

export default WorkspaceShell;
