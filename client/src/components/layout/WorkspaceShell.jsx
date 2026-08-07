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
      groupTitle: 'Overview & Scans',
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
    <div className="min-h-screen bg-background text-foreground flex flex-col selection:bg-primary/30 selection:text-primary">
      <a href="#workspace-main-content" className="skip-to-content">
        Skip to main content
      </a>

      <div className="flex flex-1">
        {/* Expanded Desktop Sidebar / Collapsible Icon Rail */}
        <aside
          className="hidden md:flex flex-col w-20 lg:w-64 bg-sidebar border-r border-sidebar-border h-screen sticky top-0 z-30 transition-all duration-300 select-none"
          aria-label="Workspace sidebar navigation"
        >
          {/* Workspace Brand Header */}
          <div className="h-16 px-4 lg:px-6 flex items-center justify-between border-b border-sidebar-border">
            <NavLink to="/" className="flex items-center gap-3 group focus-visible:ring-2 focus-visible:ring-sidebar-ring rounded-md px-1">
              <div className="bg-primary/10 p-2 rounded-md text-primary border border-primary/30">
                <Activity className="w-5 h-5" />
              </div>
              <span className="hidden lg:inline text-lg font-black tracking-tight text-sidebar-foreground font-serif">
                Health<span className="text-primary">Lens AI</span>
              </span>
            </NavLink>
          </div>

          {/* Navigation Items */}
          <div className="flex-1 overflow-y-auto py-6 px-3 lg:px-4 space-y-6">
            {navGroups.map((group, idx) => (
              <div key={idx} className="space-y-1">
                <h3 className="hidden lg:block text-[10px] font-bold text-muted-foreground uppercase tracking-wider px-3 mb-2">
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
                        `flex items-center gap-3 px-3 py-2.5 rounded-md transition-all duration-200 text-xs font-semibold ${
                          isActive
                            ? 'bg-card text-primary border border-primary/40 shadow-sm'
                            : 'text-muted-foreground hover:text-foreground hover:bg-card/50 border border-transparent'
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
          <div className="p-3 lg:p-4 border-t border-sidebar-border bg-card/40">
            <div className="flex items-center justify-between">
              <div className="hidden lg:flex flex-col truncate pr-2">
                <span className="text-xs font-bold text-sidebar-foreground truncate">{user?.email?.split('@')[0]}</span>
                <span className="text-[10px] text-primary font-mono font-medium uppercase tracking-wide">
                  {user?.role || 'GUEST'}
                </span>
              </div>
              <button
                onClick={signOut}
                title="Sign Out"
                aria-label="Sign Out"
                className="p-2 rounded-md text-muted-foreground hover:text-destructive hover:bg-muted transition-colors"
              >
                <LogOut className="w-4 h-4" />
              </button>
            </div>
          </div>
        </aside>

        {/* Workspace Main Body Wrapper */}
        <div className="flex-1 flex flex-col min-h-screen min-w-0">
          {/* Sticky Workspace Utility Header */}
          <header className="sticky top-0 z-20 h-16 bg-card/90 backdrop-blur-md border-b border-border px-4 md:px-8 flex items-center justify-between">
            <div className="flex items-center gap-3">
              {/* Mobile Drawer Trigger */}
              <button
                ref={toggleBtnRef}
                onClick={() => setMobileDrawerOpen(true)}
                className="md:hidden p-2 text-foreground rounded-md bg-muted border border-border"
                aria-label="Open workspace menu"
                aria-expanded={mobileDrawerOpen}
                aria-controls="workspace-mobile-drawer"
              >
                <Menu className="w-5 h-5" />
              </button>

              <div className="flex items-center gap-2 text-xs text-muted-foreground font-medium hidden sm:flex">
                <span>Workspace</span>
                <ChevronRight className="w-3.5 h-3.5 text-muted-foreground/60" />
                <h2 className="text-sm font-bold text-foreground">{getPageTitle()}</h2>
              </div>
              <h2 className="text-sm font-bold text-foreground sm:hidden truncate">{getPageTitle()}</h2>
            </div>

            {/* Account Info Pill */}
            <div className="flex items-center gap-3">
              {user && (
                <div className="flex items-center gap-2 px-3 py-1.5 rounded-md bg-muted border border-border text-xs">
                  <UserCheck className="w-3.5 h-3.5 text-primary" />
                  <span className="hidden sm:inline text-foreground font-medium">{user.email}</span>
                  <span className="px-1.5 py-0.5 text-[10px] font-bold rounded bg-primary/10 text-primary border border-primary/20">
                    {user.role}
                  </span>
                </div>
              )}
              <button
                onClick={signOut}
                className="hidden sm:flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-semibold text-muted-foreground hover:text-destructive hover:bg-muted border border-border transition-colors"
              >
                <LogOut className="w-3.5 h-3.5" /> Sign Out
              </button>
            </div>
          </header>

          {/* Mobile Drawer */}
          {mobileDrawerOpen && (
            <>
              <div
                className="fixed inset-0 bg-black/60 backdrop-blur-sm z-40 md:hidden"
                onClick={closeDrawer}
                aria-hidden="true"
              />
              <nav
                id="workspace-mobile-drawer"
                role="dialog"
                aria-modal="true"
                aria-label="Workspace mobile navigation drawer"
                className="fixed top-0 left-0 h-full w-72 bg-sidebar border-r border-sidebar-border z-50 p-6 flex flex-col justify-between md:hidden"
              >
                <div>
                  <div className="flex items-center justify-between pb-6 border-b border-sidebar-border">
                    <div className="flex items-center gap-2">
                      <Activity className="w-5 h-5 text-primary" />
                      <span className="font-bold text-sidebar-foreground text-base font-serif">HealthLens AI</span>
                    </div>
                    <button
                      onClick={closeDrawer}
                      className="p-2 text-muted-foreground hover:text-sidebar-foreground rounded-md"
                      aria-label="Close navigation menu"
                    >
                      <X className="w-5 h-5" />
                    </button>
                  </div>

                  <div className="py-6 space-y-6 overflow-y-auto max-h-[calc(100vh-200px)]">
                    {navGroups.map((group, gIdx) => (
                      <div key={gIdx} className="space-y-1">
                        <span className="text-[10px] font-bold text-muted-foreground uppercase tracking-wider px-2 block mb-2">
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
                                `flex items-center gap-3 px-3 py-2.5 rounded-md text-xs font-semibold ${
                                  isActive
                                    ? 'bg-card text-primary border border-primary/30'
                                    : 'text-sidebar-foreground hover:bg-card/50'
                                }`
                              }
                            >
                              <Icon className="w-4 h-4 text-primary" />
                              <span>{item.label}</span>
                            </NavLink>
                          );
                        })}
                      </div>
                    ))}
                  </div>
                </div>

                <div className="pt-6 border-t border-sidebar-border space-y-3">
                  <div className="text-xs text-muted-foreground px-1">
                    Signed in as <span className="text-sidebar-foreground font-semibold">{user?.email}</span>
                  </div>
                  <button
                    onClick={() => {
                      closeDrawer();
                      signOut();
                    }}
                    className="w-full flex items-center justify-center gap-2 py-2.5 bg-muted hover:bg-card text-destructive font-semibold rounded-md text-xs border border-border"
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
