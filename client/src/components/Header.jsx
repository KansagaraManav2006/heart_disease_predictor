import React from 'react';
import { Activity } from 'lucide-react';
import { useAuth } from '../context/useAuth';

const Header = () => {
  const { user } = useAuth();
  return (
    <header className="bg-slate-950/80 backdrop-blur-md border-b border-slate-800/80 h-16 flex items-center justify-between px-6 sticky top-0 z-30 w-full">
      <div className="flex items-center gap-3">
        <Activity className="w-5 h-5 text-teal-400" />
        <h2 className="text-sm font-bold text-slate-100">HealthLens AI Research Workspace</h2>
      </div>
      {user && (
        <span className="text-xs text-slate-400 font-mono">
          {user.email} ({user.role})
        </span>
      )}
    </header>
  );
};

export default Header;
