import React from 'react';
import { NavLink, Link } from 'react-router-dom';
import { Activity, LogIn, UserPlus, LogOut, UserCheck } from 'lucide-react';
import { useAuth } from '../context/useAuth';

const Navbar = () => {
  const { user, isAuthenticated, signOut } = useAuth();

  return (
    <nav className="bg-slate-950/80 backdrop-blur-md border-b border-slate-800 px-4 md:px-8 py-3.5 flex items-center justify-between sticky top-0 z-50">
      <Link to="/" className="flex items-center gap-3">
        <div className="bg-teal-600/20 p-2 rounded-xl text-teal-400 border border-teal-500/30">
          <Activity className="w-5 h-5" />
        </div>
        <span className="text-lg font-black text-white">
          Health<span className="text-teal-400">Lens AI</span>
        </span>
      </Link>

      <div className="flex items-center gap-4">
        {isAuthenticated ? (
          <div className="flex items-center gap-3">
            <span className="text-xs font-semibold px-2.5 py-1 rounded-full bg-teal-500/20 text-teal-300 border border-teal-500/30">
              <UserCheck className="w-3.5 h-3.5 inline mr-1" />
              {user?.role} ({user?.email?.split('@')[0]})
            </span>
            <button
              onClick={signOut}
              className="text-xs text-slate-300 hover:text-coral-400 px-3 py-1.5 rounded-lg border border-slate-800 hover:bg-slate-900 transition-colors"
            >
              Sign Out
            </button>
          </div>
        ) : (
          <div className="flex items-center gap-2">
            <Link
              to="/login"
              className="px-3.5 py-2 rounded-xl text-xs font-semibold text-slate-300 hover:text-white"
            >
              Sign In
            </Link>
            <Link
              to="/register"
              className="px-4 py-2 rounded-xl text-xs font-bold text-slate-950 bg-teal-400 hover:bg-teal-300"
            >
              Register
            </Link>
          </div>
        )}
      </div>
    </nav>
  );
};

export default Navbar;
