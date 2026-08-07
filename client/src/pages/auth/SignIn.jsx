import React, { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import Surface from '../../components/Surface';
import InputField from '../../components/InputField';
import Button from '../../components/Button';
import ErrorState from '../../components/ErrorState';
import { useAuth } from '../../context/useAuth';
import { LogIn, Lock, ShieldCheck, Eye, EyeOff } from 'lucide-react';

const SignIn = () => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const { signIn } = useAuth();
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    try {
      await signIn({ email, password });
      navigate('/dashboard');
    } catch (err) {
      setError(err.message || 'Failed to sign in. Please verify your email and password credentials.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-md mx-auto py-8 animate-fade-in">
      <div className="text-center mb-8">
        <div className="w-14 h-14 rounded-2xl bg-slate-900 border border-slate-800 text-teal-400 flex items-center justify-center mx-auto mb-4 shadow-inner">
          <Lock className="w-6 h-6" />
        </div>
        <h1 className="text-2xl font-black text-slate-100 tracking-tight">Sign In to HealthLens AI</h1>
        <p className="text-xs text-slate-400 mt-1">Access your clinical decision support &amp; research workspace</p>
      </div>

      <Surface variant="raised" accent="teal">
        {error && (
          <ErrorState
            title="Authentication Error"
            message={error}
            className="mb-6"
          />
        )}

        <form onSubmit={handleSubmit} className="space-y-4">
          <InputField
            label="Email Address"
            name="email"
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            placeholder="clinician@hospital.org"
            required
            helperText="Enter your registered account email"
          />

          <div className="relative">
            <InputField
              label="Password"
              name="password"
              type={showPassword ? 'text' : 'password'}
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="••••••••"
              required
            />
            <button
              type="button"
              onClick={() => setShowPassword((prev) => !prev)}
              aria-label={showPassword ? 'Hide password' : 'Show password'}
              className="absolute right-3 top-[34px] p-1 text-slate-400 hover:text-slate-200"
            >
              {showPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
            </button>
          </div>

          <Button
            type="submit"
            disabled={loading}
            loading={loading}
            loadingLabel="Authenticating..."
            variant="primary"
            fullWidth
            icon={LogIn}
            className="mt-2 font-bold"
          >
            Sign In
          </Button>
        </form>

        <div className="mt-6 pt-4 border-t border-slate-800 text-center text-xs text-slate-400">
          Need a research account?{' '}
          <Link to="/register" className="text-teal-400 hover:underline font-semibold">
            Create an account
          </Link>
        </div>
      </Surface>

      <div className="mt-6 p-4 rounded-xl bg-slate-900/60 border border-slate-800 text-center text-[11px] text-slate-400 flex items-center justify-center gap-2">
        <ShieldCheck className="w-4 h-4 text-teal-400 flex-shrink-0" />
        <span>Role-Based Access Control · Cryptographic Session Tokens</span>
      </div>
    </div>
  );
};

export default SignIn;
