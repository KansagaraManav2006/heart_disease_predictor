import React, { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import Surface from '../../components/Surface';
import InputField from '../../components/InputField';
import SelectField from '../../components/SelectField';
import Button from '../../components/Button';
import ErrorState from '../../components/ErrorState';
import StatusBadge from '../../components/StatusBadge';
import { useAuth } from '../../context/useAuth';
import { UserPlus, ShieldCheck, CheckCircle2, ArrowRight } from 'lucide-react';

const Register = () => {
  const [fullName, setFullName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [role, setRole] = useState('PATIENT');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [devToken, setDevToken] = useState('');
  const [successMessage, setSuccessMessage] = useState('');

  const { signUp } = useAuth();
  const navigate = useNavigate();

  const roleOptions = [
    { value: 'PATIENT', label: 'Patient / Research Subject' },
    { value: 'CLINICIAN', label: 'Clinician / Evaluator' },
  ];

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setSuccessMessage('');
    setLoading(true);

    try {
      const res = await signUp({ fullName, email, password, role });
      setSuccessMessage(res.message);
      if (res.devVerifyToken) {
        setDevToken(res.devVerifyToken);
      }
    } catch (err) {
      setError(err.message || 'Registration failed. Please review input fields and try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-md mx-auto py-8 animate-fade-in">
      <div className="text-center mb-8">
        <div className="w-14 h-14 rounded-2xl bg-slate-900 border border-slate-800 text-teal-400 flex items-center justify-center mx-auto mb-4 shadow-inner">
          <UserPlus className="w-6 h-6" />
        </div>
        <h1 className="text-2xl font-black text-slate-100 tracking-tight">Create Research Account</h1>
        <p className="text-xs text-slate-400 mt-1">Register to access risk screening decision support</p>
      </div>

      <Surface variant="raised" accent="teal">
        {error && (
          <ErrorState
            title="Registration Failed"
            message={error}
            className="mb-6"
          />
        )}

        {successMessage ? (
          <div className="space-y-4 text-center py-4">
            <div className="w-12 h-12 bg-teal-500/20 text-teal-400 rounded-full flex items-center justify-center mx-auto border border-teal-500/30">
              <CheckCircle2 className="w-6 h-6" />
            </div>
            <h3 className="text-lg font-bold text-slate-100">Registration Complete</h3>
            <p className="text-xs text-slate-300">{successMessage}</p>

            {devToken && (
              <div className="bg-slate-950 p-4 rounded-xl text-left border border-slate-800 text-xs text-slate-300 space-y-1 font-mono">
                <span className="font-bold text-teal-400 block font-sans">Development Verification Token:</span>
                <p className="break-all">{devToken}</p>
              </div>
            )}

            <div className="pt-4">
              <Button
                onClick={() => navigate('/login')}
                variant="primary"
                fullWidth
                icon={ArrowRight}
                iconPosition="right"
              >
                Proceed to Sign In
              </Button>
            </div>
          </div>
        ) : (
          <form onSubmit={handleSubmit} className="space-y-4">
            <div className="space-y-3 pb-3 border-b border-slate-800">
              <span className="text-[11px] font-bold text-teal-400 uppercase tracking-wider block">
                Account Identity
              </span>
              <InputField
                label="Full Name"
                name="fullName"
                value={fullName}
                onChange={(e) => setFullName(e.target.value)}
                placeholder="Dr. Jane Doe"
                required
              />

              <InputField
                label="Email Address"
                name="email"
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="user@hospital.org"
                required
              />

              <InputField
                label="Password (min. 8 characters)"
                name="password"
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="••••••••"
                required
              />
            </div>

            <div className="space-y-3 pt-2">
              <span className="text-[11px] font-bold text-amber-400 uppercase tracking-wider block">
                Role &amp; Clinical Context
              </span>
              <SelectField
                label="Account Role"
                name="role"
                value={role}
                onChange={(e) => setRole(e.target.value)}
                options={roleOptions}
              />
            </div>

            <Button
              type="submit"
              disabled={loading}
              loading={loading}
              loadingLabel="Creating account..."
              variant="primary"
              fullWidth
              icon={ShieldCheck}
              className="mt-4 font-bold"
            >
              Create Account
            </Button>
          </form>
        )}

        {!successMessage && (
          <div className="mt-6 pt-4 border-t border-slate-800 text-center text-xs text-slate-400">
            Already have an account?{' '}
            <Link to="/login" className="text-teal-400 hover:underline font-semibold">
              Sign in here
            </Link>
          </div>
        )}
      </Surface>
    </div>
  );
};

export default Register;
