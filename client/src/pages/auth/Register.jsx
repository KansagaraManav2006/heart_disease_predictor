import React, { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import Surface from '../../components/Surface';
import Button from '../../components/Button';
import InputField from '../../components/InputField';
import SelectField from '../../components/SelectField';
import ErrorState from '../../components/ErrorState';
import { useAuth } from '../../context/useAuth';
import { UserPlus, Activity, ShieldCheck, Eye, EyeOff } from 'lucide-react';

const Register = () => {
  const [fullName, setFullName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [role, setRole] = useState('PATIENT');
  const [showPassword, setShowPassword] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const { register } = useAuth();
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    try {
      await register({ email, password, fullName, role });
      navigate('/dashboard');
    } catch (err) {
      setError(err.message || 'Registration failed. Please check form details.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-md mx-auto py-8 md:py-12 animate-fade-in">
      <div className="text-center mb-8">
        <div className="w-12 h-12 rounded-xl bg-teal-500/20 text-teal-400 border border-teal-500/30 flex items-center justify-center mx-auto mb-4 shadow-inner">
          <Activity className="w-6 h-6" />
        </div>
        <h1 className="text-2xl md:text-3xl font-bold text-slate-100 mb-2">Create HealthLens Account</h1>
        <p className="text-xs md:text-sm text-slate-400">
          Register to run calibrated risk assessments and save session history.
        </p>
      </div>

      <Surface variant="raised" className="p-6 md:p-8">
        {error && <ErrorState title="Registration Error" message={error} className="mb-6" />}

        <form onSubmit={handleSubmit} className="space-y-4">
          <InputField
            label="Full Name"
            type="text"
            name="fullName"
            value={fullName}
            onChange={(e) => setFullName(e.target.value)}
            placeholder="e.g. Dr. Jane Smith"
            required
          />

          <InputField
            label="Email Address"
            type="email"
            name="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            placeholder="e.g. j.smith@hospital.org"
            required
            autoComplete="email"
          />

          <div className="relative">
            <InputField
              label="Password"
              type={showPassword ? 'text' : 'password'}
              name="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="At least 8 characters..."
              required
              autoComplete="new-password"
            />
            <button
              type="button"
              onClick={() => setShowPassword(!showPassword)}
              className="absolute right-3 top-9 text-slate-400 hover:text-slate-200 p-1 text-xs font-semibold"
              aria-label={showPassword ? 'Hide password' : 'Show password'}
            >
              {showPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
            </button>
          </div>

          <SelectField
            label="Account Role &amp; Access Level"
            name="role"
            value={role}
            onChange={(e) => setRole(e.target.value)}
            options={[
              { value: 'PATIENT', label: 'Patient (Personal Assessments)' },
              { value: 'CLINICIAN', label: 'Clinician (Patient Worklist & Triage)' },
              { value: 'ADMIN', label: 'Administrator (Model MLOps & Audit)' },
            ]}
            required
          />

          <Button
            type="submit"
            variant="primary"
            fullWidth
            loading={loading}
            loadingLabel="Creating Account..."
            icon={UserPlus}
            className="mt-6"
          >
            Create Account
          </Button>
        </form>

        <div className="mt-6 pt-6 border-t border-slate-800 text-center text-xs text-slate-400">
          Already registered?{' '}
          <Link to="/login" className="text-teal-400 font-semibold hover:underline">
            Sign in
          </Link>
        </div>
      </Surface>

      <div className="mt-6 text-center flex items-center justify-center gap-2 text-xs text-slate-400">
        <ShieldCheck className="w-4 h-4 text-teal-400" />
        <span>Role-Based Access Control &amp; Secure JWT Storage</span>
      </div>
    </div>
  );
};

export default Register;
