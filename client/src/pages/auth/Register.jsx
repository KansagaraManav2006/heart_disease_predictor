import React, { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import GlassCard from '../../components/GlassCard';
import InputField from '../../components/InputField';
import SelectField from '../../components/SelectField';
import Button from '../../components/Button';
import { useAuth } from '../../context/useAuth';
import { UserPlus, ShieldCheck, AlertCircle, CheckCircle2 } from 'lucide-react';

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
            setError(err.message || 'Registration failed. Please try again.');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="max-w-md mx-auto animate-fade-in-up py-12">
            <div className="text-center mb-8">
                <div className="bg-blue-500/10 w-16 h-16 rounded-2xl flex items-center justify-center mx-auto mb-4 text-blue-600 border border-blue-200">
                    <UserPlus size={32} />
                </div>
                <h1 className="text-3xl font-black text-slate-800">Create Account</h1>
                <p className="text-slate-600 text-sm mt-1">Join the HealthLens AI Research Pilot</p>
            </div>

            <GlassCard>
                {error && (
                    <div className="flex items-center gap-3 bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-xl mb-6 text-sm">
                        <AlertCircle size={18} className="flex-shrink-0" />
                        <span>{error}</span>
                    </div>
                )}

                {successMessage ? (
                    <div className="space-y-4 text-center py-4">
                        <div className="w-12 h-12 bg-green-100 text-green-600 rounded-full flex items-center justify-center mx-auto">
                            <CheckCircle2 size={28} />
                        </div>
                        <h3 className="text-xl font-bold text-slate-800">Registration Received!</h3>
                        <p className="text-sm text-slate-600">{successMessage}</p>

                        {devToken && (
                            <div className="bg-slate-100 p-4 rounded-xl text-left border border-slate-200 text-xs text-slate-700 space-y-1">
                                <span className="font-bold text-slate-900 block">Development Quick-Verify:</span>
                                <p className="break-all font-mono">Token: {devToken}</p>
                            </div>
                        )}

                        <div className="pt-4">
                            <Button
                                onClick={() => navigate('/login')}
                                className="w-full bg-blue-600 hover:bg-blue-700 text-white font-bold py-2.5 rounded-xl"
                            >
                                Proceed to Sign In
                            </Button>
                        </div>
                    </div>
                ) : (
                    <form onSubmit={handleSubmit} className="space-y-4">
                        <InputField
                            label="Full Name"
                            name="fullName"
                            value={fullName}
                            onChange={(e) => setFullName(e.target.value)}
                            placeholder="Dr. Jane Doe or John Smith"
                            required
                        />

                        <InputField
                            label="Email Address"
                            name="email"
                            type="email"
                            value={email}
                            onChange={(e) => setEmail(e.target.value)}
                            placeholder="user@example.com"
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

                        <SelectField
                            label="Account Role"
                            name="role"
                            value={role}
                            onChange={(e) => setRole(e.target.value)}
                            options={roleOptions}
                        />

                        <Button
                            type="submit"
                            disabled={loading}
                            className="w-full flex items-center justify-center gap-2 py-3 bg-blue-600 hover:bg-blue-700 text-white rounded-xl font-bold shadow-md mt-2"
                        >
                            <ShieldCheck size={18} />
                            {loading ? 'Creating account...' : 'Create Account'}
                        </Button>
                    </form>
                )}

                {!successMessage && (
                    <div className="mt-6 text-center text-sm text-slate-600 pt-4 border-t border-slate-200">
                        Already have an account?{' '}
                        <Link to="/login" className="text-blue-600 hover:underline font-semibold">
                            Sign in here
                        </Link>
                    </div>
                )}
            </GlassCard>
        </div>
    );
};

export default Register;
