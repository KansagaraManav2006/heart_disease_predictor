import React, { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import GlassCard from '../../components/GlassCard';
import InputField from '../../components/InputField';
import Button from '../../components/Button';
import { useAuth } from '../../context/useAuth';
import { LogIn, Lock, AlertCircle } from 'lucide-react';

const SignIn = () => {
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
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
            navigate('/');
        } catch (err) {
            setError(err.message || 'Failed to sign in. Please check your credentials.');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="max-w-md mx-auto animate-fade-in-up py-12">
            <div className="text-center mb-8">
                <div className="bg-blue-500/10 w-16 h-16 rounded-2xl flex items-center justify-center mx-auto mb-4 text-blue-600 border border-blue-200">
                    <Lock size={32} />
                </div>
                <h1 className="text-3xl font-black text-slate-800">Sign In</h1>
                <p className="text-slate-600 text-sm mt-1">Access your HealthLens AI research dashboard</p>
            </div>

            <GlassCard>
                {error && (
                    <div className="flex items-center gap-3 bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-xl mb-6 text-sm">
                        <AlertCircle size={18} className="flex-shrink-0" />
                        <span>{error}</span>
                    </div>
                )}

                <form onSubmit={handleSubmit} className="space-y-5">
                    <InputField
                        label="Email Address"
                        name="email"
                        type="email"
                        value={email}
                        onChange={(e) => setEmail(e.target.value)}
                        placeholder="patient@example.com"
                        required
                    />

                    <InputField
                        label="Password"
                        name="password"
                        type="password"
                        value={password}
                        onChange={(e) => setPassword(e.target.value)}
                        placeholder="••••••••"
                        required
                    />

                    <Button
                        type="submit"
                        disabled={loading}
                        className="w-full flex items-center justify-center gap-2 py-3 bg-blue-600 hover:bg-blue-700 text-white rounded-xl font-bold shadow-md"
                    >
                        <LogIn size={18} />
                        {loading ? 'Signing in...' : 'Sign In'}
                    </Button>
                </form>

                <div className="mt-6 text-center text-sm text-slate-600 pt-4 border-t border-slate-200">
                    Don't have an account?{' '}
                    <Link to="/register" className="text-blue-600 hover:underline font-semibold">
                        Register here
                    </Link>
                </div>
            </GlassCard>
        </div>
    );
};

export default SignIn;
