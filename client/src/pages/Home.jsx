import React from 'react';
import { useNavigate } from 'react-router-dom';
import GlassCard from '../components/GlassCard';
import Button from '../components/Button';
import { Activity, Heart, Shield, Clock, FileText } from 'lucide-react';

const Home = () => {
    const navigate = useNavigate();

    return (
        <div className="max-w-6xl mx-auto animate-fade-in-up">
            {/* Hero Section */}
            <div className="text-center mb-16 mt-8">
                <h1 className="text-4xl md:text-6xl font-black mb-4 tracking-tight">
                    Advanced Health <span className="text-primary">Screening</span>
                </h1>
                <p className="text-xl text-slate-300 max-w-2xl mx-auto leading-relaxed">
                    AI-powered system using Machine Learning algorithms to predict the risk of various diseases based on your health parameters.
                </p>
            </div>

            {/* Primary Actions */}
            <div className="grid md:grid-cols-2 gap-8 mb-16">
                <GlassCard className="glass-hover flex flex-col h-full transform transition-all hover:scale-105 border-t-4 border-t-blue-500">
                    <div className="bg-blue-500/20 w-16 h-16 rounded-2xl flex items-center justify-center mb-6 text-blue-400">
                        <Activity size={32} />
                    </div>
                    <h2 className="text-2xl font-bold text-white mb-2">Diabetes Risk Scan</h2>
                    <p className="text-slate-400 mb-8 flex-grow">
                        Advanced glucose & metabolic analysis including HbA1c levels, BMI assessment, and genetic factor evaluation.
                    </p>
                    <Button
                        onClick={() => navigate('/diabetes')}
                        className="bg-blue-600 hover:bg-blue-500 shadow-blue-500/20"
                    >
                        Start Diabetes Scan
                    </Button>
                </GlassCard>

                <GlassCard className="glass-hover flex flex-col h-full transform transition-all hover:scale-105 border-t-4 border-t-red-500">
                    <div className="bg-red-500/20 w-16 h-16 rounded-2xl flex items-center justify-center mb-6 text-red-400">
                        <Heart size={32} />
                    </div>
                    <h2 className="text-2xl font-bold text-white mb-2">Cardiac Risk Scan</h2>
                    <p className="text-slate-400 mb-8 flex-grow">
                        Comprehensive cardiovascular health check evaluating blood pressure, cholesterol, and lifetstyle factors.
                    </p>
                    <Button
                        onClick={() => navigate('/heart')}
                        className="bg-red-600 hover:bg-red-500 shadow-red-500/20"
                    >
                        Start Cardiac Scan
                    </Button>
                </GlassCard>
            </div>

            {/* Features/Stats Section */}
            <div className="mb-16">
                <h2 className="text-3xl font-bold text-center mb-10 text-white">Why Use HealthPredict?</h2>
                <div className="grid md:grid-cols-3 gap-6">
                    <div className="glass p-6 text-center rounded-2xl">
                        <div className="bg-primary/20 w-12 h-12 rounded-full flex items-center justify-center mx-auto mb-4 text-primary">
                            <Shield size={24} />
                        </div>
                        <h3 className="text-lg font-bold text-white mb-2">Accurate Predictions</h3>
                        <p className="text-sm text-slate-400">Models trained on thousands of validated medical records.</p>
                    </div>
                    <div className="glass p-6 text-center rounded-2xl">
                        <div className="bg-primary/20 w-12 h-12 rounded-full flex items-center justify-center mx-auto mb-4 text-primary">
                            <Clock size={24} />
                        </div>
                        <h3 className="text-lg font-bold text-white mb-2">Instant Results</h3>
                        <p className="text-sm text-slate-400">Get your health risk assessment in seconds.</p>
                    </div>
                    <div className="glass p-6 text-center rounded-2xl">
                        <div className="bg-primary/20 w-12 h-12 rounded-full flex items-center justify-center mx-auto mb-4 text-primary">
                            <FileText size={24} />
                        </div>
                        <h3 className="text-lg font-bold text-white mb-2">Private & Secure</h3>
                        <p className="text-sm text-slate-400">All predictions are processed without storing personal data.</p>
                    </div>
                </div>
            </div>

            {/* Important Notice */}
            <GlassCard className="bg-yellow-500/10 border-yellow-500/30 text-center relative overflow-hidden">
                <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-yellow-400 to-orange-500"></div>
                <h3 className="text-xl font-bold text-yellow-500 mb-2">⚠️ Educational Purposes Only</h3>
                <p className="text-slate-300 max-w-3xl mx-auto">
                    This system provides statistical predictions based on machine learning models. It is <strong>NOT</strong> a substitute for professional medical advice, diagnosis, or treatment. Always consult qualified healthcare providers.
                </p>
            </GlassCard>
        </div>
    );
};

export default Home;
