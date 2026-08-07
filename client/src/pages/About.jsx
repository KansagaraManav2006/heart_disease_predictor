import React from 'react';
import GlassCard from '../components/GlassCard';
import { Code, Database, Server, Component, AlertTriangle } from 'lucide-react';

const About = () => {
    return (
        <div className="max-w-4xl mx-auto animate-fade-in-up pb-12">
            <div className="text-center mb-12">
                <h1 className="text-4xl md:text-5xl font-black mb-4 text-slate-800">About the Project</h1>
                <p className="text-lg text-slate-600">HealthPredict — Risk Screening Tool v3.0</p>
            </div>

            {/* Research Use Disclaimer */}
            <div className="flex items-start gap-4 bg-amber-50 border border-amber-300 rounded-2xl p-6 mb-10">
                <AlertTriangle className="text-amber-600 w-6 h-6 flex-shrink-0 mt-0.5" />
                <div>
                    <h2 className="text-lg font-bold text-amber-900 mb-1">Research Use Only</h2>
                    <p className="text-amber-800 text-sm leading-relaxed">
                        This tool is a <strong>research and educational prototype</strong>. It is not a regulated medical device,
                        not clinically validated, and must not be used as a substitute for professional medical advice,
                        diagnosis, or treatment. Always consult a qualified healthcare professional for clinical decisions.
                    </p>
                </div>
            </div>

            <GlassCard className="mb-12">
                <h2 className="text-2xl font-bold text-slate-800 mb-4 border-b border-slate-200 pb-2">Project Overview</h2>
                <p className="text-slate-600 leading-relaxed mb-4">
                    HealthPredict is an AI-powered health risk screening platform for research purposes.
                    It uses machine learning models trained on publicly available medical datasets to provide
                    exploratory risk indicators for diabetes and cardiovascular conditions.
                </p>
                <p className="text-slate-600 leading-relaxed">
                    The purpose of this project is to demonstrate the integration of modern web technologies
                    (React 19, Tailwind CSS v4) with data science models (scikit-learn, Python 3) for
                    accessible health informatics research.
                </p>
            </GlassCard>

            <h2 className="text-2xl font-bold text-slate-800 mb-6 text-center">Technology Stack</h2>
            <div className="grid md:grid-cols-2 gap-6 mb-12">
                <GlassCard className="hover:-translate-y-1 transition-transform">
                    <div className="flex items-center gap-4 mb-4">
                        <div className="bg-blue-500/10 p-3 rounded-xl text-blue-600 border border-blue-200">
                            <Component size={24} />
                        </div>
                        <h3 className="text-xl font-bold text-slate-800">Frontend</h3>
                    </div>
                    <ul className="text-slate-600 space-y-2 list-disc list-inside">
                        <li>React 19</li>
                        <li>Vite 7 Build Tool</li>
                        <li>Tailwind CSS v4</li>
                        <li>React Router v7</li>
                    </ul>
                </GlassCard>

                <GlassCard className="hover:-translate-y-1 transition-transform">
                    <div className="flex items-center gap-4 mb-4">
                        <div className="bg-green-500/10 p-3 rounded-xl text-green-600 border border-green-200">
                            <Server size={24} />
                        </div>
                        <h3 className="text-xl font-bold text-slate-800">Backend</h3>
                    </div>
                    <ul className="text-slate-600 space-y-2 list-disc list-inside">
                        <li>Node.js &amp; Express 4</li>
                        <li>RESTful API Architecture</li>
                        <li>Origin-restricted CORS</li>
                        <li>Session-only history (no persistence)</li>
                    </ul>
                </GlassCard>

                <GlassCard className="hover:-translate-y-1 transition-transform md:col-span-2">
                    <div className="flex items-center gap-4 mb-4">
                        <div className="bg-purple-500/10 p-3 rounded-xl text-purple-600 border border-purple-200">
                            <Database size={24} />
                        </div>
                        <h3 className="text-xl font-bold text-slate-800">Machine Learning</h3>
                    </div>
                    <div className="grid md:grid-cols-2 gap-4">
                        <ul className="text-slate-600 space-y-2 list-disc list-inside">
                            <li>Python 3 &amp; Pandas</li>
                            <li>Scikit-Learn</li>
                            <li>PyMuPDF + Pytesseract (OCR)</li>
                        </ul>
                        <ul className="text-slate-600 space-y-2 list-disc list-inside">
                            <li>Logistic Regression</li>
                            <li>StandardScaler Normalization</li>
                            <li>Research-grade models (not clinical)</li>
                        </ul>
                    </div>
                </GlassCard>
            </div>

            <GlassCard className="mb-8">
                <h2 className="text-xl font-bold text-slate-800 mb-3 border-b border-slate-200 pb-2">Privacy</h2>
                <p className="text-slate-600 text-sm leading-relaxed">
                    No personal data is transmitted to external services. Assessment inputs are processed locally
                    and are held only in server memory for the duration of the current session — they are never
                    written to disk or shared with third parties. Uploaded documents are deleted from the server
                    immediately after OCR extraction completes.
                </p>
            </GlassCard>

            <div className="text-center">
                <p className="text-slate-500">Developed by Smit Kansagara &amp; Contributors | © 2026</p>
                <p className="text-xs text-slate-400 mt-1">For research and educational purposes only.</p>
            </div>
        </div>
    );
};

export default About;
