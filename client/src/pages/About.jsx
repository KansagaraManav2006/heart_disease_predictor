import React from 'react';
import GlassCard from '../components/GlassCard';
import { Code, Database, Server, Component } from 'lucide-react';

const About = () => {
    return (
        <div className="max-w-4xl mx-auto animate-fade-in-up pb-12">
            <div className="text-center mb-12">
                <h1 className="text-4xl md:text-5xl font-black mb-4 text-slate-800">About the Project</h1>
                <p className="text-lg text-slate-600">Healthcare Prediction System v2.0</p>
            </div>

            <GlassCard className="mb-12">
                <h2 className="text-2xl font-bold text-slate-800 mb-4 border-b border-slate-200 pb-2">Project Overview</h2>
                <p className="text-slate-600 leading-relaxed mb-4">
                    The Disease Prediction System is an AI-powered health risk assessment platform.
                    It utilizes Machine Learning algorithms trained on comprehensive medical datasets to provide accurate,
                    instant risk profiling for common chronic conditions like Diabetes and Heart Disease.
                </p>
                <p className="text-slate-600 leading-relaxed">
                    The purpose of this project is to showcase the integration of modern web technologies
                    (React, Tailwind CSS) with data science models (Scikit-learn, Python) to create accessible
                    health informatics tools.
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
                        <li>React 18</li>
                        <li>Vite Build Tool</li>
                        <li>Tailwind CSS (Solid Variables)</li>
                        <li>React Router v6</li>
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
                        <li>Node.js & Express</li>
                        <li>RESTful API Architecture</li>
                        <li>CORS Configuration</li>
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
                            <li>Python 3 & Pandas</li>
                            <li>Scikit-Learn</li>
                            <li>Jupyter Notebooks</li>
                        </ul>
                        <ul className="text-slate-600 space-y-2 list-disc list-inside">
                            <li>Logistic Regression</li>
                            <li>StandardScaler Normalization</li>
                            <li>Pickle Serialization</li>
                        </ul>
                    </div>
                </GlassCard>
            </div>

            <div className="text-center">
                <p className="text-slate-500">Developed by Smit Kansagara | © 2026</p>
            </div>
        </div>
    );
};

export default About;
