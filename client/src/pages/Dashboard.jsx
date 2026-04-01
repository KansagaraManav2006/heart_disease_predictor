import React, { useState, useEffect } from 'react';
import { getHistory } from '../services/api';
import GlassCard from '../components/GlassCard';
import { Activity, Heart, Clock, TrendingUp, TrendingDown, AlertCircle } from 'lucide-react';

const Dashboard = () => {
    const [history, setHistory] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState('');

    useEffect(() => {
        const fetchHistory = async () => {
            try {
                const data = await getHistory();
                // Sort by date descending
                const sorted = data.sort((a, b) => new Date(b.date) - new Date(a.date));
                setHistory(sorted);
            } catch (err) {
                console.error("Failed to load history:", err);
                setError('Failed to load patient history.');
            } finally {
                setLoading(false);
            }
        };

        fetchHistory();
    }, []);

    let userId = localStorage.getItem("userId");
    if (!userId) {
        userId = "user_" + Date.now();
        localStorage.setItem("userId", userId);
    }

    const userHistory = history.filter(item => item.userId === userId);
    const diabetesHistory = userHistory.filter(item => item.type === 'diabetes');
    const heartHistory = userHistory.filter(item => item.type === 'heart');

    const renderComparison = (list, metricKey, label, unit = '') => {
        if (list.length < 2) return null;
        const current = list[0].inputs[metricKey];
        const previous = list[1].inputs[metricKey];
        
        if (current === undefined || previous === undefined) return null;

        const diff = Number(current) - Number(previous);
        const isImprovement = diff <= 0; // Assuming lower is better for most metrics (BMI, Glucose, BP)
        const diffText = isImprovement ? '↓ Improved' : '↑ Increase';
        const colorClass = isImprovement ? 'text-green-600 bg-green-50 border-green-200' : 'text-orange-600 bg-orange-50 border-orange-200';

        return (
            <div className="flex items-center justify-between p-3 bg-white rounded-lg border border-slate-100 mb-2 shadow-sm">
                <span className="text-sm font-medium text-slate-600">{label}</span>
                <div className="flex items-center gap-2">
                    <span className="text-sm font-bold text-slate-600">{previous}{unit}</span>
                    <span className="text-slate-400 font-bold">→</span>
                    <span className="font-bold text-slate-800">{current}{unit}</span>
                    <span className={`ml-2 flex items-center text-xs font-bold px-2 py-1 rounded-full border ${colorClass}`}>
                        ({diffText})
                    </span>
                </div>
            </div>
        );
    };

    if (loading) {
        return (
            <div className="flex items-center justify-center min-h-[60vh]">
                <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-600"></div>
            </div>
        );
    }

    return (
        <div className="max-w-5xl mx-auto pb-12 animate-fade-in">
            <div className="mb-8">
                <h1 className="text-3xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-slate-800 to-slate-600 mb-2">
                    Patient Dashboard
                </h1>
                <p className="text-slate-500">
                    Track health metrics and prediction history.
                </p>
            </div>

            {error && (
                <div className="mb-6 p-4 bg-red-50 text-red-600 rounded-xl border border-red-100 flex items-center gap-3">
                    <AlertCircle size={20} />
                    {error}
                </div>
            )}

            <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-8">
                {/* Diabetes Tracking */}
                <GlassCard className="border-t-4 border-t-blue-500/50">
                    <div className="flex items-center gap-3 mb-6 border-b border-borderLight pb-4">
                        <div className="p-2.5 bg-blue-100 text-blue-600 rounded-xl">
                            <Activity size={24} />
                        </div>
                        <h2 className="text-xl font-bold text-slate-800">Diabetes Tracking</h2>
                    </div>

                    {diabetesHistory.length >= 2 ? (
                        <div className="mb-6">
                            <h3 className="text-sm font-bold text-slate-500 uppercase tracking-wider mb-3">Recent Changes</h3>
                            {renderComparison(diabetesHistory, 'glucose', 'Fasting Glucose', ' mg/dL')}
                            {renderComparison(diabetesHistory, 'hba1c', 'HbA1c', '%')}
                            {renderComparison(diabetesHistory, 'bmi', 'BMI')}
                        </div>
                    ) : (
                        <p className="text-sm text-slate-500 mb-6 italic">Not enough data to compare</p>
                    )}

                    <div>
                        <h3 className="text-sm font-bold text-slate-500 uppercase tracking-wider mb-3">History</h3>
                        <div className="space-y-3 max-h-60 overflow-y-auto pr-2 custom-scrollbar">
                            {diabetesHistory.map((item, idx) => (
                                <div key={idx} className="p-3 bg-slate-50 rounded-lg border border-slate-100 flex items-center justify-between">
                                    <div className="flex items-center gap-3">
                                        <Clock size={16} className="text-slate-400" />
                                        <span className="text-sm text-slate-600">{new Date(item.date).toLocaleDateString()}</span>
                                    </div>
                                    <span className={`text-xs font-bold px-2 py-1 rounded-full ${item.prediction === 1 ? 'bg-red-100 text-red-700' : 'bg-green-100 text-green-700'}`}>
                                        {item.prediction === 1 ? 'High Risk' : 'Low Risk'} ({Math.round(item.probability * 100)}%)
                                    </span>
                                </div>
                            ))}
                            {diabetesHistory.length === 0 && <p className="text-sm text-slate-500">No history found.</p>}
                        </div>
                    </div>
                </GlassCard>

                {/* Heart Disease Tracking */}
                <GlassCard className="border-t-4 border-t-red-500/50">
                    <div className="flex items-center gap-3 mb-6 border-b border-borderLight pb-4">
                        <div className="p-2.5 bg-red-100 text-red-600 rounded-xl">
                            <Heart size={24} />
                        </div>
                        <h2 className="text-xl font-bold text-slate-800">Heart Disease Tracking</h2>
                    </div>

                    {heartHistory.length >= 2 ? (
                        <div className="mb-6">
                            <h3 className="text-sm font-bold text-slate-500 uppercase tracking-wider mb-3">Recent Changes</h3>
                            {renderComparison(heartHistory, 'systolic_bp', 'Systolic BP')}
                            {renderComparison(heartHistory, 'cholesterol', 'Cholesterol')}
                        </div>
                    ) : (
                        <p className="text-sm text-slate-500 mb-6 italic">Not enough data to compare</p>
                    )}

                    <div>
                        <h3 className="text-sm font-bold text-slate-500 uppercase tracking-wider mb-3">History</h3>
                        <div className="space-y-3 max-h-60 overflow-y-auto pr-2 custom-scrollbar">
                            {heartHistory.map((item, idx) => (
                                <div key={idx} className="p-3 bg-slate-50 rounded-lg border border-slate-100 flex items-center justify-between">
                                    <div className="flex items-center gap-3">
                                        <Clock size={16} className="text-slate-400" />
                                        <span className="text-sm text-slate-600">{new Date(item.date).toLocaleDateString()}</span>
                                    </div>
                                    <span className={`text-xs font-bold px-2 py-1 rounded-full ${item.prediction === 1 ? 'bg-red-100 text-red-700' : 'bg-green-100 text-green-700'}`}>
                                        {item.prediction === 1 ? 'High Risk' : 'Low Risk'} ({Math.round(item.probability * 100)}%)
                                    </span>
                                </div>
                            ))}
                            {heartHistory.length === 0 && <p className="text-sm text-slate-500">No history found.</p>}
                        </div>
                    </div>
                </GlassCard>
            </div>
        </div>
    );
};

export default Dashboard;
