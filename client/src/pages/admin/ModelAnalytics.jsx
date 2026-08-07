import React, { useState, useEffect } from 'react';
import GlassCard from '../../components/GlassCard';
import { getModels } from '../../services/api';
import { Cpu, ShieldCheck, Activity, Heart, CheckCircle2, AlertTriangle, Layers, BarChart2 } from 'lucide-react';

const ModelAnalytics = () => {
    const [models, setModels] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState('');

    useEffect(() => {
        const fetchRegistry = async () => {
            try {
                const data = await getModels();
                setModels(data);
            } catch (err) {
                setError(err.message || 'Failed to load model registry analytics.');
            } finally {
                setLoading(false);
            }
        };
        fetchRegistry();
    }, []);

    if (loading) {
        return (
            <div className="flex items-center justify-center py-20 text-slate-500">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mr-3" />
                <span>Loading model registry &amp; evaluation metrics...</span>
            </div>
        );
    }

    return (
        <div className="max-w-5xl mx-auto animate-fade-in-up pb-12">
            <div className="flex items-center justify-between mb-8">
                <div>
                    <h1 className="text-3xl font-black text-slate-800 flex items-center gap-3">
                        <Cpu className="text-blue-600" size={32} />
                        Model Registry &amp; Analytics
                    </h1>
                    <p className="text-slate-600 text-sm mt-1">Champion model performance, calibration, subgroup fairness, and artifact lineage</p>
                </div>
                <div className="bg-green-50 text-green-800 border border-green-200 px-4 py-2 rounded-xl text-xs font-bold flex items-center gap-2">
                    <CheckCircle2 size={16} /> All Champion Artifacts Active
                </div>
            </div>

            {error && (
                <div className="bg-red-50 border border-red-200 text-red-700 p-4 rounded-xl mb-6 text-sm flex items-center gap-3">
                    <AlertTriangle size={20} />
                    <span>{error}</span>
                </div>
            )}

            <div className="space-y-8">
                {models.map((mod) => {
                    const m = mod.metrics || {};
                    const cm = m.confusion_matrix || {};
                    const sg = m.subgroups || {};

                    return (
                        <GlassCard key={mod.id || mod.versionName} className="hover:border-blue-300 transition-colors">
                            {/* Header Banner */}
                            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-slate-200 pb-4 mb-6">
                                <div className="flex items-center gap-3">
                                    <div className={`p-3 rounded-2xl ${mod.condition === 'DIABETES' ? 'bg-blue-100 text-blue-600' : 'bg-red-100 text-red-600'}`}>
                                        {mod.condition === 'DIABETES' ? <Activity size={24} /> : <Heart size={24} />}
                                    </div>
                                    <div>
                                        <div className="flex items-center gap-2">
                                            <h3 className="text-xl font-bold text-slate-800">{mod.versionName}</h3>
                                            <span className="bg-blue-600 text-white text-[10px] font-bold px-2 py-0.5 rounded-full uppercase tracking-wider">
                                                CHAMPION
                                            </span>
                                        </div>
                                        <p className="text-xs text-slate-500 font-mono mt-0.5">Hash: {mod.artifactHash || 'sha256-verified'}</p>
                                    </div>
                                </div>

                                <div className="text-right text-xs text-slate-500">
                                    <span className="font-bold text-slate-700 block">Registered Artifact URI</span>
                                    <span className="font-mono text-slate-600">{mod.artifactUri}</span>
                                </div>
                            </div>

                            {/* Core Performance Grid */}
                            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
                                <MetricBox label="ROC-AUC Score" value={m.roc_auc ? m.roc_auc.toFixed(4) : 'N/A'} color="text-blue-600" />
                                <MetricBox label="PR-AUC Score" value={m.pr_auc ? m.pr_auc.toFixed(4) : 'N/A'} color="text-purple-600" />
                                <MetricBox label="Brier Calibration" value={m.brier_score ? m.brier_score.toFixed(4) : 'N/A'} color="text-teal-600" subtitle="Lower is better" />
                                <MetricBox label="Balanced Accuracy" value={m.balanced_accuracy ? (m.balanced_accuracy * 100).toFixed(1) + '%' : 'N/A'} color="text-indigo-600" />
                            </div>

                            {/* Detailed Evaluation Panels */}
                            <div className="grid md:grid-cols-2 gap-6">
                                {/* Subgroup Fairness Analysis */}
                                <div className="bg-slate-50 p-4 rounded-xl border border-slate-200">
                                    <h4 className="text-xs font-bold uppercase tracking-wider text-slate-700 mb-3 flex items-center gap-1.5">
                                        <Layers size={15} className="text-blue-600" /> Subgroup Fairness (ROC-AUC)
                                    </h4>
                                    <div className="space-y-2 text-xs">
                                        <div className="flex justify-between p-2 bg-white rounded-lg border border-slate-100">
                                            <span className="text-slate-600">Male Subgroup ROC-AUC:</span>
                                            <span className="font-bold text-slate-800">{sg.male_auc ? sg.male_auc.toFixed(4) : 'N/A'}</span>
                                        </div>
                                        <div className="flex justify-between p-2 bg-white rounded-lg border border-slate-100">
                                            <span className="text-slate-600">Female Subgroup ROC-AUC:</span>
                                            <span className="font-bold text-slate-800">{sg.female_auc ? sg.female_auc.toFixed(4) : 'N/A'}</span>
                                        </div>
                                        <div className="flex justify-between p-2 bg-white rounded-lg border border-slate-100">
                                            <span className="text-slate-600">Age &lt; 50 Subgroup:</span>
                                            <span className="font-bold text-slate-800">{sg.age_under_50_auc ? sg.age_under_50_auc.toFixed(4) : 'N/A'}</span>
                                        </div>
                                        <div className="flex justify-between p-2 bg-white rounded-lg border border-slate-100">
                                            <span className="text-slate-600">Age &ge; 50 Subgroup:</span>
                                            <span className="font-bold text-slate-800">{sg.age_over_50_auc ? sg.age_over_50_auc.toFixed(4) : 'N/A'}</span>
                                        </div>
                                    </div>
                                </div>

                                {/* Confusion Matrix Summary */}
                                <div className="bg-slate-50 p-4 rounded-xl border border-slate-200">
                                    <h4 className="text-xs font-bold uppercase tracking-wider text-slate-700 mb-3 flex items-center gap-1.5">
                                        <BarChart2 size={15} className="text-purple-600" /> Confusion Matrix (Test Set 20%)
                                    </h4>
                                    <div className="grid grid-cols-2 gap-2 text-center text-xs font-mono">
                                        <div className="bg-green-50 border border-green-200 p-2.5 rounded-lg">
                                            <span className="text-[10px] text-green-700 block font-sans font-bold">TRUE NEGATIVE (TN)</span>
                                            <span className="text-base font-bold text-green-900">{cm.tn ? cm.tn.toLocaleString() : 0}</span>
                                        </div>
                                        <div className="bg-amber-50 border border-amber-200 p-2.5 rounded-lg">
                                            <span className="text-[10px] text-amber-700 block font-sans font-bold">FALSE POSITIVE (FP)</span>
                                            <span className="text-base font-bold text-amber-900">{cm.fp ? cm.fp.toLocaleString() : 0}</span>
                                        </div>
                                        <div className="bg-red-50 border border-red-200 p-2.5 rounded-lg">
                                            <span className="text-[10px] text-red-700 block font-sans font-bold">FALSE NEGATIVE (FN)</span>
                                            <span className="text-base font-bold text-red-900">{cm.fn ? cm.fn.toLocaleString() : 0}</span>
                                        </div>
                                        <div className="bg-blue-50 border border-blue-200 p-2.5 rounded-lg">
                                            <span className="text-[10px] text-blue-700 block font-sans font-bold">TRUE POSITIVE (TP)</span>
                                            <span className="text-base font-bold text-blue-900">{cm.tp ? cm.tp.toLocaleString() : 0}</span>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </GlassCard>
                    );
                })}
            </div>
        </div>
    );
};

const MetricBox = ({ label, value, color, subtitle }) => (
    <div className="bg-slate-50 p-4 rounded-xl border border-slate-200 text-center">
        <span className="text-[11px] font-bold text-slate-500 uppercase tracking-wider block mb-1">{label}</span>
        <span className={`text-2xl font-black ${color}`}>{value}</span>
        {subtitle && <span className="text-[10px] text-slate-400 block mt-0.5">{subtitle}</span>}
    </div>
);

export default ModelAnalytics;
