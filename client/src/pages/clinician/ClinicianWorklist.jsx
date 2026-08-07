import React, { useState, useEffect } from 'react';
import GlassCard from '../../components/GlassCard';
import { getAssignedPatients } from '../../services/api';
import { Users, AlertTriangle, ShieldCheck, Activity, Heart, Calendar } from 'lucide-react';

const ClinicianWorklist = () => {
    const [patients, setPatients] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState('');

    useEffect(() => {
        const fetchWorklist = async () => {
            try {
                const data = await getAssignedPatients();
                setPatients(data);
            } catch (err) {
                setError(err.message || 'Failed to load clinician worklist.');
            } finally {
                setLoading(false);
            }
        };
        fetchWorklist();
    }, []);

    if (loading) {
        return (
            <div className="flex items-center justify-center py-20 text-slate-500">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mr-3" />
                <span>Loading assigned patient worklist...</span>
            </div>
        );
    }

    return (
        <div className="max-w-5xl mx-auto animate-fade-in-up pb-12">
            <div className="flex items-center justify-between mb-8">
                <div>
                    <h1 className="text-3xl font-black text-slate-800 flex items-center gap-3">
                        <Users className="text-blue-600" size={32} />
                        Clinician Worklist
                    </h1>
                    <p className="text-slate-600 text-sm mt-1">Review assigned patient profiles and cardiometabolic risk assessments</p>
                </div>
                <div className="bg-blue-50 text-blue-800 border border-blue-200 px-4 py-2 rounded-xl text-xs font-bold flex items-center gap-2">
                    <ShieldCheck size={16} /> Verified Clinician Access
                </div>
            </div>

            {error && (
                <div className="bg-red-50 border border-red-200 text-red-700 p-4 rounded-xl mb-6 text-sm flex items-center gap-3">
                    <AlertTriangle size={20} />
                    <span>{error}</span>
                </div>
            )}

            {patients.length === 0 ? (
                <GlassCard className="text-center py-12">
                    <Users className="mx-auto text-slate-400 mb-3" size={48} />
                    <h3 className="text-lg font-bold text-slate-700">No Assigned Patients</h3>
                    <p className="text-sm text-slate-500 max-w-md mx-auto mt-1">
                        Patients can grant you access to review their cardiometabolic risk assessments by entering your clinician email in their patient portal.
                    </p>
                </GlassCard>
            ) : (
                <div className="space-y-6">
                    {patients.map(({ grantId, grantedAt, patient }) => {
                        const latestAssessment = patient.assessments?.[0];
                        return (
                            <GlassCard key={grantId} className="hover:border-blue-300 transition-colors">
                                <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-slate-200 pb-4 mb-4">
                                    <div>
                                        <h3 className="text-xl font-bold text-slate-800">{patient.fullName}</h3>
                                        <p className="text-xs text-slate-500">{patient.user?.email}</p>
                                    </div>
                                    <div className="flex items-center gap-3 text-xs text-slate-500">
                                        <span className="flex items-center gap-1">
                                            <Calendar size={14} /> Granted: {new Date(grantedAt).toLocaleDateString()}
                                        </span>
                                    </div>
                                </div>

                                {latestAssessment ? (
                                    <div className="bg-slate-50 p-4 rounded-xl border border-slate-200 flex items-center justify-between">
                                        <div className="flex items-center gap-3">
                                            {latestAssessment.condition === 'DIABETES' ? (
                                                <Activity className="text-blue-600" size={24} />
                                            ) : (
                                                <Heart className="text-red-600" size={24} />
                                            )}
                                            <div>
                                                <span className="text-xs font-bold uppercase tracking-wider text-slate-500">
                                                    Latest {latestAssessment.condition} Scan
                                                </span>
                                                <p className="text-sm text-slate-700 font-medium">
                                                    Prob: {(latestAssessment.probability * 100).toFixed(1)}% — Version: {latestAssessment.modelVersion}
                                                </p>
                                            </div>
                                        </div>

                                        <span
                                            className={`px-3 py-1 rounded-full text-xs font-bold uppercase ${
                                                latestAssessment.riskBand === 'HIGH'
                                                    ? 'bg-red-100 text-red-800 border border-red-300'
                                                    : latestAssessment.riskBand === 'MODERATE'
                                                    ? 'bg-amber-100 text-amber-800 border border-amber-300'
                                                    : 'bg-green-100 text-green-800 border border-green-300'
                                            }`}
                                        >
                                            {latestAssessment.riskBand} RISK
                                        </span>
                                    </div>
                                ) : (
                                    <p className="text-xs text-slate-400 italic">No assessments recorded yet by this patient.</p>
                                )}
                            </GlassCard>
                        );
                    })}
                </div>
            )}
        </div>
    );
};

export default ClinicianWorklist;
