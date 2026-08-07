import React, { useState, useEffect } from 'react';
import GlassCard from '../../components/GlassCard';
import { getAuditEvents } from '../../services/api';
import { ShieldCheck, Search, Lock, AlertTriangle, Clock, User, FileText } from 'lucide-react';

const AuditLog = () => {
    const [events, setEvents] = useState([]);
    const [total, setTotal] = useState(0);
    const [searchTerm, setSearchTerm] = useState('');
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState('');

    useEffect(() => {
        const fetchAudit = async () => {
            try {
                const data = await getAuditEvents();
                setEvents(data.events || []);
                setTotal(data.total || 0);
            } catch (err) {
                setError(err.message || 'Failed to load audit events.');
            } finally {
                setLoading(false);
            }
        };
        fetchAudit();
    }, []);

    const filteredEvents = events.filter((evt) => {
        const term = searchTerm.toLowerCase();
        return (
            evt.action?.toLowerCase().includes(term) ||
            evt.entityType?.toLowerCase().includes(term) ||
            evt.actor?.email?.toLowerCase().includes(term)
        );
    });

    if (loading) {
        return (
            <div className="flex items-center justify-center py-20 text-slate-500">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mr-3" />
                <span>Loading security &amp; access audit trail...</span>
            </div>
        );
    }

    return (
        <div className="max-w-5xl mx-auto animate-fade-in-up pb-12">
            {/* Header */}
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-8">
                <div>
                    <h1 className="text-3xl font-black text-slate-800 flex items-center gap-3">
                        <Lock className="text-blue-600" size={32} />
                        Security &amp; Access Audit Trail
                    </h1>
                    <p className="text-slate-600 text-sm mt-1">Append-only audit logging of system access, authorization grants, and data modifications</p>
                </div>
                <div className="bg-blue-50 text-blue-800 border border-blue-200 px-4 py-2 rounded-xl text-xs font-bold flex items-center gap-2">
                    <ShieldCheck size={16} /> OWASP ASVS L2 Verified
                </div>
            </div>

            {/* Filter Box */}
            <GlassCard className="mb-8 border border-slate-200">
                <div className="flex items-center gap-3 bg-white p-3 rounded-xl border border-slate-200">
                    <Search size={18} className="text-slate-400" />
                    <input
                        type="text"
                        value={searchTerm}
                        onChange={(e) => setSearchTerm(e.target.value)}
                        placeholder="Search audit trail by action (e.g. USER_LOGGED_IN, ASSESSMENT_CREATED) or email..."
                        className="w-full bg-transparent text-sm text-slate-800 focus:outline-none"
                    />
                </div>
            </GlassCard>

            {error && (
                <div className="bg-red-50 border border-red-200 text-red-700 p-4 rounded-xl mb-6 text-sm flex items-center gap-3">
                    <AlertTriangle size={20} />
                    <span>{error}</span>
                </div>
            )}

            {/* Audit Log Table / Timeline */}
            <GlassCard>
                <div className="flex items-center justify-between pb-4 mb-4 border-b border-slate-200">
                    <h3 className="text-sm font-bold uppercase tracking-wider text-slate-700 flex items-center gap-2">
                        <FileText size={16} className="text-blue-600" />
                        Audit Events ({filteredEvents.length} of {total})
                    </h3>
                    <span className="text-xs text-slate-500 font-mono">Log Storage: PostgreSQL (Append-Only)</span>
                </div>

                <div className="space-y-3">
                    {filteredEvents.map((evt) => (
                        <div key={evt.id} className="bg-slate-50 p-4 rounded-xl border border-slate-200 flex flex-col md:flex-row md:items-center justify-between gap-3 text-xs">
                            <div className="flex items-start gap-3">
                                <div className="p-2 bg-blue-100 text-blue-700 rounded-lg font-bold mt-0.5">
                                    <Clock size={16} />
                                </div>
                                <div>
                                    <div className="flex items-center gap-2">
                                        <span className="font-bold text-slate-800 text-sm">{evt.action}</span>
                                        <span className="bg-slate-200 text-slate-700 px-2 py-0.5 rounded font-mono text-[10px]">
                                            {evt.entityType}
                                        </span>
                                    </div>
                                    <div className="text-slate-600 mt-1 flex items-center gap-2">
                                        <User size={13} className="text-slate-400" />
                                        <span>Actor: {evt.actor?.email || evt.actorId || 'System'}</span>
                                        <span className="text-slate-400">({evt.actor?.role || 'SYSTEM'})</span>
                                    </div>
                                    {evt.metadata && (
                                        <div className="mt-2 text-[11px] font-mono text-slate-500 bg-white p-2 rounded border border-slate-200">
                                            Metadata: {JSON.stringify(evt.metadata)}
                                        </div>
                                    )}
                                </div>
                            </div>

                            <div className="text-right text-slate-400 font-mono text-[11px] whitespace-nowrap">
                                <div>{new Date(evt.createdAt).toLocaleDateString()}</div>
                                <div>{new Date(evt.createdAt).toLocaleTimeString()}</div>
                                <div className="text-slate-500">IP: {evt.ipAddress || '127.0.0.1'}</div>
                            </div>
                        </div>
                    ))}
                </div>
            </GlassCard>
        </div>
    );
};

export default AuditLog;
