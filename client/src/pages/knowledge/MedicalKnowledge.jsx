import React, { useState, useEffect } from 'react';
import GlassCard from '../../components/GlassCard';
import InputField from '../../components/InputField';
import Button from '../../components/Button';
import { queryKnowledge, getKnowledgeDocuments } from '../../services/api';
import { BookOpen, Search, AlertOctagon, ExternalLink, ShieldCheck, FileText } from 'lucide-react';

const MedicalKnowledge = () => {
    const [query, setQuery] = useState('');
    const [response, setResponse] = useState(null);
    const [documents, setDocuments] = useState([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');

    useEffect(() => {
        const fetchDocs = async () => {
            try {
                const docs = await getKnowledgeDocuments();
                setDocuments(docs);
            } catch (err) {
                console.error('Failed to load knowledge documents:', err);
            }
        };
        fetchDocs();
    }, []);

    const handleSearch = async (e) => {
        e.preventDefault();
        if (!query.trim()) return;

        setLoading(true);
        setError('');
        setResponse(null);

        try {
            const res = await queryKnowledge(query);
            setResponse(res);
        } catch (err) {
            setError(err.message || 'Failed to query medical knowledge base.');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="max-w-5xl mx-auto animate-fade-in-up pb-12">
            {/* Page Header */}
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-8">
                <div>
                    <h1 className="text-3xl font-black text-slate-800 flex items-center gap-3">
                        <BookOpen className="text-blue-600" size={32} />
                        Evidence-Grounded Medical Knowledge
                    </h1>
                    <p className="text-slate-600 text-sm mt-1">Search peer-reviewed guidelines from WHO, ADA, ACC/AHA, and CDC with exact citations</p>
                </div>
                <div className="bg-blue-50 text-blue-800 border border-blue-200 px-4 py-2 rounded-xl text-xs font-bold flex items-center gap-2">
                    <ShieldCheck size={16} /> Verified Guideline Sources
                </div>
            </div>

            {/* Search Box */}
            <GlassCard className="mb-8 border-2 border-blue-500/20">
                <form onSubmit={handleSearch} className="space-y-4">
                    <div className="flex flex-col md:flex-row gap-3">
                        <div className="flex-1">
                            <InputField
                                label="Search Medical Terminology or Guideline Topics"
                                name="query"
                                value={query}
                                onChange={(e) => setQuery(e.target.value)}
                                placeholder="e.g. What is the HbA1c threshold for diabetes? or Blood pressure ASCVD guidelines"
                                required
                            />
                        </div>
                        <div className="md:pt-7">
                            <Button
                                type="submit"
                                disabled={loading}
                                className="w-full md:w-auto flex items-center justify-center gap-2 bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 px-6 rounded-xl shadow-md"
                            >
                                <Search size={18} />
                                {loading ? 'Searching Guidelines...' : 'Query Guidelines'}
                            </Button>
                        </div>
                    </div>
                </form>
            </GlassCard>

            {error && (
                <div className="bg-red-50 border border-red-200 text-red-700 p-4 rounded-xl mb-6 text-sm">
                    {error}
                </div>
            )}

            {/* Emergency Escalation Alert Card */}
            {response?.isEmergency && (
                <div className="bg-red-600 text-white p-6 rounded-2xl shadow-xl mb-8 border-2 border-red-700 animate-pulse">
                    <div className="flex items-start gap-4">
                        <AlertOctagon size={36} className="flex-shrink-0 mt-1" />
                        <div>
                            <h3 className="text-xl font-black uppercase tracking-wider mb-2">CRITICAL MEDICAL EMERGENCY ALERT</h3>
                            <p className="text-sm font-medium leading-relaxed mb-4">{response.emergencyEscalationMessage}</p>
                            <div className="bg-white/20 p-3 rounded-xl text-xs font-bold inline-block">
                                Immediate Action Required: Call 911 or your local emergency hotline. Do not wait for website responses.
                            </div>
                        </div>
                    </div>
                </div>
            )}

            {/* Response & Citation Card */}
            {response && !response.isEmergency && (
                <GlassCard className="mb-8 border-l-4 border-l-blue-600">
                    <div className="mb-4 pb-3 border-b border-slate-200">
                        <span className="text-xs font-bold uppercase tracking-wider text-blue-600">Synthesized Guideline Response</span>
                        <h3 className="text-xl font-bold text-slate-800 mt-1">"{response.query}"</h3>
                    </div>

                    <p className="text-sm text-slate-700 leading-relaxed mb-6 bg-slate-50 p-4 rounded-xl border border-slate-200 font-medium">
                        {response.answer}
                    </p>

                    {/* Citations List */}
                    {response.citations && response.citations.length > 0 && (
                        <div className="space-y-3 mb-6">
                            <h4 className="text-xs font-bold uppercase tracking-wider text-slate-500">Verified Citation Sources ({response.citations.length})</h4>
                            {response.citations.map((cite) => (
                                <div key={cite.id} className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm">
                                    <div className="flex items-center justify-between gap-2 mb-1">
                                        <span className="font-bold text-slate-800 text-sm">{cite.sourceTitle}</span>
                                        <a
                                            href={cite.url}
                                            target="_blank"
                                            rel="noopener noreferrer"
                                            className="text-xs text-blue-600 hover:underline font-semibold flex items-center gap-1"
                                        >
                                            Source Link <ExternalLink size={12} />
                                        </a>
                                    </div>
                                    <div className="text-xs font-semibold text-blue-800 mb-2">
                                        {cite.organization} ({cite.publicationYear}) — {cite.section}
                                    </div>
                                    <p className="text-xs text-slate-600 italic bg-slate-50 p-2.5 rounded-lg border border-slate-100 font-mono">
                                        "{cite.snippet}"
                                    </p>
                                </div>
                            ))}
                        </div>
                    )}

                    <div className="text-[11px] text-slate-500 bg-slate-100 p-3 rounded-lg border border-slate-200">
                        <strong>Disclaimer:</strong> {response.disclaimer}
                    </div>
                </GlassCard>
            )}

            {/* Guideline Reference Library Cards */}
            <div>
                <h3 className="text-xl font-bold text-slate-800 mb-4 flex items-center gap-2">
                    <FileText className="text-blue-600" size={22} />
                    Approved Guideline Reference Library ({documents.length})
                </h3>

                <div className="grid md:grid-cols-2 gap-6">
                    {documents.map((doc) => (
                        <GlassCard key={doc.id} className="flex flex-col justify-between">
                            <div>
                                <span className="text-xs font-bold text-blue-600 uppercase tracking-wider block mb-1">
                                    {doc.organization}
                                </span>
                                <h4 className="text-base font-bold text-slate-800 mb-2">{doc.sourceTitle}</h4>
                                <p className="text-xs text-slate-600 line-clamp-3 mb-4">{doc.snippet}</p>
                            </div>

                            <div className="pt-3 border-t border-slate-200 flex items-center justify-between text-xs">
                                <span className="text-slate-500 font-medium">Year: {doc.publicationYear}</span>
                                <a
                                    href={doc.url}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    className="text-blue-600 hover:underline font-bold flex items-center gap-1"
                                >
                                    Read Source <ExternalLink size={12} />
                                </a>
                            </div>
                        </GlassCard>
                    ))}
                </div>
            </div>
        </div>
    );
};

export default MedicalKnowledge;
