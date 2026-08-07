import React, { useState, useEffect } from 'react';
import Surface from '../../components/Surface';
import PageHeader from '../../components/PageHeader';
import SearchBar from '../../components/SearchBar';
import Button from '../../components/Button';
import StatusBadge from '../../components/StatusBadge';
import ErrorState from '../../components/ErrorState';
import { queryKnowledge, getKnowledgeDocuments } from '../../services/api';
import { BookOpen, Search, AlertOctagon, ExternalLink, FileText, Sparkles } from 'lucide-react';

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
        setDocuments(docs || []);
      } catch (err) {
        console.error('Failed to load knowledge library:', err);
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
    <div className="space-y-8 animate-fade-in">
      <PageHeader
        title="Evidence-Grounded Guidelines Search"
        subtitle="Search peer-reviewed clinical guidelines from WHO, ADA, ACC/AHA, and CDC with verified citations."
        badge={{ label: 'Guideline Index Active', status: 'healthy' }}
      />

      {/* Single Primary Search Bar Interaction */}
      <Surface variant="flat" accent="teal" className="p-6">
        <form onSubmit={handleSearch} className="space-y-3">
          <label htmlFor="knowledge-search-input" className="block text-xs font-semibold uppercase tracking-wider text-slate-300">
            Query Clinical Guidelines &amp; Diagnostic Thresholds
          </label>

          <div className="flex flex-col md:flex-row gap-3">
            <div className="flex-1">
              <SearchBar
                id="knowledge-search-input"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onClear={() => setQuery('')}
                placeholder="e.g. HbA1c diagnostic thresholds or ASCVD blood pressure target guidelines..."
                ariaLabel="Search medical terminology or guideline topics"
              />
            </div>
            <Button
              type="submit"
              disabled={loading || !query.trim()}
              loading={loading}
              loadingLabel="Querying Guidelines..."
              variant="primary"
              icon={Search}
              className="font-bold px-6"
            >
              Query Guidelines
            </Button>
          </div>
        </form>
      </Surface>

      {error && <ErrorState title="Knowledge Query Failed" message={error} />}

      {/* Emergency Escalation Banner (Coral without continuous pulsing) */}
      {response?.isEmergency && (
        <Surface variant="flat" accent="coral" className="bg-coral-950/40 border-coral-500/50 p-6 text-coral-200">
          <div className="flex items-start gap-4">
            <div className="w-12 h-12 rounded-xl bg-coral-500/20 text-coral-400 border border-coral-500/40 flex items-center justify-center flex-shrink-0">
              <AlertOctagon className="w-6 h-6" />
            </div>
            <div>
              <StatusBadge label="CRITICAL EMERGENCY ALERT" status="high_risk" className="mb-2" />
              <h3 className="text-lg font-bold text-slate-100 mb-2">Immediate Medical Attention Required</h3>
              <p className="text-xs md:text-sm text-slate-200 leading-relaxed mb-4">
                {response.emergencyEscalationMessage}
              </p>
              <div className="bg-slate-950 p-3 rounded-xl border border-coral-500/30 text-xs font-bold text-coral-300 inline-block">
                Immediate Action Required: Call 911 or your local emergency hotline immediately.
              </div>
            </div>
          </div>
        </Surface>
      )}

      {/* Synthesized Response — One Amber Hero Surface */}
      {response && !response.isEmergency && (
        <Surface variant="hero" accent="amber" className="space-y-6">
          <div className="flex items-center justify-between pb-4 border-b border-slate-800">
            <div className="flex items-center gap-2">
              <Sparkles className="w-4 h-4 text-amber-400" />
              <span className="text-xs font-bold text-amber-300 uppercase tracking-wider">
                Generated Explanation &amp; Synthesis
              </span>
            </div>
            <StatusBadge label="Evidence Grounded" status="warning" size="sm" />
          </div>

          <div className="bg-slate-950 p-5 rounded-2xl border border-slate-800 text-xs md:text-sm text-slate-200 leading-relaxed font-medium">
            <span className="text-slate-400 block text-xs mb-2">Target Query: "{response.query}"</span>
            <p className="leading-relaxed">{response.answer}</p>
          </div>

          {/* Explicitly Distinguished Retrieved Evidence Citation Cards */}
          {response.citations && response.citations.length > 0 && (
            <div className="space-y-3 pt-4 border-t border-slate-800">
              <h4 className="text-xs font-bold uppercase tracking-wider text-slate-400 flex items-center gap-2">
                <FileText className="w-4 h-4 text-teal-400" />
                Retrieved Evidence ({response.citations.length} Verified Sources)
              </h4>

              <div className="space-y-3">
                {response.citations.map((cite) => (
                  <div key={cite.id} className="bg-slate-900 p-4 rounded-xl border border-slate-800 space-y-2 text-xs">
                    <div className="flex items-center justify-between gap-2">
                      <span className="font-bold text-slate-100 text-sm">{cite.sourceTitle}</span>
                      <a
                        href={cite.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-xs text-teal-400 hover:text-teal-300 font-semibold underline flex items-center gap-1"
                      >
                        Read Full Guideline Source <ExternalLink className="w-3 h-3" />
                      </a>
                    </div>
                    <div className="text-[11px] text-teal-300 font-medium">
                      {cite.organization} ({cite.publicationYear}) — {cite.section}
                    </div>
                    <p className="text-xs text-slate-400 italic bg-slate-950 p-3 rounded-lg border border-slate-800/80 font-mono leading-relaxed">
                      "{cite.snippet}"
                    </p>
                  </div>
                ))}
              </div>
            </div>
          )}

          <div className="text-[11px] text-slate-400 bg-slate-950 p-3 rounded-xl border border-slate-800">
            <strong className="text-slate-300">Governance Disclaimer:</strong> {response.disclaimer}
          </div>
        </Surface>
      )}

      {/* Flat Citation Library Cards */}
      <div>
        <h3 className="text-sm font-bold text-slate-200 uppercase tracking-wider mb-4 flex items-center gap-2">
          <BookOpen className="w-4 h-4 text-teal-400" /> Verified Reference Library ({documents.length} Guidelines)
        </h3>

        <div className="grid md:grid-cols-2 gap-6">
          {documents.map((doc) => (
            <Surface key={doc.id} variant="flat" className="flex flex-col justify-between p-5">
              <div>
                <span className="text-[11px] font-bold text-teal-400 uppercase tracking-wider block mb-1">
                  {doc.organization}
                </span>
                <h4 className="text-sm font-bold text-slate-100 mb-2">{doc.sourceTitle}</h4>
                <p className="text-xs text-slate-400 line-clamp-3 leading-relaxed mb-4">{doc.snippet}</p>
              </div>

              <div className="pt-3 border-t border-slate-800 flex items-center justify-between text-xs">
                <span className="text-slate-400 font-mono tabular-nums">Year: {doc.publicationYear}</span>
                <a
                  href={doc.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-teal-400 hover:text-teal-300 font-semibold underline flex items-center gap-1"
                >
                  Source Link <ExternalLink className="w-3 h-3" />
                </a>
              </div>
            </Surface>
          ))}
        </div>
      </div>
    </div>
  );
};

export default MedicalKnowledge;
