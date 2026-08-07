import React, { useState, useEffect } from 'react';
import Surface from '../../components/Surface';
import PageHeader from '../../components/PageHeader';
import SearchBar from '../../components/SearchBar';
import RiskBadge from '../../components/RiskBadge';
import StatusBadge from '../../components/StatusBadge';
import EmptyState from '../../components/EmptyState';
import ErrorState from '../../components/ErrorState';
import { TableSkeleton } from '../../components/Skeleton';
import Button from '../../components/Button';
import { getAssignedPatients } from '../../services/api';
import { Users, Activity, Heart, Calendar, Eye, Filter } from 'lucide-react';

const ClinicianWorklist = () => {
  const [patients, setPatients] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [searchQuery, setSearchQuery] = useState('');
  const [riskFilter, setRiskFilter] = useState('ALL');

  useEffect(() => {
    const fetchWorklist = async () => {
      try {
        const data = await getAssignedPatients();
        setPatients(data || []);
      } catch (err) {
        setError(err.message || 'Failed to load assigned clinician worklist.');
      } finally {
        setLoading(false);
      }
    };
    fetchWorklist();
  }, []);

  const filteredPatients = patients.filter((item) => {
    const patientName = item.patient?.fullName?.toLowerCase() || '';
    const patientEmail = item.patient?.user?.email?.toLowerCase() || '';
    const query = searchQuery.toLowerCase();
    const matchesSearch = patientName.includes(query) || patientEmail.includes(query);

    const latestRisk = item.patient?.assessments?.[0]?.riskBand || 'LOW';
    const matchesRisk = riskFilter === 'ALL' || latestRisk === riskFilter;

    return matchesSearch && matchesRisk;
  });

  return (
    <div className="space-y-8 animate-fade-in">
      <PageHeader
        title="Clinician Worklist &amp; Patient Triage"
        subtitle="Review assigned patient profiles, active access grants, and cardiometabolic screening stratified by risk."
        badge={{ label: 'Verified Access', status: 'healthy' }}
      />

      {/* Toolbar: Search, Filters & Counter */}
      <Surface variant="flat" className="p-4 flex flex-col md:flex-row items-center justify-between gap-4">
        <div className="w-full md:w-80">
          <SearchBar
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            onClear={() => setSearchQuery('')}
            placeholder="Search patient name or email..."
            ariaLabel="Search worklist by patient name or email"
          />
        </div>

        <div className="flex flex-wrap items-center gap-3 w-full md:w-auto">
          <div className="flex items-center gap-2 text-xs font-semibold text-slate-400">
            <Filter className="w-3.5 h-3.5" />
            <span>Filter Risk:</span>
          </div>

          <div className="flex gap-1 bg-slate-900 p-1 rounded-xl border border-slate-800 text-xs">
            {['ALL', 'HIGH', 'MODERATE', 'LOW'].map((filter) => (
              <button
                key={filter}
                onClick={() => setRiskFilter(filter)}
                className={`px-3 py-1 rounded-lg font-semibold transition-all ${
                  riskFilter === filter
                    ? 'bg-slate-800 text-teal-400 shadow-sm border border-teal-500/30'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                {filter}
              </button>
            ))}
          </div>

          <div className="px-3 py-1 rounded-xl bg-slate-900 border border-slate-800 text-xs font-mono text-slate-300">
            {filteredPatients.length} patient{filteredPatients.length === 1 ? '' : 's'}
          </div>
        </div>
      </Surface>

      {/* Worklist Main Area */}
      {loading ? (
        <TableSkeleton rows={4} cols={5} />
      ) : error ? (
        <ErrorState title="Worklist Service Error" message={error} />
      ) : filteredPatients.length === 0 ? (
        <EmptyState
          icon={Users}
          title="No Assigned Patients Match Criteria"
          description="Patients can grant you access to review their cardiometabolic risk assessments by entering your email in their portal."
        />
      ) : (
        <Surface variant="flat" className="p-0 overflow-hidden">
          {/* Desktop Semantic Table */}
          <div className="hidden md:block overflow-x-auto">
            <table className="w-full text-left text-xs" aria-label="Clinician assigned patient worklist">
              <thead className="bg-slate-900 text-slate-400 border-b border-slate-800 sticky top-0 font-semibold uppercase tracking-wider">
                <tr>
                  <th className="p-4">Patient Name &amp; Contact</th>
                  <th className="p-4">Access Granted</th>
                  <th className="p-4">Latest Condition Assessment</th>
                  <th className="p-4">Probability / Model</th>
                  <th className="p-4">Risk Stratum Priority</th>
                  <th className="p-4 text-right">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/80 bg-slate-950/40">
                {filteredPatients.map(({ grantId, grantedAt, patient }) => {
                  const latest = patient.assessments?.[0];
                  return (
                    <tr
                      key={grantId}
                      className="hover:bg-slate-900/60 transition-colors focus-within:bg-slate-900/80"
                    >
                      <td className="p-4">
                        <div className="font-bold text-slate-100 text-sm">{patient.fullName}</div>
                        <div className="text-slate-400 text-[11px] font-mono">{patient.user?.email}</div>
                      </td>
                      <td className="p-4 text-slate-400">
                        <div className="flex items-center gap-1.5">
                          <Calendar className="w-3.5 h-3.5 text-slate-500" />
                          <span>{new Date(grantedAt).toLocaleDateString()}</span>
                        </div>
                      </td>
                      <td className="p-4">
                        {latest ? (
                          <div className="flex items-center gap-2">
                            {latest.condition === 'DIABETES' ? (
                              <Activity className="w-4 h-4 text-teal-400" />
                            ) : (
                              <Heart className="w-4 h-4 text-violet-400" />
                            )}
                            <span className="font-semibold text-slate-200">
                              {latest.condition} Assessment
                            </span>
                          </div>
                        ) : (
                          <span className="text-slate-500 italic">No Assessment</span>
                        )}
                      </td>
                      <td className="p-4 font-mono text-slate-300">
                        {latest ? (
                          <div>
                            <span className="font-bold">{(latest.probability * 100).toFixed(1)}%</span>
                            <span className="text-slate-500 text-[10px] block">{latest.modelVersion}</span>
                          </div>
                        ) : (
                          '—'
                        )}
                      </td>
                      <td className="p-4">
                        {latest ? (
                          <RiskBadge riskBand={latest.riskBand} size="sm" />
                        ) : (
                          <StatusBadge label="UNTESTED" status="info" size="sm" />
                        )}
                      </td>
                      <td className="p-4 text-right">
                        <Button
                          variant="ghost"
                          size="sm"
                          icon={Eye}
                          aria-label={`View clinical file for ${patient.fullName}`}
                        >
                          View File
                        </Button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          {/* Mobile Card List View */}
          <div className="md:hidden divide-y divide-slate-800">
            {filteredPatients.map(({ grantId, patient }) => {
              const latest = patient.assessments?.[0];
              return (
                <div key={grantId} className="p-4 space-y-3">
                  <div className="flex items-center justify-between">
                    <div>
                      <h4 className="font-bold text-slate-100 text-sm">{patient.fullName}</h4>
                      <p className="text-xs text-slate-400 font-mono">{patient.user?.email}</p>
                    </div>
                    {latest && <RiskBadge riskBand={latest.riskBand} size="sm" />}
                  </div>

                  {latest ? (
                    <div className="bg-slate-900 p-3 rounded-xl border border-slate-800 text-xs flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        {latest.condition === 'DIABETES' ? (
                          <Activity className="w-4 h-4 text-teal-400" />
                        ) : (
                          <Heart className="w-4 h-4 text-violet-400" />
                        )}
                        <span className="font-semibold text-slate-300">{latest.condition}</span>
                      </div>
                      <span className="font-mono font-bold text-slate-200">
                        {(latest.probability * 100).toFixed(1)}%
                      </span>
                    </div>
                  ) : (
                    <p className="text-xs text-slate-500 italic">No assessments submitted yet.</p>
                  )}
                </div>
              );
            })}
          </div>
        </Surface>
      )}
    </div>
  );
};

export default ClinicianWorklist;
