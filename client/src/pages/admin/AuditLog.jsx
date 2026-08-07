import React, { useState, useEffect } from 'react';
import Surface from '../../components/Surface';
import PageHeader from '../../components/PageHeader';
import SearchBar from '../../components/SearchBar';
import StatusBadge from '../../components/StatusBadge';
import EmptyState from '../../components/EmptyState';
import ErrorState from '../../components/ErrorState';
import { TableSkeleton } from '../../components/Skeleton';
import { getAuditEvents } from '../../services/api';
import { Lock, FileText, Calendar, User, ShieldCheck, Clock } from 'lucide-react';

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
        setError(err.message || 'Failed to load append-only audit trail.');
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

  return (
    <div className="space-y-8 animate-fade-in">
      <PageHeader
        title="Security Audit &amp; Compliance Trail"
        subtitle="Append-only audit trail logging system access, role authorizations, and model assessment operations."
        badge={{ label: 'OWASP ASVS L2', status: 'healthy' }}
      />

      {/* Toolbar Search Bar */}
      <Surface variant="flat" className="p-4 flex flex-col sm:flex-row items-center justify-between gap-4">
        <div className="w-full sm:w-96">
          <SearchBar
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            onClear={() => setSearchTerm('')}
            placeholder="Filter by action (e.g. USER_LOGGED_IN) or email..."
            ariaLabel="Search audit events log"
          />
        </div>

        <div className="text-xs font-mono text-slate-400">
          Showing <span className="text-slate-100 font-bold">{filteredEvents.length}</span> of{' '}
          <span className="text-slate-100 font-bold">{total}</span> events
        </div>
      </Surface>

      {error && <ErrorState title="Audit Trail Error" message={error} />}

      {/* Main Audit Log Content */}
      {loading ? (
        <TableSkeleton rows={5} cols={5} />
      ) : filteredEvents.length === 0 ? (
        <EmptyState
          icon={Lock}
          title="No Audit Events Found"
          description="No security or access events matched your current search parameters."
        />
      ) : (
        <Surface variant="flat" className="p-0 overflow-hidden">
          {/* Desktop Table View */}
          <div className="hidden md:block overflow-x-auto">
            <table className="w-full text-left text-xs" aria-label="System security audit log">
              <thead className="bg-slate-900 text-slate-400 border-b border-slate-800 sticky top-0 font-semibold uppercase tracking-wider">
                <tr>
                  <th className="p-4">Action Event</th>
                  <th className="p-4">Entity Context</th>
                  <th className="p-4">Actor / Role</th>
                  <th className="p-4">Timestamp</th>
                  <th className="p-4">Redacted Metadata</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/80 bg-slate-950/40 font-mono">
                {filteredEvents.map((evt) => (
                  <tr key={evt.id} className="hover:bg-slate-900/60 transition-colors">
                    <td className="p-4 font-sans font-bold text-slate-100 text-sm">
                      <div className="flex items-center gap-2">
                        <Clock className="w-4 h-4 text-teal-400 flex-shrink-0" />
                        <span>{evt.action}</span>
                      </div>
                    </td>
                    <td className="p-4">
                      <StatusBadge label={evt.entityType || 'SYSTEM'} status="secondary" size="sm" />
                    </td>
                    <td className="p-4 font-sans">
                      <div className="text-slate-200 font-medium">{evt.actor?.email || evt.actorId || 'System'}</div>
                      <div className="text-[10px] text-teal-400 font-mono font-bold uppercase">
                        {evt.actor?.role || 'SYSTEM'}
                      </div>
                    </td>
                    <td className="p-4 text-slate-400 text-[11px]">
                      <div>{new Date(evt.createdAt).toLocaleDateString()}</div>
                      <div className="text-slate-500">{new Date(evt.createdAt).toLocaleTimeString()}</div>
                    </td>
                    <td className="p-4">
                      <div className="text-[11px] text-slate-400 bg-slate-900 p-2 rounded-lg border border-slate-800 max-w-xs truncate">
                        {evt.metadata ? JSON.stringify(evt.metadata) : '—'}
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Mobile Cards List View */}
          <div className="md:hidden divide-y divide-slate-800">
            {filteredEvents.map((evt) => (
              <div key={evt.id} className="p-4 space-y-2 text-xs">
                <div className="flex items-center justify-between">
                  <span className="font-bold text-slate-100 text-sm">{evt.action}</span>
                  <StatusBadge label={evt.entityType || 'SYSTEM'} status="secondary" size="sm" />
                </div>
                <div className="text-slate-400 font-medium">
                  Actor: <span className="text-slate-200">{evt.actor?.email || 'System'}</span> ({evt.actor?.role || 'SYSTEM'})
                </div>
                <div className="text-[11px] text-slate-500 font-mono">
                  {new Date(evt.createdAt).toLocaleString()}
                </div>
                {evt.metadata && (
                  <div className="text-[11px] font-mono text-slate-400 bg-slate-900 p-2 rounded-lg border border-slate-800 truncate">
                    {JSON.stringify(evt.metadata)}
                  </div>
                )}
              </div>
            ))}
          </div>
        </Surface>
      )}
    </div>
  );
};

export default AuditLog;
