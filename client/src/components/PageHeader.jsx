import React from 'react';
import StatusBadge from './StatusBadge';

const PageHeader = ({ title, subtitle, badge = null, action = null, className = '' }) => {
  return (
    <div className={`flex flex-col md:flex-row md:items-center justify-between gap-4 pb-6 border-b border-slate-800/80 ${className}`}>
      <div className="space-y-1">
        <div className="flex items-center gap-3 flex-wrap">
          <h1 className="text-2xl md:text-3xl font-bold tracking-tight text-slate-100">{title}</h1>
          {badge && <StatusBadge status={badge.status || 'healthy'} label={badge.label} size="sm" />}
        </div>
        {subtitle && <p className="text-xs md:text-sm text-slate-400 max-w-3xl leading-relaxed">{subtitle}</p>}
      </div>

      {action && <div className="flex-shrink-0">{action}</div>}
    </div>
  );
};

export default PageHeader;
