import React from 'react';
import StatusBadge from './StatusBadge';

const PageHeader = ({
  title,
  subtitle = null,
  badge = null, // { label: string, status: string }
  action = null, // React node
  className = '',
}) => {
  return (
    <div className={`flex flex-col md:flex-row md:items-center justify-between gap-4 mb-8 pb-6 border-b border-slate-800/80 ${className}`}>
      <div>
        <div className="flex items-center gap-3 mb-1">
          <h1 className="text-2xl md:text-3xl font-black text-slate-100 tracking-tight">{title}</h1>
          {badge && <StatusBadge label={badge.label} status={badge.status} size="sm" />}
        </div>
        {subtitle && <p className="text-sm text-slate-400 max-w-3xl leading-relaxed">{subtitle}</p>}
      </div>

      {action && <div className="flex items-center gap-3 flex-shrink-0">{action}</div>}
    </div>
  );
};

export default PageHeader;
