import React from 'react';

const Skeleton = ({ className = '', variant = 'text', width, height }) => {
  const variantStyles = {
    text: 'h-4 w-full rounded-md',
    title: 'h-6 w-3/4 rounded-lg',
    card: 'h-32 w-full rounded-2xl',
    avatar: 'h-10 w-10 rounded-full',
    tile: 'h-24 w-full rounded-2xl',
    row: 'h-12 w-full rounded-xl',
  }[variant] || 'h-4 w-full rounded-md';

  const style = {
    ...(width ? { width } : {}),
    ...(height ? { height } : {}),
  };

  return (
    <div
      aria-hidden="true"
      className={`bg-slate-800/80 animate-skeleton border border-slate-700/40 ${variantStyles} ${className}`}
      style={style}
    />
  );
};

export const TableSkeleton = ({ rows = 5, cols = 4 }) => (
  <div className="w-full space-y-3 p-4 bg-slate-850 rounded-2xl border border-slate-800" aria-label="Loading table content">
    <div className="flex gap-4 mb-4">
      {Array.from({ length: cols }).map((_, idx) => (
        <Skeleton key={idx} variant="text" className="h-5 bg-slate-800" />
      ))}
    </div>
    {Array.from({ length: rows }).map((_, rIdx) => (
      <div key={rIdx} className="flex gap-4 py-2 border-t border-slate-800">
        {Array.from({ length: cols }).map((_, cIdx) => (
          <Skeleton key={cIdx} variant="row" className="h-8" />
        ))}
      </div>
    ))}
  </div>
);

export const DashboardSkeleton = () => (
  <div className="space-y-6" aria-label="Loading dashboard metrics">
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
      {Array.from({ length: 4 }).map((_, idx) => (
        <Skeleton key={idx} variant="tile" />
      ))}
    </div>
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <Skeleton variant="card" className="h-64" />
      <Skeleton variant="card" className="h-64" />
    </div>
  </div>
);

export default Skeleton;
