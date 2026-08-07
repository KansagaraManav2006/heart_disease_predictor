import React from 'react';

export const Skeleton = ({ className = '', ...props }) => {
  return (
    <div
      className={`bg-slate-800/60 animate-skeleton rounded-xl ${className}`}
      {...props}
    />
  );
};

export const CardSkeleton = () => (
  <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 space-y-4">
    <div className="flex items-center justify-between">
      <Skeleton className="h-5 w-1/3 rounded-lg" />
      <Skeleton className="h-6 w-16 rounded-full" />
    </div>
    <Skeleton className="h-4 w-3/4 rounded-lg" />
    <Skeleton className="h-20 w-full rounded-xl" />
    <div className="flex gap-3 pt-2">
      <Skeleton className="h-10 w-28 rounded-xl" />
      <Skeleton className="h-10 w-28 rounded-xl" />
    </div>
  </div>
);

export const DashboardSkeleton = () => (
  <div className="space-y-6">
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
      {[1, 2, 3, 4].map((i) => (
        <div key={i} className="bg-slate-900 border border-slate-800 rounded-2xl p-5 space-y-3">
          <div className="flex justify-between items-center">
            <Skeleton className="h-4 w-24 rounded-lg" />
            <Skeleton className="h-10 w-10 rounded-xl" />
          </div>
          <Skeleton className="h-8 w-20 rounded-lg" />
          <Skeleton className="h-3 w-32 rounded-lg" />
        </div>
      ))}
    </div>
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <CardSkeleton />
      <CardSkeleton />
    </div>
  </div>
);

export const TableSkeleton = ({ rows = 5, cols = 4 }) => (
  <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 overflow-hidden space-y-3">
    <div className="flex gap-4 pb-2 border-b border-slate-800">
      {Array.from({ length: cols }).map((_, idx) => (
        <Skeleton key={idx} className="h-4 flex-1 rounded-lg" />
      ))}
    </div>
    {Array.from({ length: rows }).map((_, rIdx) => (
      <div key={rIdx} className="flex gap-4 py-2">
        {Array.from({ length: cols }).map((_, cIdx) => (
          <Skeleton key={cIdx} className="h-4 flex-1 rounded-lg" />
        ))}
      </div>
    ))}
  </div>
);

export default Skeleton;
