import React from 'react';

const Surface = ({
  children,
  variant = 'flat', // 'flat' | 'raised' | 'hero'
  accent = 'none', // 'none' | 'teal' | 'amber' | 'coral'
  className = '',
  style,
  ...props
}) => {
  const variantClasses = {
    flat: 'bg-slate-850/90 border border-slate-800/80 rounded-2xl shadow-sm',
    raised: 'bg-slate-800/90 border border-slate-700/60 rounded-2xl shadow-md shadow-slate-950/40',
    hero: 'bg-slate-800/70 backdrop-blur-md border border-teal-500/30 rounded-2xl shadow-xl shadow-teal-950/20 relative overflow-hidden',
  }[variant] || 'bg-slate-850/90 border border-slate-800/80 rounded-2xl';

  const accentClasses = {
    none: '',
    teal: 'border-t-2 border-t-teal-400',
    amber: 'border-t-2 border-t-amber-400',
    coral: 'border-t-2 border-t-coral-500',
  }[accent] || '';

  return (
    <div
      className={`${variantClasses} ${accentClasses} p-6 md:p-8 ${className}`}
      style={style}
      {...props}
    >
      {variant === 'hero' && (
        <div className="absolute top-0 right-0 -mt-8 -mr-8 w-40 h-40 bg-teal-500/10 rounded-full blur-3xl pointer-events-none" />
      )}
      {children}
    </div>
  );
};

export default Surface;
