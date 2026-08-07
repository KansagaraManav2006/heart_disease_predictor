import React from 'react';

const Surface = ({
  children,
  variant = 'flat', // 'flat' | 'raised' | 'glass' | 'hero'
  accent = 'none', // 'none' | 'teal' | 'amber' | 'coral' | 'cyan' | 'violet'
  interactive = false,
  className = '',
  style,
  ...props
}) => {
  const variantClasses = {
    flat: 'bg-slate-900 border border-slate-800 rounded-2xl shadow-sm text-slate-100',
    raised: 'bg-slate-850 border border-slate-700/80 rounded-2xl shadow-md text-slate-100',
    glass: 'bg-slate-900/80 backdrop-blur-xl border border-slate-700/60 rounded-3xl shadow-xl text-slate-100',
    hero: 'bg-slate-900/85 backdrop-blur-2xl border border-slate-700/80 rounded-3xl shadow-2xl text-slate-100 relative overflow-hidden',
  }[variant] || 'bg-slate-900 border border-slate-800 rounded-2xl text-slate-100';

  const accentClasses = {
    none: '',
    teal: 'border-t-2 border-t-teal-500',
    amber: 'border-t-2 border-t-amber-500',
    coral: 'border-t-2 border-t-coral-500',
    cyan: 'border-t-2 border-t-cyan-500',
    violet: 'border-t-2 border-t-violet-500',
  }[accent] || '';

  const interactiveClasses = interactive
    ? 'transition-all duration-200 hover:-translate-y-0.5 hover:border-slate-600 hover:shadow-lg cursor-pointer'
    : '';

  return (
    <div
      className={`${variantClasses} ${accentClasses} ${interactiveClasses} p-6 md:p-8 ${className}`}
      style={style}
      {...props}
    >
      {variant === 'hero' && (
        <div className="absolute top-0 right-0 -mt-10 -mr-10 w-48 h-48 bg-teal-500/10 rounded-full blur-3xl pointer-events-none" />
      )}
      {children}
    </div>
  );
};

export default Surface;
