import React from 'react';

const Surface = ({
  children,
  variant = 'flat', // 'flat' | 'raised' | 'hero'
  accent = 'none', // 'none' | 'teal' | 'amber' | 'coral'
  interactive = false,
  className = '',
  style,
  ...props
}) => {
  const variantClasses = {
    flat: 'bg-card text-card-foreground border border-border rounded-md shadow-sm',
    raised: 'bg-muted/60 text-foreground border border-border rounded-md shadow-md',
    hero: 'bg-card text-card-foreground border-2 border-primary/40 rounded-md shadow-xl relative overflow-hidden',
  }[variant] || 'bg-card text-card-foreground border border-border rounded-md';

  const accentClasses = {
    none: '',
    teal: 'border-t-2 border-t-primary',
    amber: 'border-t-2 border-t-amber-600',
    coral: 'border-t-2 border-t-destructive',
  }[accent] || '';

  const interactiveClasses = interactive
    ? 'transition-all duration-200 hover:-translate-y-0.5 hover:border-stone-500 hover:shadow-lg cursor-pointer'
    : '';

  return (
    <div
      className={`${variantClasses} ${accentClasses} ${interactiveClasses} p-6 md:p-8 ${className}`}
      style={style}
      {...props}
    >
      {variant === 'hero' && (
        <div className="absolute top-0 right-0 -mt-8 -mr-8 w-40 h-40 bg-primary/10 rounded-full blur-3xl pointer-events-none" />
      )}
      {children}
    </div>
  );
};

export default Surface;
