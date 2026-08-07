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

  return (
    <div
      className={`${variantClasses} ${accentClasses} p-6 md:p-8 ${className}`}
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
