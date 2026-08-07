import React from 'react';

const IconButton = ({
  icon: Icon,
  onClick,
  'aria-label': ariaLabel,
  variant = 'secondary',
  size = 'md',
  disabled = false,
  className = '',
  ...props
}) => {
  if (!ariaLabel) {
    console.warn('IconButton requires an explicit aria-label for accessibility compliance.');
  }

  const sizeClasses = {
    sm: 'w-8 h-8 p-1.5',
    md: 'w-11 h-11 p-2.5',
    lg: 'w-12 h-12 p-3',
  }[size] || 'w-11 h-11 p-2.5';

  const iconSizes = {
    sm: 'w-4 h-4',
    md: 'w-5 h-5',
    lg: 'w-6 h-6',
  }[size] || 'w-5 h-5';

  const variantClasses = {
    secondary: 'bg-slate-800 hover:bg-slate-700 text-slate-300 hover:text-white border border-slate-700/80',
    ghost: 'bg-transparent hover:bg-slate-800/80 text-slate-400 hover:text-white border border-transparent',
    teal: 'bg-teal-600/20 hover:bg-teal-600/30 text-teal-300 border border-teal-500/40',
    coral: 'bg-coral-600/20 hover:bg-coral-600/30 text-coral-300 border border-coral-500/40',
  }[variant] || 'bg-slate-800 hover:bg-slate-700 text-slate-300';

  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      aria-label={ariaLabel}
      className={`inline-flex items-center justify-center rounded-xl transition-all duration-200 focus-visible:ring-2 focus-visible:ring-teal-400 focus-visible:ring-offset-2 focus-visible:ring-offset-slate-950 active:scale-95 disabled:opacity-50 disabled:cursor-not-allowed ${sizeClasses} ${variantClasses} ${className}`}
      {...props}
    >
      {Icon && <Icon className={iconSizes} />}
    </button>
  );
};

export default IconButton;
