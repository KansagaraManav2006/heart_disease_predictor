import React from 'react';
import { Loader2 } from 'lucide-react';

const Button = ({
  children,
  onClick,
  type = 'button',
  variant = 'primary', // 'primary' | 'ai' | 'secondary' | 'ghost' | 'danger' | 'icon'
  size = 'md', // 'sm' | 'md' | 'lg'
  className = '',
  disabled = false,
  loading = false,
  loadingLabel = null,
  fullWidth = false,
  icon: Icon = null,
  iconPosition = 'left',
  'aria-label': ariaLabel,
  ...props
}) => {
  const baseClasses = 'inline-flex items-center justify-center font-semibold rounded-xl transition-all duration-200 focus-visible:ring-2 focus-visible:ring-teal-400 focus-visible:ring-offset-2 focus-visible:ring-offset-slate-950 active:scale-[0.98] disabled:opacity-50 disabled:cursor-not-allowed disabled:active:scale-100 min-h-[44px] min-w-[44px] select-none';

  const sizeClasses = {
    sm: 'px-3 py-2 text-xs gap-1.5 min-h-[38px]',
    md: 'px-4 py-2.5 text-sm gap-2 min-h-[44px]',
    lg: 'px-6 py-3.5 text-base gap-2.5 min-h-[50px]',
  }[size] || 'px-4 py-2.5 text-sm gap-2 min-h-[44px]';

  const variantClasses = {
    primary: 'bg-teal-600 hover:bg-teal-500 text-slate-950 font-bold border border-teal-400/30 shadow-md shadow-teal-950/40',
    ai: 'bg-amber-500 hover:bg-amber-400 text-slate-950 font-bold border border-amber-300/40 shadow-md shadow-amber-950/40',
    secondary: 'bg-slate-850 hover:bg-slate-800 text-slate-200 border border-slate-700/80 shadow-sm',
    ghost: 'bg-transparent hover:bg-slate-800/80 text-slate-300 border border-transparent shadow-none',
    danger: 'bg-coral-600 hover:bg-coral-500 text-white font-bold shadow-md border border-coral-400/40 shadow-coral-950/40',
    icon: 'p-2.5 bg-slate-850 hover:bg-slate-800 text-slate-200 border border-slate-700/80 rounded-xl w-11 h-11 shadow-sm',
  }[variant] || 'bg-teal-600 hover:bg-teal-500 text-slate-950 font-bold';

  const widthClass = fullWidth ? 'w-full' : '';
  const isDisabled = disabled || loading;

  return (
    <button
      type={type}
      onClick={onClick}
      disabled={isDisabled}
      aria-label={ariaLabel || (typeof children === 'string' ? children : undefined)}
      aria-busy={loading}
      className={`${baseClasses} ${variant !== 'icon' ? sizeClasses : ''} ${variantClasses} ${widthClass} ${className}`}
      {...props}
    >
      {loading ? (
        <>
          <Loader2 className="w-4 h-4 animate-spin text-current" />
          <span>{loadingLabel || (typeof children === 'string' ? children : 'Loading...')}</span>
        </>
      ) : (
        <>
          {Icon && iconPosition === 'left' && <Icon className="w-4 h-4 flex-shrink-0" />}
          {children}
          {Icon && iconPosition === 'right' && <Icon className="w-4 h-4 flex-shrink-0" />}
        </>
      )}
    </button>
  );
};

export default Button;
