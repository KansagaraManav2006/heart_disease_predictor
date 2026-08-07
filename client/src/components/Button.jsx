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
  const baseClasses = 'inline-flex items-center justify-center font-semibold rounded-md transition-all duration-200 focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background active:scale-[0.98] disabled:opacity-50 disabled:cursor-not-allowed disabled:active:scale-100 min-h-[44px] min-w-[44px] shadow-sm';

  const sizeClasses = {
    sm: 'px-3 py-1.5 text-xs gap-1.5',
    md: 'px-4 py-2.5 text-sm gap-2',
    lg: 'px-6 py-3.5 text-base gap-2.5',
  }[size] || 'px-4 py-2.5 text-sm gap-2';

  const variantClasses = {
    primary: 'bg-primary hover:bg-red-700 text-primary-foreground border border-red-800/40 shadow-md',
    ai: 'bg-accent hover:bg-amber-600 text-accent-foreground border border-amber-500/40 font-bold',
    secondary: 'bg-card hover:bg-muted text-foreground border border-border hover:border-stone-500 shadow-sm',
    ghost: 'bg-transparent hover:bg-muted text-foreground border border-transparent shadow-none',
    danger: 'bg-destructive hover:bg-red-600 text-destructive-foreground shadow-md border border-red-500/40',
    icon: 'p-2.5 bg-card hover:bg-muted text-foreground border border-border rounded-md shadow-sm',
  }[variant] || 'bg-primary hover:bg-red-700 text-primary-foreground';

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
