import React from 'react';
import { ShieldCheck, AlertTriangle, ShieldAlert } from 'lucide-react';

const RiskBadge = ({ riskBand = 'LOW', score = null, size = 'md', className = '' }) => {
  const normalizedBand = (riskBand || 'LOW').toUpperCase();

  const isHigh = normalizedBand === 'HIGH' || normalizedBand === 'ELEVATED';
  const isModerate = normalizedBand === 'MODERATE' || normalizedBand === 'MEDIUM';

  const config = isHigh
    ? {
        bg: 'bg-destructive/20 border-destructive/40 text-destructive-foreground font-bold',
        label: 'HIGH RISK',
        Icon: ShieldAlert,
      }
    : isModerate
    ? {
        bg: 'bg-accent/20 border-accent/40 text-accent-foreground font-bold',
        label: 'MODERATE RISK',
        Icon: AlertTriangle,
      }
    : {
        bg: 'bg-muted text-foreground border-border font-semibold',
        label: 'LOW RISK',
        Icon: ShieldCheck,
      };

  const { Icon } = config;

  const sizeClasses = size === 'lg'
    ? 'px-4 py-2 text-base rounded-md gap-2 font-bold'
    : size === 'sm'
    ? 'px-2 py-0.5 text-xs rounded-md gap-1 font-semibold'
    : 'px-3 py-1.5 text-xs rounded-md gap-1.5 font-semibold';

  return (
    <span className={`inline-flex items-center border ${config.bg} ${sizeClasses} ${className}`}>
      <Icon className={size === 'lg' ? 'w-5 h-5' : 'w-4 h-4'} />
      <span>{config.label}</span>
      {score !== null && <span className="opacity-80 font-mono tabular-nums">({score}%)</span>}
    </span>
  );
};

export default RiskBadge;
