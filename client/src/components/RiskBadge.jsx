import React from 'react';
import { ShieldCheck, AlertTriangle, ShieldAlert } from 'lucide-react';

const RiskBadge = ({ riskBand = 'LOW', score = null, size = 'md', className = '' }) => {
  const normalizedBand = (riskBand || 'LOW').toUpperCase();

  const isHigh = normalizedBand === 'HIGH' || normalizedBand === 'ELEVATED';
  const isModerate = normalizedBand === 'MODERATE' || normalizedBand === 'MEDIUM';

  const config = isHigh
    ? {
        bg: 'bg-coral-500/15 border-coral-500/30 text-coral-300 font-bold',
        label: 'HIGH RISK',
        Icon: ShieldAlert,
      }
    : isModerate
    ? {
        bg: 'bg-amber-500/15 border-amber-500/30 text-amber-300 font-bold',
        label: 'MODERATE RISK',
        Icon: AlertTriangle,
      }
    : {
        bg: 'bg-teal-500/15 border-teal-500/30 text-teal-300 font-semibold',
        label: 'LOW RISK',
        Icon: ShieldCheck,
      };

  const { Icon } = config;

  const sizeClasses = size === 'lg'
    ? 'px-4 py-1.5 text-sm rounded-full gap-2 font-bold'
    : size === 'sm'
    ? 'px-2.5 py-0.5 text-xs rounded-full gap-1 font-semibold'
    : 'px-3 py-1 text-xs rounded-full gap-1.5 font-semibold';

  return (
    <span className={`inline-flex items-center border ${config.bg} ${sizeClasses} ${className}`}>
      <Icon className={size === 'lg' ? 'w-4 h-4' : 'w-3.5 h-3.5'} />
      <span>{config.label}</span>
      {score !== null && <span className="opacity-80 font-mono tabular-nums">({score}%)</span>}
    </span>
  );
};

export default RiskBadge;
